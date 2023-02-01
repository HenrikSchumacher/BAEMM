#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>


// TODO: Priority 1:
// TODO: averaging operator
// TODO: mass matrix
// TODO: wrapper
// TODO: internal management of MTL::Buffer (round_up, copy, etc.)

// TODO: Priority 2:
// TODO: single and double layer potential operator for far field.
// TODO: diagonal part of single layer boundary operator
// TODO: evaluate incoming waves on surface -> Dirichlet and Neumann operators.
// TODO: Manage many waves and many wave directions.
// TODO: GMRES on GPU

// TODO: Priority 3:
// TODO: Calderon preconditioner ->local curl operators.

namespace BAEMM
{
    //https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/index.html#//apple_ref/doc/uid/TP40016642-CH27-SW1
    

    class Helmholtz_Metal
    {
    public:
        
        using Int        = uint32_t;
//        using Int        = int;
        using Real       = float;
        using Complex    = std::complex<Real>;
        
        using UInt       = uint32_t;
        
        using Sparse_T = Sparse::MatrixCSR<Complex,Int,Int>;
        
        using NS::StringEncoding::UTF8StringEncoding;
        
        static constexpr Real zero  = 0;
        static constexpr Real one   = 1;
        static constexpr Real two   = 2;
        static constexpr Real three = 3;
        
        static constexpr Real half  = one / two;
        static constexpr Real third = one / three;
        
        static constexpr Real pi      = M_PI;
        static constexpr Real two_pi  = two * pi;
        static constexpr Real four_pi = two * two_pi;
        
//        static constexpr Real one_over_two_pi  = one / two_pi;
        static constexpr Real one_over_four_pi = one / four_pi;
            
        
    private:
        
        MTL::Device* device;

        const int OMP_thread_count = 1;
        
        std::map<std::string, MTL::ComputePipelineState *> pipelines;
        
        MTL::CommandQueue * command_queue;
        
        const Int vertex_count;
        const Int simplex_count;
        
        Tensor2<Real,Int> vertex_coords;
        Tensor2<Int ,Int> triangles;
        
        MTL::Buffer * areas      = nullptr;
        MTL::Buffer * mid_points = nullptr;
        MTL::Buffer * normals    = nullptr;
        
        Sparse_T AvOp;
        Sparse_T AvOpTransp;
        
        MTL::Buffer * B_buf = nullptr;
        MTL::Buffer * C_buf = nullptr;
        
        Complex * restrict B_ptr = nullptr;
        Complex * restrict C_ptr = nullptr;
        
        Int wave_chunk_size    = 16;
        Int wave_count         =  0;
        Int ldB                =  0;
        Int ldC                =  0;
        Int block_size         = 64;
        Int block_count        =  0;
        Int n_rounded          =  0;

        Real c [4][2] = {{}};
        
        bool single_layer = false;
        bool double_layer = false;
        bool adjdbl_layer = false;
        
    public:
        
        template<typename ExtReal,typename ExtInt>
        Helmholtz_Metal(
            MTL::Device * device_,
            ptr<ExtReal> vertex_coords_, ExtInt vertex_count_,
            ptr<ExtInt>  triangles_    , ExtInt simplex_count_,
            int OMP_thread_count_
        )
        :   device           ( device_ )
        ,   OMP_thread_count ( OMP_thread_count_ )
        ,   vertex_count     ( vertex_count_ )
        ,   simplex_count    ( simplex_count_ )
        ,   vertex_coords    ( vertex_coords_, vertex_count_,  3 )
        ,   triangles        ( triangles_,     simplex_count_, 3 )
        {
            tic(ClassName());
            const uint size  =     simplex_count * sizeof(Real);
            const uint size4 = 4 * simplex_count * sizeof(Real);
            
            areas      = device->newBuffer(size,  MTL::ResourceStorageModeManaged);
            mid_points = device->newBuffer(size4, MTL::ResourceStorageModeManaged);
            normals    = device->newBuffer(size4, MTL::ResourceStorageModeManaged);
            
            mut<Real> areas_      = static_cast<Real *>(     areas->contents());
            mut<Real> mid_points_ = static_cast<Real *>(mid_points->contents());
            mut<Real> normals_    = static_cast<Real *>(   normals->contents());

            

            
            Tensor1<Int,Int>     outer ( simplex_count + 1 );
            Tensor1<Int,Int>     inner ( 3 * simplex_count );
            Tensor1<Complex,Int> vals  ( 3 * simplex_count );
            outer[0] = 0;
            
//            mut<Int>     outer = AvOp.Outer().data();
//            mut<Int>     inner = AvOp.Inner().data();
//            mut<Complex> vals  = AvOp.Values().data();
//
            
            // We pad 3-vector with an additional float so that we can use float3 in the metal kernels. (float3 has size 4 * 4 Byte to preserve alignement.)
            #pragma omp parallel for num_threads( OMP_thread_count ) schedule( static )
            for( Int i = 0; i < simplex_count; ++i )
            {   
                Tiny::Vector<4,Real,Int> x;
                Tiny::Vector<4,Real,Int> y;
                Tiny::Vector<4,Real,Int> z;
                Tiny::Vector<4,Real,Int> nu;
                
                Int i_0 = triangles(i,0);
                Int i_1 = triangles(i,1);
                Int i_2 = triangles(i,2);
                
                x[0] = vertex_coords(i_0,0);
                x[1] = vertex_coords(i_0,1);
                x[2] = vertex_coords(i_0,2);
                
                y[0] = vertex_coords(i_1,0);
                y[1] = vertex_coords(i_1,1);
                y[2] = vertex_coords(i_1,2);
                
                z[0] = vertex_coords(i_2,0);
                z[1] = vertex_coords(i_2,1);
                z[2] = vertex_coords(i_2,2);
                
                mid_points_[4*i+0] = third * ( x[0] + y[0] + z[0] );
                mid_points_[4*i+1] = third * ( x[1] + y[1] + z[1] );
                mid_points_[4*i+2] = third * ( x[2] + y[2] + z[2] );
                mid_points_[4*i+3] = zero;
                
                y[0] -= x[0]; y[1] -= x[1]; y[2] -= x[2];
                z[0] -= x[0]; z[1] -= x[1]; z[2] -= x[2];
                
                nu[0] = y[1] * z[2] - y[2] * z[1];
                nu[1] = y[2] * z[0] - y[0] * z[2];
                nu[2] = y[0] * z[1] - y[1] * z[0];

                const Real a = std::sqrt( nu[0] * nu[0] + nu[1] * nu[1] + nu[2] * nu[2] );
                areas_[i] = a;

                outer[i+1]   = 3 * i;
                inner[3*i+0] = i_0;
                inner[3*i+1] = i_1;
                inner[3*i+2] = i_2;
                vals [3*i+0] = a * third;
                vals [3*i+1] = a * third;
                vals [3*i+2] = a * third;
                
                nu /= a;

                normals_[4*i+0] = nu[0];
                normals_[4*i+1] = nu[1];
                normals_[4*i+2] = nu[2];
                normals_[4*i+3] = zero;
            }
            
                 areas->didModifyRange({0,size });
            mid_points->didModifyRange({0,size4});
               normals->didModifyRange({0,size4});
            
            AvOp = Sparse_T(
                std::move(outer), std::move(inner), std::move(vals),
                simplex_count, vertex_count,
                OMP_thread_count
            );
            
            AvOp.SortInner();

            AvOpTransp = AvOp.Transpose();
            
            command_queue = device->newCommandQueue();
            
            if( command_queue == nullptr )
            {
                std::cout << "Failed to find the command queue." << std::endl;
                return;
            }
            
            toc(ClassName());
        }
        
        ~Helmholtz_Metal() = default;

    private:
        
        void CreatePipelineState(
            const std::string & fun_name,       // name of function in code string
            const std::string & fun_fullname,   // name in std::map Pipelines
            const std::string & code,           // string of actual Metal code
            const std::string * param_types,    // types of compile-time parameters (converted to string)
            const std::string * param_names,    // name of compile-time parameters
            const std::string * param_vals,     // values of compile-time parameters (converted to string)
            Int  param_count                   // number of compile-time parameters
        )
        {
            tic("CreatePipeline(" + fun_fullname + ")");
            
            std::stringstream full_code;
            
            // Create compile-time constant. Will be prependend to code string.
            for( Int i = 0; i < param_count; ++i )
            {
                full_code << "constant constexpr " << param_types[i] << " " << param_names[i] << " = " << param_vals[i] <<";\n";
            }
            
            full_code << code;
            
            NS::String * code_NS_String = NS::String::string(full_code.str().c_str(), UTF8StringEncoding);
            
            NS::Error *error = nullptr;
            
            MTL::CompileOptions * opt = MTL::CompileOptions::alloc();
            
            opt->init();
            
            opt->setFastMathEnabled(false);
            
            MTL::Library * lib = device->newLibrary(
                code_NS_String,
                opt, // <-- crucial for distinguishing from the function that loads from file
                &error
            );
            
            if( lib == nullptr )
            {
                std::cout << "Failed to compile library from string for function "
                          << fun_fullname << ", error "
                          << error->description()->utf8String() << std::endl;
                std::exit(-1);
            }
            
            bool found = false;
            
            // Go through all functions in the library to find ours.
            for( NS::UInteger i = 0; i < lib->functionNames()->count(); ++i )
            {
                found = true;
                
                auto name_nsstring = lib->functionNames()->object(i)->description();
                
                if( fun_name == name_nsstring->utf8String() )
                {
                    // This MTL::Function object is needed only temporarily.
                    MTL::Function * fun = lib->newFunction(name_nsstring);
    
                    // Create pipeline from function.
                    pipelines[fun_fullname] = device->newComputePipelineState(fun, &error);
    
                    if( pipelines[fun_fullname] == nullptr )
                    {
                        std::cout << "Failed to created pipeline state object for "
                                  << fun_name << ", error "
                                  << error->description()->utf8String() << std::endl;
                        return;
                    }
                }
            }
            
            if( found )
            {
                print(std::string("CreatePipeline: Found Metal kernel ") + fun_name +".");
            }
            else
            {
                eprint(std::string("CreatePipeline: Did not find Metal kernel ") + fun_name +" in source code.");
            }
            
            toc("CreatePipeline(" + fun_fullname + ")");
        }
        
        
        
//        MTL::ComputePipelineState * pipeline = GetPipelineState(
//              "Helmholtz__Neumann_to_Dirichlet",
//              std::string(
//#include "Neumann_to_Dirichlet.metal"
//              ),
//              {"uint","uint"},
//              {"chunk_size","n_waves"},
//              {chunk_size,n_waves}
//          );
        
        MTL::ComputePipelineState * GetPipelineState(
            const std::string & fun_name,       // name of function in code string
            const std::string & code,           // string of actual Metal code
            const std::vector<std::string> & param_types,    // types of compile-time parameters (converted to string)
            const std::vector<std::string> & param_names,    // name of compile-time parameters
            const std::vector<std::string> & param_vals     // values of compile-time parameters
        )
        {
            std::stringstream fun_fullname_stream;
            
            fun_fullname_stream << fun_name;
            
            for( const auto & s : param_vals )
            {
                fun_fullname_stream << "_" << s;
            }
            
            std::string fun_fullname = fun_fullname_stream.str();
            
            std::string tag = "GetPipelineState(" + fun_fullname + ")";
            
            ptic(tag);
            
            if( pipelines.count(fun_fullname) == 0 )
            {
                std::stringstream full_code;
                
                
                if( param_types.size() != param_names.size() )
                {
                    eprint("CreatePipeline: param_types.size() != param_names.size().");
                    ptoc(tag);
                    return nullptr;
                }
                
                
                if( param_types.size() != param_vals.size() )
                {
                    eprint("CreatePipeline: param_types.size() != param_vals.size().");
                    ptoc(tag);
                    return nullptr;
                }
                
                size_t param_count = param_types.size();
                
                // Create compile-time constant. Will be prependend to code string.
                for( size_t i = 0; i < param_count; ++i )
                {
                    full_code << "constant constexpr " << param_types[i] << " " << param_names[i] << " = " << param_vals[i] <<";\n";
                }
                
                full_code << code;
                
                NS::String * code_NS_String = NS::String::string(full_code.str().c_str(), UTF8StringEncoding);
                
                NS::Error *error = nullptr;
                
                MTL::Library * lib = device->newLibrary(
                    code_NS_String,
                    nullptr, // <-- crucial for distinguishing from the function that loads from file
                    &error
                );
                
                if( lib == nullptr )
                {
                    std::cout << "Failed to compile library from string for function "
                    << fun_fullname << ", error "
                    << error->description()->utf8String() << std::endl;
//                    std::exit(-1);
                    
                    return nullptr;
                }
                
                bool found = false;
                
                // Go through all functions in the library to find ours.
                for( NS::UInteger i = 0; i < lib->functionNames()->count(); ++i )
                {
                    found = true;
                    
                    auto name_nsstring = lib->functionNames()->object(i)->description();
                    
                    if( fun_name == name_nsstring->utf8String() )
                    {
                        // This MTL::Function object is needed only temporarily.
                        MTL::Function * fun = lib->newFunction(name_nsstring);
                        
                        // Create pipeline from function.
                        pipelines[fun_fullname] = device->newComputePipelineState(fun, &error);
                        
                        if( pipelines[fun_fullname] == nullptr )
                        {
                            std::cout << "Failed to created pipeline state object for "
                            << fun_name << ", error "
                            << error->description()->utf8String() << std::endl;
                            return nullptr;
                        }
                    }
                }
                
                if( found )
                {
//                    print(std::string("CreatePipeline: Found Metal kernel ") + fun_name +".");
                    ptoc(tag);
                    return pipelines[fun_fullname];
                }
                else
                {
                    eprint(std::string("CreatePipeline: Did not find Metal kernel ") + fun_name +" in source code.");
                    ptoc(tag);
                    return nullptr;
                }
            }
            else
            {
                ptoc(tag);
                return pipelines[fun_fullname];
            }
        }

        
    public:
        
        void LoadCoefficients( const std::array<Complex,4> & coeff )
        {
            // We have to process the coefficients anyways.
            // Hence we can already multiply by one_over_four_pi so that the kernels don't have to do that each time. (The performance gain is not measureable, though.)
            
            SetMassMatrixCoefficient (coeff[0]);
            SetSingleLayerCoefficient(coeff[1]);
            SetDoubleLayerCoefficient(coeff[2]);
            SetAdjDblLayerCoefficient(coeff[3]);
        }

        Complex GetMassMatrixCoefficient() const
        {
            return Complex( c[0][0], c[0][1] );
        }
        
        
        void SetMassMatrixCoefficient( const Complex & z )
        {
            c[0][0] = real(z);
            c[0][1] = imag(z);
        }
        
        Complex GetSingleLayerCoefficient() const
        {
            return Complex( c[1][0] * four_pi, c[1][1] * four_pi );
        }
        
        
        void SetSingleLayerCoefficient( const Complex & z )
        {
            c[1][0] = real(z) * one_over_four_pi;
            c[1][1] = imag(z) * one_over_four_pi;
            single_layer = (c[1][0] != zero) || (c[1][1] != zero);
        }
        
        Complex GetDoubleLayerCoefficient() const
        {
            return Complex( c[2][0] * four_pi, c[2][1] * four_pi );
        }
        
        void SetDoubleLayerCoefficient( const Complex & z )
        {
            c[2][0] = real(z) * one_over_four_pi;
            c[2][1] = imag(z) * one_over_four_pi;
            double_layer = (c[2][0] != zero) || (c[2][1] != zero);
        }
        
        Complex GetAdjDblLayerCoefficient() const
        {
            return Complex( c[3][0] * four_pi, c[3][1] * four_pi );
        }
        
        void SetAdjDblLayerCoefficient( const Complex & z )
        {
            c[3][0] = real(z) * one_over_four_pi;
            c[3][1] = imag(z) * one_over_four_pi;
            adjdbl_layer = (c[3][0] != zero) || (c[3][1] != zero);
        }
        
        void RequireBuffers( const Int wave_count_, const Int block_size_, const Int wave_chunk_size_ )
        {
            const Int new_ld          = RoundUpTo( wave_count_, wave_chunk_size_ );
            const Int new_block_count = DivideRoundUp(simplex_count, block_size_ );
            const Int new_n_rounded   = new_block_count * block_size_;
            const Int new_size        = new_n_rounded * new_ld;
            
            if( new_size > ldB * n_rounded )
            {
                print("Reallocating size "+ToString(new_size) );
                B_buf = device->newBuffer(new_size * sizeof(Complex), MTL::ResourceStorageModeManaged);
                C_buf = device->newBuffer(new_size * sizeof(Complex), MTL::ResourceStorageModeManaged);
                
                B_ptr = reinterpret_cast<Complex *>(B_buf->contents());
                C_ptr = reinterpret_cast<Complex *>(C_buf->contents());
            }
            
            wave_count      = wave_count_;
            wave_chunk_size = wave_chunk_size_;
            ldB = ldC       = new_ld;
            block_size      = block_size_;
            block_count     = new_block_count;
            n_rounded       = new_n_rounded;
        }
        
        void RequireBuffers( const Int wave_count_ )
        {
            RequireBuffers( wave_count_, block_size, wave_chunk_size );
        }

//        Complex * B_ptr()
//        {
//            return reinterpret_cast<Complex *>(B_buf->contents());
//        }
//
//        const Complex * B_ptr() const
//        {
//            return reinterpret_cast<Complex *>(B_buf->contents());
//        }
        
        Complex & B( const Int i, const Int k )
        {
            return B_ptr[ldB * i + k];
        }
        
        const Complex & B( const Int i, const Int k ) const
        {
            return B_ptr[ldB * i + k];
        }
        
//        Complex * C_ptr()
//        {
//            return reinterpret_cast<Complex *>(C_buf->contents());
//        }
//
//        const Complex * C_ptr() const
//        {
//            return reinterpret_cast<Complex *>(C_buf->contents());
//        }
        
        Complex & C( const Int i, const Int k )
        {
            return C_ptr[ldC * i + k];
        }
        
        const Complex & C( const Int i, const Int k ) const
        {
            return C_ptr[ldC * i + k];
        }
        
        void ReadB( ptr<Complex> input, const Int ld_input, const Int wave_count_ )
        {
//            tic("ReadB");
            RequireBuffers( wave_count_ );

            #pragma omp parallel for num_threads( OMP_thread_count )
            for( Int i = 0; i < simplex_count; ++i )
            {
                copy_buffer( &input[ld_input * i], &B_ptr[ldB * i], wave_count );
            }
            
            B_buf->didModifyRange({0,simplex_count * ldB * sizeof(Complex)});
            
//            toc("ReadB");
        }
        
        void ReadB( ptr<Complex> input, const Int wave_count_ )
        {
            ReadB( input, wave_count_, wave_count_ );
        }
        
        void WriteB( mut<Complex> output, const Int ld_output )
        {
            Complex * B_ = reinterpret_cast<Complex *>(B_buf->contents());
            
            #pragma omp parallel for num_threads( OMP_thread_count )
            for( Int i = 0; i < simplex_count; ++i )
            {
                copy_buffer( &B_[ldB * i], &output[ld_output * i], wave_count );
            }
        }
        
        void ReadC( ptr<Complex> input, const Int ld_input, const Int wave_count_ )
        {
//            tic("ReadC");
            RequireBuffers( wave_count_ );
            
            #pragma omp parallel for num_threads( OMP_thread_count )
            for( Int i = 0; i < simplex_count; ++i )
            {
                copy_buffer( &input[ld_input * i], &C_ptr[ldC * i], wave_count );
            }
            
            C_buf->didModifyRange({0,simplex_count * ldC * sizeof(Complex)});
            
//            toc("ReadC");
        }
        
        void ReadC( ptr<Complex> input, const Int wave_count_ )
        {
            ReadB( input, wave_count_, wave_count_ );
        }
        
        void WriteC( mut<Complex> output, const Int ld_output )
        {
//            tic("WriteC");
            
            #pragma omp parallel for num_threads( OMP_thread_count )
            for( Int i = 0; i < simplex_count; ++i )
            {
                copy_buffer( &C_ptr[ldC * i], &output[ld_output * i], wave_count );
            }
            
//            toc("WriteC");
        }
        
    public:
        
        void ApplyBoundaryOperators_PL(
            const Complex & alpha, ptr<Complex> B_in,  const Int ldB_in,
            const Complex & beta,  mut<Complex> C_out, const Int ldC_out,
            const std::vector<Real>      & kappa,
            const std::array <Complex,4> & coeff,
            const Int wave_count_
        )
        {
            tic(ClassName()+"::ApplyBoundaryOperators_PL");
            // TODO: Aim is to implement the following:
            //
            // Computes
            //
            //     C_out = alpha * A * B_in + beta * C_out,
            //
            // where B_in and C_out out are matrices of size vertex_count x wave_count_ and
            // represent the vertex values of  wave_count_ piecewise-linear functions.
            // The operator A is a linear combination of several operators:
            //
            // A =   coeff[0] * MassMatrix
            //     + coeff[1] * SingleLayerOp
            //     + coeff[2] * DoubleLayerOp
            //     + coeff[3] * AdjDblLayerOp
            
            // TODO: Explain how kappa is distributed over this data!
            
            if( kappa.size() != wave_count_ / wave_chunk_size )
            {
                eprint(ClassName()+"::ApplyBoundaryOperators_PL: kappa.size() != wave_count_ / wave_chuck_size.");
            }
            
            // TODO: Apply multiple of mass matrix.
            // TODO: Adjust coeffients!
            print("a");
            LoadCoefficients(coeff);
            print("b");
            RequireBuffers( wave_count_, block_size, wave_chunk_size );
            print("c");
//            AvOp.Dot( Complex(1), B_in, ldB_in, Complex(0), B_ptr, ldB, wave_count_ );
//            print("d");
//            B_buf->didModifyRange({0, n_rounded * ldB});
//            
////            BoundaryOperatorKernel_C( kappa );
//            
//            // TODO: Apply diagonal part of single layer boundary operator.
//            
//            // TODO: Is there some diagonal part of double layer and adjdbl boundary operator?
//            print("e");
//            AvOpTransp.Dot( alpha, C_ptr, ldC, beta, C_out, ldC_out, wave_count_ );
//            print("f");
            
            toc(ClassName()+"::ApplyBoundaryOperators_PL");
        }
        
//#include "Neumann_to_Dirichlet.hpp"
//
//#include "Neumann_to_Dirichlet2.hpp"
//
//#include "Neumann_to_Dirichlet3.hpp"
//
//#include "Neumann_to_Dirichlet4.hpp"
        
#include "Helmholtz_Metal/BoundaryOperatorKernel_C.hpp"
        
//#include "Helmholtz_Metal/BoundaryOperatorKernel_ReIm.hpp"
        
        
//#include "GEMM2.hpp"
        
    public:
        
        std::string ClassName() const
        {
            return "Helmholtz_Metal";
        }
        
    };
        
} // namespace BAEMM
