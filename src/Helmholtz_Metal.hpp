#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>


// TODO: Priority 1:
// TODO: diagonal part of single layer boundary operator
// TODO: single and double layer potential operator for far field.
// TODO: averaging operator
// TODO: mass matrix
// TODO: wrapper
// TODO: internal management of MTL::Buffer (round_up, copy, etc.)

// TODO: Priority 2:
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
        using Real       = float;
        using Complex    = std::complex<Real>;
        
        using UInt        = uint32_t;
        
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

        std::map<std::string, MTL::ComputePipelineState * > pipelines;
        
        MTL::CommandQueue * command_queue;
        
        const Int m;
        const Int n;
        
        Tensor2<Real,Int> vertex_coords;
        Tensor2<Int ,Int> triangles;
        
        MTL::Buffer * areas;
        MTL::Buffer * mid_points;
        MTL::Buffer * normals;
        
        float coeff_over_four_pi [4][2] = {{}};
        
    public:
        
        template<typename ExtReal,typename ExtInt>
        Helmholtz_Metal(
            MTL::Device * device_,
            ptr<ExtReal> vertex_coords_, ExtInt vertex_count_,
            ptr<ExtInt>  triangles_    , ExtInt simplex_count_
        )
        :   device( device_ )
        ,   m             ( vertex_count_ )
        ,   n             ( simplex_count_ )
        ,   vertex_coords ( vertex_coords_, vertex_count_,  3 )
        ,   triangles     ( triangles_,     simplex_count_, 3 )
        {
            tic(ClassName());
            const Int size  =     n * sizeof(Real);
            const Int size4 = 4 * n * sizeof(Real);
            
            areas      = device->newBuffer(size,  MTL::ResourceStorageModeManaged);
            mid_points = device->newBuffer(size4, MTL::ResourceStorageModeManaged);
            normals    = device->newBuffer(size4, MTL::ResourceStorageModeManaged);
            
            mut<Real> areas_      = static_cast<Real *>(     areas->contents());
            mut<Real> mid_points_ = static_cast<Real *>(mid_points->contents());
            mut<Real> normals_    = static_cast<Real *>(   normals->contents());
            
            Tiny::Vector<4,Real,Int> x;
            Tiny::Vector<4,Real,Int> y;
            Tiny::Vector<4,Real,Int> z;
            Tiny::Vector<4,Real,Int> nu;
            
            // We pad 3-vector with an additional float so that we can use float3 in the metal kernels. (float3 has size 4 * 4 Byte to preserve alignement.)
            for( Int i = 0; i < n; ++i )
            {
                x[0] = vertex_coords(triangles(i,0),0);
                x[1] = vertex_coords(triangles(i,0),1);
                x[2] = vertex_coords(triangles(i,0),2);
                
                y[0] = vertex_coords(triangles(i,1),0);
                y[1] = vertex_coords(triangles(i,1),1);
                y[2] = vertex_coords(triangles(i,1),2);
                
                z[0] = vertex_coords(triangles(i,2),0);
                z[1] = vertex_coords(triangles(i,2),1);
                z[2] = vertex_coords(triangles(i,2),2);
                
                mid_points_[4*i+0] = third * ( x[0] + y[0] + z[0] );
                mid_points_[4*i+1] = third * ( x[1] + y[1] + z[1] );
                mid_points_[4*i+2] = third * ( x[2] + y[2] + z[2] );
                mid_points_[4*i+3] = 0;
                
                y[0] -= x[0]; y[1] -= x[1]; y[2] -= x[2];
                z[0] -= x[0]; z[1] -= x[1]; z[2] -= x[2];
                
                nu[0] = y[1] * z[2] - y[2] * z[1];
                nu[1] = y[2] * z[0] - y[0] * z[2];
                nu[2] = y[0] * z[1] - y[1] * z[0];

                const Real a = std::sqrt( nu[0] * nu[0] + nu[1] * nu[1] + nu[2] * nu[2] );
                areas_[i] = a;

                nu /= a;

                normals_[4*i+0] = nu[0];
                normals_[4*i+1] = nu[1];
                normals_[4*i+2] = nu[2];
                normals_[4*i+3] = zero;
            }
            
                 areas->didModifyRange({0,size });
            mid_points->didModifyRange({0,size4});
               normals->didModifyRange({0,size4});

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
        
        void LoadCoefficients( const std::array<Complex,3> & coeff )
        {
            // We have to process the coefficients anyways.
            // Hence we can already multiply by one_over_four_pi so that the kernels don't have to do that each time. (The performance gain is not measureable, though.)
            
            SetSingleLayerCoefficient(coeff[0]);
            SetDoubleLayerCoefficient(coeff[1]);
            SetAdjDblLayerCoefficient(coeff[2]);
        }
        
        Complex GetSingleLayerCoefficient() const
        {
            return Complex( coeff_over_four_pi[0][0] * four_pi, coeff_over_four_pi[0][1] * four_pi);
        }
        
        
        void SetSingleLayerCoefficient( const Complex & z )
        {
            coeff_over_four_pi[0][0] = real(z) * one_over_four_pi;
            coeff_over_four_pi[0][1] = imag(z) * one_over_four_pi;
        }
        
        Complex GetDoubleLayerCoefficient() const
        {
            return Complex( coeff_over_four_pi[1][0] * four_pi, coeff_over_four_pi[1][1] * four_pi);
        }
        
        void SetDoubleLayerCoefficient( const Complex & z )
        {
            coeff_over_four_pi[1][0] = real(z) * one_over_four_pi;
            coeff_over_four_pi[1][1] = imag(z) * one_over_four_pi;
        }
        
        Complex GetAdjDblLayerCoefficient() const
        {
            return Complex( coeff_over_four_pi[3][0] * four_pi, coeff_over_four_pi[3][1] * four_pi);
        }
        
        void SetAdjDblLayerCoefficient( const Complex & z )
        {
            coeff_over_four_pi[2][0] = real(z) * one_over_four_pi;
            coeff_over_four_pi[2][1] = imag(z) * one_over_four_pi;
        }
        
        
//#include "Neumann_to_Dirichlet.hpp"
//
//#include "Neumann_to_Dirichlet2.hpp"
//
//#include "Neumann_to_Dirichlet3.hpp"
//
//#include "Neumann_to_Dirichlet4.hpp"
        
#include "BoundaryOperatorKernel_C.hpp"
        
#include "BoundaryOperatorKernel_ReIm.hpp"
        
        
//#include "GEMM2.hpp"
        
    public:
        
        std::string ClassName() const
        {
            return "Helmholtz_Metal";
        }
        
    };
        
} // namespace BAEMM
