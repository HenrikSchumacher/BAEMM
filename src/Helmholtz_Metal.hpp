#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>


// TODO: Priority 1:
// TODO: Debug wrapper
// TODO: diagonal part of single layer boundary operator

// TODO: Priority 2:
// TODO: single and double layer potential operator for far field.
// TODO: evaluate incoming waves on surface -> Dirichlet and Neumann operators.
// TODO: Manage many waves and many wave directions.
// TODO: GMRES on GPU

// TODO: Priority 3:
// TODO: Calderon preconditioner ->local curl operators.


// DONE: averaging operator
// DONE: internal management of MTL::Buffer (round_up, copy, etc.)
// DONE: mass matrix
// DONE: wrapper

namespace BAEMM
{
    //https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/index.html#//apple_ref/doc/uid/TP40016642-CH27-SW1
    

    class Helmholtz_Metal
    {
    public:
        
//        using Int        = uint32_t;
        using LInt       = std::size_t;
        using Int        = int;
        using Real       = float;
        using Complex    = std::complex<Real>;
        
        using UInt       = uint32_t;
        
        using Sparse_T = Sparse::MatrixCSR<Complex,Int,LInt>;
        
        using NS::StringEncoding::UTF8StringEncoding;
        
        static constexpr Real zero  = 0;
        static constexpr Real one   = 1;
        static constexpr Real two   = 2;
        static constexpr Real three = 3;
        
        static constexpr Real half     = one / two;
        static constexpr Real third    = one / three;
        static constexpr Real sixth    = half * third;
        static constexpr Real twelveth = half * sixth;
        
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
        
        Sparse_T Mass;
        
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
            Initialize();
        }
        
        ~Helmholtz_Metal() = default;

#include "Helmholtz_Metal/Initialize.hpp"
        
#include "Helmholtz_Metal/GetPipelineState.hpp"
        
#include "Helmholtz_Metal/InputOutput.hpp"
        
        
#include "Helmholtz_Metal/BoundaryOperatorKernel_C.hpp"
        
//#include "Helmholtz_Metal/BoundaryOperatorKernel_ReIm.hpp"

    public:
        
        Int VertexCount() const
        {
            return vertex_count;
        }
        
        Int SimplexCount() const
        {
            return simplex_count;
        }
        
        const Sparse_T & MassMatrix() const
        {
            return Mass;
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

            
            // TODO: Adjust coeffients!

            LoadCoefficients(coeff);

            RequireBuffers( wave_count_, block_size, wave_chunk_size );

            AvOp.Dot( Complex(1), B_in, ldB_in, Complex(0), B_ptr, ldB, wave_count_ );
            
            BoundaryOperatorKernel_C( kappa );
            
            // TODO: Apply diagonal part of single layer boundary operator.
            
            // TODO: Is there some diagonal part of double layer and adjdbl boundary operator?

            AvOpTransp.Dot( alpha, C_ptr, ldC, beta, C_out, ldC_out, wave_count_ );
            
            Mass.Dot(
                alpha * Complex(c[0][0],c[0][1]), B_in, ldB_in, Complex(1), C_out, ldC_out, wave_count_
            );
            
            toc(ClassName()+"::ApplyBoundaryOperators_PL");
        }
        
    public:
        
        std::string ClassName() const
        {
            return "Helmholtz_Metal";
        }
        
    };
        
} // namespace BAEMM
