#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "MatrixPanelRowMajor.hpp"

namespace HeavyMetal
{
    //https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/index.html#//apple_ref/doc/uid/TP40016642-CH27-SW1
    
    using namespace Tools;

    class BLAS_Capella
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
        
        std::map<std::string, MTL::ComputePipelineState *> pipelines;
        
        MTL::CommandQueue * command_queue;
        
    public:
        
        BLAS_Capella() = delete;
        
        BLAS_Capella( MTL::Device* device_ )
        :   device( device_ )
        {
            print("");
            command_queue = device->newCommandQueue();
            
            if( command_queue == nullptr )
            {
                std::cout << "Failed to find the command queue." << std::endl;
                return;
            }
            print("");
        }
        
        ~BLAS_Capella() = default;
        
#include "Helmholtz_Metal/GetPipelineState.hpp"
        
    public:
        
        void GEMM_CM_Ref(
            const uint M,
            const uint N,
            const uint K,
            const float   alpha,
            MTL::Buffer * A,
            MTL::Buffer * B,
            const float   beta,
            MTL::Buffer * C
        )
        {
            cblas_sgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                alpha, reinterpret_cast<float*>(A->contents()), M,
                       reinterpret_cast<float*>(B->contents()), K,
                beta,  reinterpret_cast<float*>(C->contents()), M
            );
        }
        
        void GEMM_CM_C_Ref(
            const uint M,
            const uint N,
            const uint K,
            const std::complex<float> alpha,
            MTL::Buffer * A,
            MTL::Buffer * B,
            const std::complex<float> beta,
            MTL::Buffer * C
        )
        {
            cblas_zgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                &alpha, reinterpret_cast<std::complex<float>*>(A->contents()), M,
                        reinterpret_cast<std::complex<float>*>(B->contents()), K,
                &beta,  reinterpret_cast<std::complex<float>*>(C->contents()), M
            );
        }
        
        void GEMM_RM_Ref(
            const uint M,
            const uint N,
            const uint K,
            const float   alpha,
            MTL::Buffer * A,
            MTL::Buffer * B,
            const float   beta,
            MTL::Buffer * C
        )
        {
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                alpha, reinterpret_cast<float*>(A->contents()), K,
                       reinterpret_cast<float*>(B->contents()), N,
                beta,  reinterpret_cast<float*>(C->contents()), N
            );
        }
        
        void GEMM_RM_Naive(
            const uint M,
            const uint N,
            const uint K,
            const float   alpha,
            MTL::Buffer * A,
            MTL::Buffer * B,
            const float   beta,
            MTL::Buffer * C
        )
        {
            ptr<float> a = reinterpret_cast<float*>(A->contents());
            ptr<float> b = reinterpret_cast<float*>(B->contents());
            mut<float> c = reinterpret_cast<float*>(C->contents());
            
            zerofy_buffer(c,M*N,8);
            
            #pragma omp parallel for num_threads( 8 )
            for( uint k = 0; k < K; ++k )
            {
                for( uint i = 0; i < M; ++i )
                {
                    for( uint j = 0; j < N; ++j )
                    {
                        c[N*i+j]+= a[K*i+k] * b[N*k+j];
                    }
                }
            }
        }
        
        
        void GEMM_CM_Naive(
            const uint M,
            const uint N,
            const uint K,
            const float   alpha,
            MTL::Buffer * A,
            MTL::Buffer * B,
            const float   beta,
            MTL::Buffer * C
        )
        {
            ptr<float> a = reinterpret_cast<float*>(A->contents());
            ptr<float> b = reinterpret_cast<float*>(B->contents());
            mut<float> c = reinterpret_cast<float*>(C->contents());
            
            zerofy_buffer(c,M*N,8);
            
        #pragma omp parallel for num_threads( 8 )
            for( uint k = 0; k < K; ++k )
            {
                for( uint i = 0; i < M; ++i )
                {
                    for( uint j = 0; j < N; ++j )
                    {
                        c[M*j+i]+= a[M*k+i] * b[K*j+k];
                    }
                }
            }
        }
        
//#include "AddReduce/AddReduce.hpp"

//#include "GEMM/GEMM_CM.hpp"
        
#include "GEMM/GEMM_CM_C.hpp"
        
//#include "GEMM/GEMM_RM_NVidea.hpp"
//
//#include "GEMM/GEMM_RM.hpp"
        

        
//#include "GEMM/GEMM_PRM.hpp"
        
    public:
        
        std::string ClassName() const
        {
            return "BLAS_Capella";
        }
        
    };
        
} // namespace HeavyMetal

