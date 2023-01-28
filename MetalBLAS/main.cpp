//
//  main.cpp
//

//https://larsgeb.github.io/2022/04/22/m1-gpu.html
//https://github.com/larsgeb/m1-gpu-cpp

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <iostream>
#include <random>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <Accelerate/Accelerate.h>


#undef TENSORS_BOUND_CHECKS
#include <complex>
#include "../Repulsor/Tensors/Tools/Tools.hpp"
#include "../Repulsor/Tensors/Tensors.hpp"

using namespace Tools;
using namespace Tensors;

#include "MetalBLAS.hpp"

#include "../src/HeavyMetal.hpp"




int main(int argc, const char * argv[])
{
    using Float = float;
    using Int  = int;
    using UInt = uint;
    
    NS::AutoreleasePool* p_pool = NS::AutoreleasePool::alloc()->init();
    
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    
    MetalBLAS ops (device);
    
    HeavyMetal::BLAS_Capella capella (device);
    
    const UInt thread_count = 8;
    
    Profiler::Clear("/Users/Henrik");

    valprint("thread_count",thread_count);


    const UInt M = 16*1024;
    const UInt N = 64;
    const UInt K = 16*1024;

    MTL::Buffer * A = device->newBuffer(
        M * K * sizeof(Float), MTL::ResourceStorageModeManaged
    );
    mut<Float> A_ = reinterpret_cast<Float*>(A->contents());

    MTL::Buffer * B = device->newBuffer(
        K * N * sizeof(Float), MTL::ResourceStorageModeManaged
    );
    mut<Float> B_ = reinterpret_cast<Float*>(B->contents());

    MTL::Buffer * C = device->newBuffer(
        M * N * sizeof(Float), MTL::ResourceStorageModeManaged
    );
    mut<Float> C_ = reinterpret_cast<Float*>(C->contents());

    MTL::Buffer * C_True = device->newBuffer(
        M * N * sizeof(Float), MTL::ResourceStorageModeManaged
    );
    mut<Float> C_True_ = reinterpret_cast<Float*>(C_True->contents());
    
    Tensor2<Float,Int> Z ( M, N );
    
    std::random_device r;
    std::default_random_engine engine ( r() );

    std::uniform_real_distribution<Float> unif(static_cast<Float>(-1),static_cast<Float>(1));

    for( UInt i = 0; i < M * K; ++i )
    {
        A_[i] = unif(engine);
    }

    for( UInt i = 0; i < K * N; ++i )
    {
        B_[i] = unif(engine);
    }

    std::fill( C_, &C_[M*N], static_cast<Float>(0) );
    
    constexpr UInt repetitions = 10;
    double time;
    
    const double Tflops = 2. * M * N * K * repetitions * std::pow(10.,-12);
    const double GB     = sizeof(float) *  (  M * N + M * K + K * N ) * repetitions * std::pow(10.,-9);
    print("Theoretical GPU Performance = 10.4 Tflop/s (fp32)");
    print("Theoretical GPU Bandwidth   = 400 GB/s (fp32)");
    
    valprint("Tflops",Tflops);
    valprint("GB    ",GB);
    
    print("");
    print("Matrix multiplication C = A * B requires "+ToString(Tflops)+" Tflop.");
    print("");
    
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    capella.GEMM_CM_Ref(M, N, K, alpha, A, B, beta, C_True );
    tic("capella.GEMM_CM_Ref");
    for( UInt r = 0; r < repetitions; ++r )
    {
        capella.GEMM_CM_Ref(M, N, K, alpha, A, B, beta, C_True );
    }
    time=toc("capella.GEMM_CM_Ref");
    copy_buffer(C_True_, Z.data(), M*N);
    Float C_True_MaxNorm;
    C_True_MaxNorm = Z.MaxNorm();
    print("Throughput = "+ToString(Tflops/time)+" Tflop/s");
    print("Throughput = "+ToString(GB/time)+" GB/s");
    
    print("");
    
    const auto f = [&]( double time, ptr<Float> C_)
    {
        print("Throughput = "+ToString(Tflops/time)+" Tflop/s");
        print("Throughput = "+ToString(GB/time)+" GB/s");
        
        copy_buffer(C_, Z.data(), M*N);
        combine_buffers<ScalarFlag::Minus,ScalarFlag::Plus>(
            Float(-1), C_True_, Float(1), Z.data(), static_cast<size_t>(M*N)
        );

        valprint("error", Z.MaxNorm() / C_True_MaxNorm );
        print("");
    };
    
    capella.GEMM_CM(M, N, K, alpha, A, B, beta, C, 16, 16 );
    tic("capella.GEMM_CM");
    for( UInt r = 0; r < repetitions; ++r )
    {
        capella.GEMM_CM(M, N, K, alpha, A, B, beta, C, 16, 16 );
    }
    time = toc("capella.GEMM_CM");
    f(time, C_);
    
    
    print("");
    print("");
    print("");
    
    capella.GEMM_RM_Ref(M, N, K, alpha, A, B, beta, C_True );
    tic("capella.GEMM_RM_Ref");
    for( UInt r = 0; r < repetitions; ++r )
    {
        capella.GEMM_RM_Ref(M, N, K, alpha, A, B, beta, C_True );
    }
    time=toc("capella.GEMM_RM_Ref");
    copy_buffer(C_True_, Z.data(), M*N);
//    Float C_True_MaxNorm;
    C_True_MaxNorm = Z.MaxNorm();
    print("Throughput = "+ToString(Tflops/time)+" Tflop/s");
    print("Throughput = "+ToString(GB/time)+" GB/s");
    
    dump(C_True_MaxNorm);
    print("");
    
//    capella.GEMM_RM_NVidea(M, N, K, alpha, A, B, beta, C, 16 );
//    tic("capella.GEMM_RM_NVidea");
//    for( UInt r = 0; r < repetitions; ++r )
//    {
//        capella.GEMM_RM_NVidea(M, N, K, alpha, A, B, beta, C, 16 );
//    }
//    time = toc("capella.GEMM_RM_NVidea");
//    f(time, C_);

    
    capella.GEMM_RM(M, N, K, alpha, A, B, beta, C );
    tic("capella.GEMM_RM");
    for( UInt r = 0; r < repetitions; ++r )
    {
        capella.GEMM_RM(M, N, K, alpha, A, B, beta, C );
    }
    time = toc("capella.GEMM_RM");
    f(time, C_);
    
    print("");
    

    
//    capella.GEMM_RM(M, N, K, alpha, A, B, beta, C, 16, 16 );
//    tic("capella.GEMM_RM");
//    for( UInt r = 0; r < repetitions; ++r )
//    {
//        capella.GEMM_RM(M, N, K, alpha, A, B, beta, C, 16, 16 );
//    }
//    time = toc("capella.GEMM_RM");
//    f(time, C_);
    
    
    p_pool->release();
    
    return 0;
}





//    tic("A");
//    HeavyMetal::MatrixPanelRowMajor<4,4,float,uint> A_PM ( M, K );
//    HeavyMetal::MatrixPanelRowMajor<4,4,float,uint> B_PM ( K, N );
//    HeavyMetal::MatrixPanelRowMajor<4,4,float,uint> C_PM ( M, N );
//    toc("A");
//
//    tic("B");
//    A_PM.FromRowMajor(A_,thread_count);
//    A_PM.ToPanelRowMajor(A_,thread_count);
//
//    B_PM.FromRowMajor(B_,thread_count);
//    B_PM.ToPanelRowMajor(B_,thread_count);
//    toc("B");
//
//    tic("capella.GEMM_PRM");
//    capella.GEMM_PRM(M, N, K, 1.f, A, B, 0.f, C);
//    for( UInt r = 0; r < repetitions; ++r )
//    {
//        capella.GEMM_PRM(M, N, K, 1.f, A, B, 0.f, C);
//    }
//    time = toc("capella.GEMM_PRM");
//    f(time, C_);
//
//
//    C_PM.FromPanelRowMajor(C_,thread_count);
//    C_PM.ToPanelRowMajor(C_,thread_count);
