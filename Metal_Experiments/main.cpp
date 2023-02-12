#include <iostream>

#include <sys/types.h>
#include <pwd.h>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <Accelerate/Accelerate.h>


#define TOOLS_ENABLE_PROFILER // enable profiler

#include "Tensors/Tensors.hpp"

using namespace Tools;
using namespace Tensors;

template<typename T>
T * ToPtr( MTL::Buffer * a )
{
    return reinterpret_cast<T*>(a->contents());
}


// We have to toggle which domain dimensions and ambient dimensions shall be supported by runtime polymorphism before we load Repulsor.hpp
// Bou can activate everything you want, but compile times might increase substatially.
using Int           = int;
using Real          = float;
using Complex       = std::complex<float>;


#include "../src/HeavyMetal.hpp"

int main(int argc, const char * argv[])
{
    // Some pool to handle reference-counted objects associated to Metal (namespace MTL).
    NS::AutoreleasePool * p_pool = NS::AutoreleasePool::alloc()->init();

    // Request the GPU device.
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
  
    HeavyMetal::BLAS_Capella capella ( device );
    
    //Loading vertex
    uint N = 4096;
    MTL::Buffer * A = device->newBuffer(N * N * sizeof(Complex), MTL::StorageMode::StorageModeShared );
    
    MTL::Buffer * B = device->newBuffer(N * N * sizeof(Complex), MTL::StorageMode::StorageModeShared );
    
    MTL::Buffer * C = device->newBuffer(N * N * sizeof(Complex), MTL::StorageMode::StorageModeShared );
    
    MTL::Buffer * C_True = device->newBuffer(N * N * sizeof(Complex), MTL::StorageMode::StorageModeShared );
    
    
    Tensor2<Complex,Int> D (N,N);
    D.Random();
    D.Write( ToPtr<Complex>(A) );
    
    D.Random();
    D.Write( ToPtr<Complex>(B) );
    
    const Complex alpha = 1;
    const Complex beta  = 0;
    
    tic("GEMM_CM_C");
    capella.GEMM_CM_C_Ref( N, N, N, alpha, A, B, beta, C_True );
    toc("GEMM_CM_C");
    
    tic("GEMM_CM_C");
    capella.GEMM_CM_C( N, N, N, alpha, A, B, beta, C, 16, 16 );
    toc("GEMM_CM_C");
    
    dump( ToPtr<Complex>(C_True)[0] );
    dump( ToPtr<Complex>(C)[0] );
    
    // Destruct the pool for managing Metal's reference counted pointers.
    p_pool->release();
    
    return 0;
}
