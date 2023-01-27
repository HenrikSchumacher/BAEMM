
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c GEMM2.metal -o GEMM2.air && xcrun -sdk macosx metallib GEMM2.air -o GEMM2.metallib

// FIXME: Comment-in the following line for run-time compilation:
R"(
// FIXME: Comment-out the following lines for run-time compilation:
//constant constexpr uint2 panel_size { 4, 4 };



// https://github.com/DmitryLyakh/CUDA_Tutorial
// TalkL https://www.youtube.com/watch?v=Zqfa80APkDk

// Claimed performances in the talk:
// cublas     2.000 Tflop/s
// _nn        0.142 Tflop/s
// _sh_nn     0.264 Tflop/s
// _sh_reg_nn 0.508 Tflop/s

// A further great place to look would be CBLast
// https://github.com/CNugteren/CLBlast/tree/master/src
//
// It's OpenCl, but it should be not too difficult to port it to Metal.


// TODO: Read https://siboehm.com/articles/22/CUDA-MMM on optimizing the the gemm kernel.

#include <metal_stdlib>
using namespace metal;

[[kernel]] void GEMM_PRM(
    constant uint     & M,
    constant uint     & N,
    constant uint     & K,        //in: matrix dimensions: C(M,N)+=A(M,k)*B(k,N)
    constant float    & alpha,
    device   float4x4 * A,  // pointer to A matrix data
    device   float4x4 * B,  // pointer to B matrix data
    constant float    & beta,
    device   float4x4 * C,  // pointer to C matrix data

    const uint2 g_count [[threadgroups_per_grid]],
    const uint2 g_size  [[threads_per_threadgroup]],
    const uint2 g_id    [[threadgroup_position_in_grid]],
    const uint2 l_id    [[thread_position_in_threadgroup]]
)
{
    const uint i = g_id.x;
    const uint j = g_id.y;
    
    // Each thread handles a 4 x 4 panel of C.
    thread float4x4 C_ij = {{0.f}};
    
    thread float4x4 A_ik;
//
    thread float4x4 B_kj;

    
    uint M_chunk_count = (M + 4 - 1) >> 2;
    uint N_chunk_count = (N + 4 - 1) >> 2;
    uint K_chunk_count = (K + 4 - 1) >> 2;
    
    for( uint k = 0; k < K_chunk_count; ++k )
    {
        A_ik = A[ K_chunk_count * i + k ];
        B_kj = B[ N_chunk_count * k + j ];
        C_ij += A_ik * B_kj;
    }
    
    C[ N_chunk_count * i + j ] = C_ij;
    
    return;
}
// FIXME: Comment-in the following line for run-time compilation:
)"

