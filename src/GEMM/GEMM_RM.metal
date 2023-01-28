
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c GEMM2.metal -o GEMM2.air && xcrun -sdk macosx metallib GEMM2.air -o GEMM2.metallib

// FIXME: Comment-in the following line for run-time compilation:
R"(

// FIXME: Comment-out the following lines for run-time compilation:
//constant constexpr Int SIMD_width = 32;
//constant constexpr Int SIMD_log = 5;



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

#include <metal_stdlib>
using namespace metal;

using Int = int;

[[kernel]] void GEMM_RM(
    device   const Int   & M,
    device   const Int   & N,
    device   const Int   & K,        //in: matrix dimensions: C(M,N)+=A(M,k)*B(k,N)
    constant const float & alpha,
    device   const float * const A,  // pointer to A matrix data
    device   const float * const B,  // pointer to B matrix data
    constant const float & beta,
    device         float * const C,  // pointer to C matrix data
                        
//    const uint2 thread_position_in_grid             [[thread_position_in_grid]],
    const uint2 threadgroup_position_in_grid        [[threadgroup_position_in_grid]],
//    const uint2 thread_position_in_threadgroup      [[thread_position_in_threadgroup]],
//    const uint2 threads_per_threadgroup             [[threads_per_threadgroup]],
//    const uint2 threadgroups_per_grid               [[threadgroups_per_grid]],
                        
    const uint simdgroup_position_in_threadgroup   [[simdgroup_index_in_threadgroup]],
    const uint thread_position_in_simdgroup        [[thread_index_in_simdgroup]]
)
{
    const Int l_i       = simdgroup_position_in_threadgroup;
    const Int g_i       = threadgroup_position_in_grid.y;
//    const Int g_i_steps = (M + SIMD_width - 1) >> SIMD_log;
    const Int i         = SIMD_width * g_i + l_i;

    
    const Int l_j       = thread_position_in_simdgroup;
    const Int g_j       = threadgroup_position_in_grid.x;
//    const Int g_j_steps  = (N + SIMD_width - 1) >> SIMD_log;
    const Int j         = SIMD_width * g_j + l_j;

//    const Int g_k_steps  = (K + SIMD_width - 1) >> SIMD_log;
    
    const Int g_k_steps  = K >> SIMD_log;
    
    // threadgroup computes
    
    threadgroup float A_shared[SIMD_width][SIMD_width];
    threadgroup float B_shared[SIMD_width][SIMD_width];
    
    thread float C_ij = 0.f;
    
    for( Int g_k = 0; g_k < g_k_steps; ++g_k )
    {
        // Coalesced load from device memory:
        // Each simdgroup of 32 threads loads a row of A.
        A_shared[l_i][l_j] = A[K * i + (SIMD_width * g_k + l_j)];

        // Coalesced load from device memory:
        // Each simdgroup of 32 threads loads a row of B.
        B_shared[l_i][l_j] = B[N * (SIMD_width * g_k + l_i) + j];

//        simdgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // TODO: As all simdgroups in the threadgroup require the same row of B_shared, this might result in massive bank conflict.
        // TODO: Maybe we can use a simd_shuffle_rotate somehow?
        for( Int l_k = 0; l_k < SIMD_width; ++l_k )
        {
            C_ij += A_shared[l_i][l_k] * B_shared[l_k][l_j];
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Each simdgroup writes a 32-row-chunk of C.
    C[N * i + j] = alpha * C_ij + beta * C[N * i + j];
}
// FIXME: Comment-in the following line for run-time compilation:
)"
