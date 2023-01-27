
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c GEMM2.metal -o GEMM2.air && xcrun -sdk macosx metallib GEMM2.air -o GEMM2.metallib

// FIXME: Comment-in the following line for run-time compilation:
R"(
// FIXME: Comment-out the following lines for run-time compilation:
//constant constexpr uint g_rows = 16;
//constant constexpr uint g_cols = 16;



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

[[kernel]] void GEMM_CM(
    device   const uint  & M,
    device   const uint  & N,
    device   const uint  & K,        //in: matrix dimensions: C(M,N)+=A(M,k)*B(k,N)
    constant const float & alpha,
    device   const float * const A,  // pointer to A matrix data
    device   const float * const B,  // pointer to B matrix data
    constant const float & beta,
    device         float * const C,  // pointer to C matrix data

    const uint2 thread_position_in_grid         [[thread_position_in_grid]],
    const uint2 threadgroup_position_in_grid    [[threadgroup_position_in_grid]],
    const uint2 thread_position_in_threadgroup  [[thread_position_in_threadgroup]],
    const uint2 threads_per_threadgroup         [[threads_per_threadgroup]],
    const uint2 threadgroups_per_grid           [[threadgroups_per_grid]]
)
{
//    uint2 thread_position_in_grid =
//            (threadgroup_position_in_grid * threads_per_threadgroup) +
//            thread_position_in_threadgroup;
    
    // {x,y} are col-major coordinates!!!
    
    const uint i            = thread_position_in_grid.x;        // global row index
    const uint g_i          = threadgroup_position_in_grid.x;   // thread(g)roup row index in grid
    const uint l_i          = thread_position_in_threadgroup.x; // (l)ocal row index in threadgroup
//    const uint g_rows       = threads_per_threadgroup.x;        // number of rows in threadgroup
    const uint g_ver_count  = threadgroups_per_grid.x;          // number of threadgroups in i-direction
    
    const uint j            = thread_position_in_grid.y;        // global column index
    const uint g_j          = threadgroup_position_in_grid.y;   // thread(g)roup column index in grid
    const uint l_j          = thread_position_in_threadgroup.y; // (l)ocal column index in threadgroup
//    const uint g_cols       = threads_per_threadgroup.y;        // number of columns in threadgroup
    const uint g_hor_count  = threadgroups_per_grid.y;          // number of threadgroups in j-direction
    
    assert( g_rows == g_size.y);
    assert( g_cols == g_size.x);
    
    
    assert( threads_per_threadgroup.x == g_rows);
    assert( threads_per_threadgroup.y == g_cols);
    
    // Important assumptions:
    // 1.)  panel_M == 4 * g_size.x
    // 2.)  panel_N == 4 * g_size.y
    // 3.)  sizeof(float) * ( panel_M * panel_K + panel_N * panel_K ) <= 32768
    // 3.a) panel_M * panel_K + panel_N * panel_K <= 32768/sizeof(float)
    // 3.b) panel_K <= 8192/(panel_M + panel_N);


    constexpr uint panel_M = 4 * g_rows;
    constexpr uint panel_N = 4 * g_cols;
    constexpr uint panel_K = ((32768/sizeof(float))/(panel_M + panel_N))/2;

    threadgroup float A_shared[panel_K][panel_M];
    threadgroup float B_shared[panel_N][panel_K];

    for( uint n_begin = g_j * panel_N; n_begin < N; n_begin += g_hor_count * panel_N )
    {
        //tile offset in Y dimension

        const uint n_upper    = n_begin + panel_N;
        const bool n_complete = n_upper <= N;
        const uint n_end      = n_complete ? n_upper : N;

        for( uint m_begin = g_i * panel_M; m_begin < M; m_begin += g_ver_count * panel_M )
        {
            //tile offset in X dimension

            const uint m_upper    = m_begin + panel_M;
            const bool m_complete = m_upper <= M;
            const uint m_end      = m_complete ? m_upper : M;


            //Initialize registers to zero:
            float A_t_k  [4]    =  {static_cast<float>(0)};
            float B_t_k  [4]    =  {static_cast<float>(0)};
            float C_tile [4][4] = {{static_cast<float>(0)}};


            for( uint k_begin = 0; k_begin < K; k_begin += panel_K )
            {
                //k_begin is the position of the CUDA thread along the K dimension
                const uint k_upper    = k_begin + panel_K;
                const bool k_complete = k_upper <= K;
                const uint k_end      = k_complete ? k_upper : K;

                // TODO: Maybe swap the two for loops?

                //Load a tile of matrix A(m_begin:panel_M, k_begin:panel_K).
                // Each thread loads 4 x 4 elements into A_shared.
                // But read is scattered for coalescence!
                
                // This loop has (k_end - k_begin)/ g_cols = panel_K / g_cols = 4 iterations.
                for( uint k = k_begin + l_j; k < k_end; k += g_cols )
                {
                    // This loop has (m_end - m_begin)/ g_rows = panel_M / g_rows = 4 iterations.
                    for( uint m = m_begin + l_i; m < m_end; m += g_rows )
                    {
                        A_shared[k - k_begin][m - m_begin] = A[k*M + m];
//                                                                ^
//                                                                |
//  In his talk, Liakh points out that it is important for coalescence that m "goes with" l_i
                    }
                }

                //Load a tile of matrix B(k_begin:panel_K, n_begin:panel_N):
                for( uint n = n_begin + l_j; n < n_end; n += g_cols )
                {
                    for( uint k = k_begin + l_i; k < k_end; k += g_rows )
                    {
                        B_shared[n - n_begin][k - k_begin] = B[n*K + k];
//                                                                    ^
//                                                                    |
//  In his talk, Liakh points out that it is important for coalescence that k "goes with" l_i
//  Weird enough, because I thought of a warp (simdgroup) being rather a row of a block.
//  But then again, CUDA seems to be column-major.
//  TODO: Are Metal threadgroups column-major or row-major?
                    }
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                //Multiply two loaded tiles to produce a tile of matrix C(m_begin:panel_M,n_begin:panel_N):
                if( k_complete )
                {
                    for( uint t_k = 0; t_k < panel_K; ++t_k )
                    {
                        for( uint t_i = 0; t_i < 4; ++t_i )
                        {
                            A_t_k[t_i] = A_shared[t_k][g_rows * t_i + l_i];
                        }

                        for( uint t_j = 0; t_j < 4; ++t_j )
                        {
                            B_t_k[t_j] = B_shared[g_cols * t_j + l_j][t_k];
                        }

                        for( uint t_j = 0; t_j < 4; ++t_j )
                        {
                            for( uint t_i = 0; t_i < 4; ++t_i )
                            {
                                C_tile[t_j][t_i] += A_t_k[t_i] * B_t_k[t_j];
                            }
                        }
                    }
                }
                else
                {
                    for( uint t_k = 0; t_k < (k_end - k_begin); ++t_k )
                    {
                        for( uint t_i = 0; t_i < 4; ++t_i )
                        {
                            A_t_k[t_i] = A_shared[t_k][g_rows * t_i + l_i];
                        }
                        
                        for( uint t_j = 0; t_j < 4; ++t_j )
                        {
                            B_t_k[t_j] = B_shared[l_j + g_cols * t_j][t_k];
                        }
                        
                        for( uint t_j = 0; t_j < 4; ++t_j )
                        {
                            for( uint t_i = 0; t_i < 4; ++t_i )
                            {
                                C_tile[t_j][t_i] += A_t_k[t_i] * B_t_k[t_j];
                            }
                        }
                    }
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

            } //k_begin
            
            //Store elements of the C matrix in global memory:

            for( uint t_i = 0; t_i < 4; ++t_i )
            {
                const uint i = (m_begin + l_i + g_rows * t_i );

                for( uint t_j = 0; t_j < 4; ++t_j )
                {
                    const uint j = (n_begin + l_j + g_cols * t_j );

                    C[M*j + i] = alpha * C_tile[t_i][t_j] + beta * C[M*j + i];
                }
            }


        } //m_begin

    } //n_begin

    return;
}
// FIXME: Comment-in the following line for run-time compilation:
)"

