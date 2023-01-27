
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

#include <metal_stdlib>
using namespace metal;

//#define naive_tile_size 32
//
//[[max_total_threads_per_threadgroup(naive_tile_size * naive_tile_size)]]
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
                            
//    const uint2 blk_count [[threadgroups_per_grid]],
//    const uint2 blk_size  [[threads_per_threadgroup]],
//    const uint2 blk       [[threadgroup_position_in_grid]],
//    const uint2 idx       [[thread_position_in_threadgroup]]
)
{
    
    //    uint2 thread_position_in_grid =
    //            (threadgroup_position_in_grid * threads_per_threadgroup) +
    //            thread_position_in_threadgroup;
    
    // Beware that Metal uses "image coordinates" starting in the top left corner,
    // and not "matrix coordinates"!

//    const uint i            = thread_position_in_grid.x;        // global row index
    const uint g_i          = threadgroup_position_in_grid.x;   // thread(g)roup row index in grid
    const uint l_i          = thread_position_in_threadgroup.x; // (l)ocal row index in threadgroup
//    const uint g_rows       = threads_per_threadgroup.x;        // number of rows in threadgroup
    const uint g_ver_count  = threadgroups_per_grid.x;          // number of threadgroups in i-direction
    
//    const uint j            = thread_position_in_grid.y;        // global column index
    const uint g_j          = threadgroup_position_in_grid.y;   // thread(g)roup column index in grid
    const uint l_j          = thread_position_in_threadgroup.y; // (l)ocal column index in threadgroup
//    const uint g_cols       = threads_per_threadgroup.y;        // number of columns in threadgroup
    const uint g_hor_count  = threadgroups_per_grid.y;          // number of threadgroups in j-direction
    
    assert( g_rows == threads_per_threadgroup.y);
    assert( g_cols == threads_per_threadgroup.x);
    
    // Important assumptions:
    // 1.)  tile_M == 4 * blk_size.x
    // 2.)  tile_N == 4 * blk_size.y
    // 3.)  sizeof(float) * ( tile_M * tile_K + tile_N * tile_K ) <= 32768
    // 3.a) tile_M * tile_K + tile_N * tile_K <= 32768/sizeof(float)
    // 3.b) tile_K <= 8192/(tile_M + tile_N);

    constexpr uint blk_size_x = 16;
    constexpr uint blk_size_y = 16;

    constexpr uint tile_M     = 4 * blk_size_x;
    constexpr uint tile_N     = 4 * blk_size_y;
    constexpr uint tile_K     = ((32768/sizeof(float))/(tile_M + tile_N))/2;

    threadgroup float A_buf[tile_K][tile_M];
    threadgroup float B_buf[tile_N][tile_K];

    for( uint n_begin = g_j * tile_N; n_begin < N; n_begin += g_hor_count *  tile_N )
    {
        //tile offset in Y dimension

        const uint n_upper    = n_begin + tile_N;
        const bool n_complete = n_upper<=N;
        const uint n_end      = n_complete ? n_upper : N;

        for( uint m_begin = g_i * tile_M; m_begin < M; m_begin += g_ver_count * tile_M )
        {
            //tile offset in X dimension

            const uint m_upper    = m_begin + tile_M;
            const bool m_complete = m_upper<=M;
            const uint m_end      = m_complete ? m_upper : M;

            if( m_complete && n_complete )
            {
                //complete tile C(tile_M,tile_N)

                //Initialize registers to zero:
                float A_reg[4]           =  {static_cast<float>(0)};
                float B_reg[4]           =  {static_cast<float>(0)};
                float C_reg[4][4] = {{static_cast<float>(0)}};


                for( uint k_begin = 0; k_begin < K; k_begin += tile_K )
                {
                    //k_begin is the position of the CUDA thread along the K dimension
                    const uint k_upper    = k_begin + tile_K;
                    const bool k_complete = k_upper<=K;
                    const uint k_end      = k_complete ? k_upper : K;

                    // TODO: Maybe swap the two for loops?

                    //Load a tile of matrix A(m_begin:tile_M, k_begin:tile_K):
                    for( uint k = k_begin + l_j; k < k_end; k += blk_size_y )
                    {
                        for( uint m = m_begin + l_i; m < m_end; m += blk_size_x )
                        {
                            A_buf[k - k_begin][m - m_begin] = A[k*M + m];
//                                                                    ^
//                                                                    |
//  In his talk, Liakh points out that it is important for coalescence that m "goes with" l_i
                        }
                    }

                    //Load a tile of matrix B(k_begin:tile_K, n_begin:tile_N):
                    for( uint n = n_begin + l_j; n < n_end; n += blk_size_y )
                    {
                        for( uint k = k_begin + l_i; k < k_end; k += blk_size_x )
                        {
                            B_buf[n - n_begin][k - k_begin] = B[n*K + k];
//                                                                    ^
//                                                                    |
//  In his talk, Liakh points out that it is important for coalescence that k "goes with" l_i
//  Weird enough, because I thought of a warp (simdgroup) being rather a row of a block.
//  But then again, CUDA seems to be column-major.
//  TODO: Are Metal threadgroups column-major or row-major?
                        }
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    //Multiply two loaded tiles to produce a tile of matrix C(m_begin:tile_M,n_begin:tile_N):
                    if( k_complete )
                    {
                        #pragma unroll
                        for( uint l = 0; l < tile_K; ++l )
                        {
                            #pragma unroll
                            for( uint i = 0; i < 4; ++i )
                            {
                                A_reg[i] = A_buf[l][l_i + blk_size_x*i];
                            }

                            #pragma unroll
                            for( uint j = 0; j < 4; ++j )
                            {
                                B_reg[j] = B_buf[l_j + blk_size_y*j][l];
                            }

                            #pragma unroll
                            for( uint j = 0; j < 4; ++j )
                            {
                                #pragma unroll
                                for( uint i = 0; i < 4; ++i )
                                {
                                    C_reg[j][i] += A_reg[i] * B_reg[j];
                                }
                            }
                        }
                    }
                    else
                    {
                        for( uint l = 0; l < (k_end - k_begin); ++l )
                        {
                            #pragma unroll
                            for( uint i = 0; i < 4; ++i )
                            {
                                A_reg[i] = A_buf[l][l_i + blk_size_x * i];
                            }
                            #pragma unroll
                            for( uint j = 0; j < 4; ++j )
                            {
                                B_reg[j] = B_buf[l_j + blk_size_y * j][l];
                            }
                            #pragma unroll
                            for( uint j = 0; j < 4; ++j )
                            {
                                #pragma unroll
                                for( uint i = 0; i < 4; ++i )
                                {
                                    C_reg[j][i] += A_reg[i] * B_reg[j];
                                }
                            }
                        }
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                } //k_begin

                //Store elements of the C matrix in global memory:
                #pragma unroll
                for( uint j = 0; j < 4; ++j )
                {
                    const uint n = (n_begin + l_j + blk_size_y * j );

                    #pragma unroll
                    for( uint i = 0; i < 4; ++i )
                    {
                        const uint m = (m_begin + l_i + blk_size_x * i );

                        C[n*M + m] = C_reg[j][i];
                    }
                }

            }
            else
            {
                //incomplete tile of C

                //Initialize registers to zero:
                float A_reg[4]           =  {static_cast<float>(0)};
                float B_reg[4]    =  {static_cast<float>(0)};
                float C_reg[4][4] = {{static_cast<float>(0)}};

                for( uint k_begin = 0; k_begin < K; k_begin += tile_K )
                {
                    //k_begin is the position of the CUDA thread along the K dimension
                    const uint k_upper    = k_begin + tile_K;
                    const bool k_complete = k_upper<=K;
                    const uint k_end      = k_complete ? k_upper : K;

                    //Load a tile of matrix A(m_begin:tile_M, k_begin:tile_K):
                    for( uint k = k_begin + l_j; k < k_end; k += blk_size_y )
                    {
                        for( uint m = m_begin + l_i; m < m_end; m += blk_size_x )
                        {
                            A_buf[k - k_begin][m - m_begin] = A[k*M + m];
                        }
                    }

                    //Load a tile of matrix B(k_begin:tile_K, n_begin:tile_N):
                    for( uint n = n_begin + l_j; n < n_end; n += blk_size_y )
                    {
                        for( uint k = k_begin + l_i; k < k_end; k += blk_size_x )
                        {
                            B_buf[n - n_begin][k - k_begin] = B[n*K + k];
                        }
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    //Multiply two loaded tiles to produce a tile of matrix C(m_begin:tile_M,n_begin:tile_N):
                    for( uint l = 0; l < (k_end - k_begin); ++l )
                    {
                        for( uint i = 0, j = l_j; j < n_end - n_begin; j += blk_size_y, i++ )
                        {
                            B_reg[i] = B_buf[j][l];
                        }
                        for( uint i = 0, j = l_i; j < m_end - m_begin; j += blk_size_x, i++ )
                        {
                            A_reg[i] = A_buf[l][j];
                        }
                        #pragma unroll
                        for( uint j = 0; j < 4; ++j )
                        {
                            #pragma unroll
                            for( uint i = 0; i < 4; ++i )
                            {
                                C_reg[j][i] += A_reg[i] * B_reg[j];
                            }
                        }
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                } //k_begin

                //Store element of the C matrix in global memory:
                for(
                    uint n = n_begin + l_j, j = 0;
                    n < n_end;
                    n += blk_size_y, ++j
                )
                {
                    for(
                        uint m = m_begin + l_i, i = 0;
                        m < m_end;
                        m += blk_size_x, ++i
                    )
                    {
                        C[n*M + m] = C_reg[j][i];
                    }
                }

            }

        } //m_begin

    } //n_begin
}
// FIXME: Comment-in the following line for run-time compilation:
)"
