
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
    
    
//    const uint groupId  = M * g_j  + g_i;
//    const uint threadId = (g_rows * g_cols) * groupId + g_rows * l_j + l_i;
    
    assert( g_rows == threads_per_threadgroup.x);
    assert( g_cols == threads_per_threadgroup.y);
    
    // Important assumptions:
    // 1.)  tile_M == 4 * g_rows
    // 2.)  tile_N == 4 * g_cols
    // 3.)  sizeof(float) * ( tile_M * tile_K + tile_N * tile_K ) <= 32768
    // 3.a) tile_M * tile_K + tile_N * tile_K <= 32768/sizeof(float)
    // 3.b) tile_K <= 8192 / (tile_M + tile_N);

    constexpr uint tile_M     = 4 * g_rows;
    constexpr uint tile_N     = 4 * g_cols;
    constexpr uint tile_K     = ((32768/sizeof(float))/(tile_M + tile_N))/2;

    threadgroup float A_shared[tile_K][tile_M];
    threadgroup float B_shared[tile_N][tile_K];

    for( uint j_begin = g_j * tile_N; j_begin < N; j_begin += g_hor_count * tile_N )
    {
        //tile offset in j dimension

        const uint j_upper    = j_begin + tile_N;
        const bool j_complete = j_upper <= N;
        const uint j_end      = j_complete ? j_upper : N;

        for( uint i_begin = g_i * tile_M; i_begin < M; i_begin += g_ver_count * tile_M )
        {
            //tile offset in i dimension

            const uint i_upper    = i_begin + tile_M;
            const bool i_complete = i_upper <= M;
            const uint i_end      = i_complete ? i_upper : M;

            if( i_complete && i_complete )
            {
                //complete tile C(tile_M,tile_N)

                //Initialize registers to zero:
                float A_t_k[4]     =  {static_cast<float>(0)};
                float B_t_k[4]     =  {static_cast<float>(0)};
                float C_tile[4][4] = {{static_cast<float>(0)}};


                for( uint k_begin = 0; k_begin < K; k_begin += tile_K )
                {
                    //k_begin is the position of the CUDA thread along the K dimension
                    const uint k_upper    = k_begin + tile_K;
                    const bool k_complete = k_upper <= K;
                    const uint k_end      = k_complete ? k_upper : K;

                    //Load a tile of matrix A( i_begin:tile_M, k_begin:tile_K ):
                    for( uint k = k_begin + l_j; k < k_end; k += g_cols )
                    {
                        for( uint i = i_begin + l_i; i < i_end; i += g_rows )
                        {
                            assert( k - k_begin < tileK );
                            assert( i - i_begin < tileM );
                            A_shared[k - k_begin][i - i_begin] = A[M * k + i];
//                                                                      ^
//                                                                      |
//  In his talk, Liakh points out that it is important for coalescence that i "goes with" l_i
                        }
                    }

                    //Load a tile of matrix B( k_begin:tile_K, j_begin:tile_N ):
                    for( uint j = j_begin + l_j; j < j_end; j += g_cols )
                    {
                        for( uint k = k_begin + l_i; k < k_end; k += g_rows )
                        {
                            B_shared[j - j_begin][k - k_begin] = B[K * j + k];
//                                                                      ^
//                                                                      |
//  In his talk, Liakh points out that it is important for coalescence that k "goes with" l_i
//  Weird enough, because I thought of a warp (simdgroup) being rather a row of a block.
//  But then again, CUDA seems to be column-major.
//  TODO: Are Metal threadgroups column-major or row-major?
                        }
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    // Multiply two loaded tiles to produce a tile of matrix
                    // C( i_begin:tile_M, j_begin:tile_N ):
                    if( k_complete )
                    {
                        for( uint t_k = 0; t_k < tile_K; ++t_k )
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

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                } // k_begin

                //Store elements of the C matrix in global memory:
                for( uint t_j = 0; t_j < 4; ++t_j )
                {
                    const uint j = j_begin + g_cols * t_j + l_j;

                    for( uint t_i = 0; t_i < 4; ++t_i )
                    {
                        const uint i = i_begin + g_rows * t_i + l_i;

                        C[M * j + i] = C_tile[t_j][t_i];
                    }
                }

            }
            else
            {
                //incomplete tile of C

                //Initialize registers to zero:
                float A_t_k [4]    =  {static_cast<float>(0)};
                float B_t_k [4]    =  {static_cast<float>(0)};
                float C_tile[4][4] = {{static_cast<float>(0)}};

                for( uint k_begin = 0; k_begin < K; k_begin += tile_K )
                {
                    //k_begin is the position of the CUDA thread along the K dimension
                    const uint k_upper    = k_begin + tile_K;
                    const bool k_complete = k_upper<=K;
                    const uint k_end      = k_complete ? k_upper : K;

                    //Load a tile of matrix A( i_begin:tile_M, k_begin:tile_K ):
                    for( uint k = k_begin + l_j; k < k_end; k += g_cols )
                    {
                        for( uint i = i_begin + l_i; i < i_end; i += g_rows )
                        {
                            A_shared[k - k_begin][i - i_begin] = A[M * k + i];
                        }
                    }

                    //Load a tile of matrix B( k_begin:tile_K, j_begin:tile_N ):
                    for( uint j = j_begin + l_j; j < j_end; j += g_cols )
                    {
                        for( uint k = k_begin + l_i; k < k_end; k += g_rows )
                        {
                            B_shared[j - j_begin][k - k_begin] = B[K * j + k];
                        }
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    // Multiply two loaded tiles to produce a tile of matrix
                    // C( i_begin:tile_M, j_begin:tile_N ):
                    for( uint t_k = 0; t_k < (k_end - k_begin); ++t_k )
                    {
                        for( uint i = 0, j = l_j; j < j_end - j_begin; j += g_cols, ++i )
                        {
                            B_t_k[i] = B_shared[j][t_k];
                        }
                        for( uint i = 0, j = l_i; j < i_end - i_begin; j += g_rows, ++i )
                        {
                            A_t_k[i] = A_shared[t_k][j];
                        }

                        for( uint t_j = 0; t_j < 4; ++t_j )
                        {
                            for( uint t_i = 0; t_i < 4; ++t_i )
                            {
                                C_tile[t_j][t_i] += A_t_k[t_i] * B_t_k[t_j];
                            }
                        }
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                } //k_begin

                //Store element of the C matrix in global memory:
                for(
                    uint j = j_begin + l_j, t_j = 0;
                    j < j_end;
                    j += g_cols, ++t_j
                )
                {
                    for(
                        uint i = i_begin + l_i, t_i = 0;
                        i < i_end;
                        i += g_rows, ++t_i
                    )
                    {
                        C[M * j + i] = C_tile[t_j][t_i];
                    }
                }

            }

        } // i_begin

    } // j_begin
}
// FIXME: Comment-in the following line for run-time compilation:
)"
