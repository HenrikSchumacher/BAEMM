//
//  BLAS.metal
//  Metal_TP
//
//  Created by Henrik on 05.09.22.
//
// We have run the following command in the terminal:

//  xcrun -sdk macosx metal -c BLAS.metal -o BLAS.air && xcrun -sdk macosx metallib BLAS.air -o BLAS.metallib



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

using Real = float;

using UInt = uint;
using Int  = int;

//#define naive_tile_size 32
//
//[[max_total_threads_per_threadgroup(naive_tile_size * naive_tile_size)]]
[[kernel]] void gpu_gemm_nn(
    device const Real * __restrict__ const A,  // pointer to A matrix data
    device const Real * __restrict__ const B,  // pointer to B matrix data
    device       Real * __restrict__ const C,  // pointer to C matrix data
    device const UInt & M,
    device const UInt & N,
    device const UInt & K,          //in: matrix dimensions: C(M,N)+=A(M,k)*B(k,N)
                            
    const uint2 blk_count [[threadgroups_per_grid]],
    const uint2 blk_size  [[threads_per_threadgroup]],
    const uint2 blk       [[threadgroup_position_in_grid]],
    const uint2 idx       [[thread_position_in_threadgroup]]
)
{
    const UInt n_begin = blk.y * blk_size.y + idx.y; //global thread index Y
    const UInt n_step  = blk_count.y * blk_size.y;
    
    const UInt m_begin = blk.x * blk_size.x + idx.x; //global thread index X
    const UInt m_step  = blk_count.x * blk_size.x;
    
    for( UInt n = n_begin; n < N; n+= n_step )
    {
        for( UInt m = m_begin; m < M; m+= m_step )
        {
            Real tmp = static_cast<Real>(0);

            for(UInt k = 0; k < K; ++k )
            {
                tmp += A[k*M + m] * B[n*K + k];
            }
            C[n*M + m] = tmp;
        }
    }
    
    return;
}

[[kernel]] void gpu_gemm_sh_nn(
    device const Real * __restrict__ const A,  // pointer to A matrix data
    device const Real * __restrict__ const B,  // pointer to B matrix data
    device       Real * __restrict__ const C,  // pointer to C matrix data
    device const UInt & M,
    device const UInt & N,
    device const UInt & K,          //in: matrix dimensions: C(M,N)+=A(M,k)*B(k,N)
                                  
    const uint2 blk_count [[threadgroups_per_grid]],
    const uint2 blk_size  [[threads_per_threadgroup]],
    const uint2 blk       [[threadgroup_position_in_grid]],
    const uint2 idx       [[thread_position_in_threadgroup]]
)
{
    constexpr UInt tile_M = 16;
    constexpr UInt tile_N = 16;
    constexpr UInt tile_K = 64;
    
    threadgroup Real A_local[tile_K][tile_M];
    threadgroup Real B_local[tile_N][tile_K];
    
    for( UInt n_begin = blk.y * blk_size.y; n_begin < N; n_begin += blk_count.y * blk_size.y )
    { //tile offset in Y dimension
        
        for( UInt m_begin = blk.x * blk_size.x; m_begin < M; m_begin += blk_count.x * blk_size.x )
        { //tile offset in X dimension
            
            Real tmp = static_cast<Real>(0); //accumulator
            
            const UInt m = m_begin + idx.x;
            const bool m_valid = m < M;
            const UInt n = n_begin + idx.y;
            const bool n_valid = n < N;
            
            for( UInt k_begin = 0; k_begin < K; k_begin += tile_K )
            {
                //k_begin is the position of the CUDA thread along the K dimension
                
                const UInt k_upper    = k_begin + tile_K;
                const bool k_complete = k_upper <= K;
                const UInt k_end = k_complete ? k_upper : K;
                
                //Load a tile of matrix A(m_begin:tile_M, k_begin:tile_K):
                if( m_valid )
                {
                    for( UInt k = k_begin + idx.y; k < k_end; k += blk_size.y )
                    {
                        A_local[k-k_begin][idx.x] = A[k*M + m];
                    }
                }
                
                //Load a tile of matrix B(k_begin:tile_K, n_begin:tile_N):
                if( n_valid )
                {
                    for( UInt k = k_begin + idx.x; k < k_end; k += blk_size.x )
                    {
                        B_local[idx.y][k-k_begin] = B[n*K + k];
                    }
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                //Multiply two loaded tiles to produce a tile of matrix C(m_begin:tile_M,n_begin:tile_N):
                if( m_valid && n_valid )
                {
                    if( k_complete )
                    {
                        //number of loop iterations is known at compile time: Unroll it
                        #pragma unroll
                        for( UInt l = 0; l < tile_K; ++l )
                        {
                            tmp += A_local[l][idx.x] * B_local[idx.y][l];
                        }
                    }
                    else
                    {
                        //number of loop iterations is not known at compile time
                        for( UInt l = 0; l < (k_end - k_begin); ++l)
                        {
                            tmp += A_local[l][idx.x] * B_local[idx.y][l];
                        }
                    }
                }
                
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
            } //k_begin
            
            //Store element of the C matrix in global memory:
            if( m < M && n < N )
            {
                C[n*M + m] = tmp;
            }
            
        } //m_begin
        
    } //n_begin

    return;
}





[[kernel]] void gpu_gemm_sh_reg_nn(
    device const Real * __restrict__ const A,  // pointer to A matrix data
    device const Real * __restrict__ const B,  // pointer to B matrix data
    device       Real * __restrict__ const C,  // pointer to C matrix data
    device const UInt & M,
    device const UInt & N,
    device const UInt & K,          //in: matrix dimensions: C(M,N)+=A(M,k)*B(k,N)

    const uint2 blk_count [[threadgroups_per_grid]],
    const uint2 blk_size  [[threads_per_threadgroup]],
    const uint2 blk       [[threadgroup_position_in_grid]],
    const uint2 idx       [[thread_position_in_threadgroup]]
)
{
    // Important assumptions:
    // 1.)  tile_M == 4 * blk_size.x
    // 2.)  tile_N == 4 * blk_size.y
    // 3.)  sizeof(Real) * ( tile_M * tile_K + tile_N * tile_K ) <= 32768
    // 3.a) tile_M * tile_K + tile_N * tile_K <= 32768/sizeof(Real)
    // 3.b) tile_K <= 8192/(tile_M + tile_N);

    constexpr UInt blk_size_x = 16;
    constexpr UInt blk_size_y = 16;

    constexpr UInt tile_M     = 4 * blk_size_x;
    constexpr UInt tile_N     = 4 * blk_size_y;
    constexpr UInt tile_K     = ((32768/sizeof(Real))/(tile_M + tile_N))/2;

    threadgroup Real A_buf[tile_K][tile_M];
    threadgroup Real B_buf[tile_N][tile_K];

    for( UInt n_begin = blk.y * tile_N; n_begin < N; n_begin += blk_count.y *  tile_N )
    {
        //tile offset in Y dimension

        const UInt n_upper    = n_begin + tile_N;
        const bool n_complete = n_upper<=N;
        const UInt n_end      = n_complete ? n_upper : N;

        for( UInt m_begin = blk.x * tile_M; m_begin < M; m_begin += blk_count.x * tile_M )
        {
            //tile offset in X dimension

            const UInt m_upper    = m_begin + tile_M;
            const bool m_complete = m_upper<=M;
            const UInt m_end      = m_complete ? m_upper : M;

            if( m_complete && n_complete )
            {
                //complete tile C(tile_M,tile_N)

                //Initialize registers to zero:
                Real A_reg[4]           =  {static_cast<Real>(0)};
                Real B_reg[4]           =  {static_cast<Real>(0)};
                Real C_reg[4][4] = {{static_cast<Real>(0)}};


                for( UInt k_begin = 0; k_begin < K; k_begin += tile_K )
                {
                    //k_begin is the position of the CUDA thread along the K dimension
                    const UInt k_upper    = k_begin + tile_K;
                    const bool k_complete = k_upper<=K;
                    const UInt k_end      = k_complete ? k_upper : K;

                    // TODO: Maybe swap the two for loops?

                    //Load a tile of matrix A(m_begin:tile_M, k_begin:tile_K):
                    for( UInt k = k_begin + idx.y; k < k_end; k += blk_size_y )
                    {
                        for( UInt m = m_begin + idx.x; m < m_end; m += blk_size_x )
                        {
                            A_buf[k - k_begin][m - m_begin] = A[k*M + m];
//                                                                    ^
//                                                                    |
//  In his talk, Liakh points out that it is important for coalescence that m "goes with" idx.x
                        }
                    }

                    //Load a tile of matrix B(k_begin:tile_K, n_begin:tile_N):
                    for( UInt n = n_begin + idx.y; n < n_end; n += blk_size_y )
                    {
                        for( UInt k = k_begin + idx.x; k < k_end; k += blk_size_x )
                        {
                            B_buf[n - n_begin][k - k_begin] = B[n*K + k];
//                                                                    ^
//                                                                    |
//  In his talk, Liakh points out that it is important for coalescence that k "goes with" idx.x
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
                        for( UInt l = 0; l < tile_K; ++l )
                        {
                            #pragma unroll
                            for( UInt i = 0; i < 4; ++i )
                            {
                                A_reg[i] = A_buf[l][idx.x + blk_size_x*i];
                            }

                            #pragma unroll
                            for( UInt j = 0; j < 4; ++j )
                            {
                                B_reg[j] = B_buf[idx.y + blk_size_y*j][l];
                            }

                            #pragma unroll
                            for( UInt j = 0; j < 4; ++j )
                            {
                                #pragma unroll
                                for( UInt i = 0; i < 4; ++i )
                                {
                                    C_reg[j][i] += A_reg[i] * B_reg[j];
                                }
                            }
                        }
                    }
                    else
                    {
                        for( UInt l = 0; l < (k_end - k_begin); ++l )
                        {
                            #pragma unroll
                            for( UInt i = 0; i < 4; ++i )
                            {
                                A_reg[i] = A_buf[l][idx.x + blk_size_x * i];
                            }
                            #pragma unroll
                            for( UInt j = 0; j < 4; ++j )
                            {
                                B_reg[j] = B_buf[idx.y + blk_size_y * j][l];
                            }
                            #pragma unroll
                            for( UInt j = 0; j < 4; ++j )
                            {
                                #pragma unroll
                                for( UInt i = 0; i < 4; ++i )
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
                for( UInt j = 0; j < 4; ++j )
                {
                    const uint n = (n_begin + idx.y + blk_size_y * j );

                    #pragma unroll
                    for( UInt i = 0; i < 4; ++i )
                    {
                        const uint m = (m_begin + idx.x + blk_size_x * i );

                        C[n*M + m] = C_reg[j][i];
                    }
                }

            }
            else
            {
                //incomplete tile of C

                //Initialize registers to zero:
                Real A_reg[4]           =  {static_cast<Real>(0)};
                Real B_reg[4]           =  {static_cast<Real>(0)};
                Real C_reg[4][4] = {{static_cast<Real>(0)}};

                for( UInt k_begin = 0; k_begin < K; k_begin += tile_K )
                {
                    //k_begin is the position of the CUDA thread along the K dimension
                    const UInt k_upper    = k_begin + tile_K;
                    const bool k_complete = k_upper<=K;
                    const UInt k_end      = k_complete ? k_upper : K;

                    //Load a tile of matrix A(m_begin:tile_M, k_begin:tile_K):
                    for( UInt k = k_begin + idx.y; k < k_end; k += blk_size_y )
                    {
                        for( UInt m = m_begin + idx.x; m < m_end; m += blk_size_x )
                        {
                            A_buf[k - k_begin][m - m_begin] = A[k*M + m];
                        }
                    }

                    //Load a tile of matrix B(k_begin:tile_K, n_begin:tile_N):
                    for( UInt n = n_begin + idx.y; n < n_end; n += blk_size_y )
                    {
                        for( UInt k = k_begin + idx.x; k < k_end; k += blk_size_x )
                        {
                            B_buf[n - n_begin][k - k_begin] = B[n*K + k];
                        }
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                    //Multiply two loaded tiles to produce a tile of matrix C(m_begin:tile_M,n_begin:tile_N):
                    for( UInt l = 0; l < (k_end - k_begin); ++l )
                    {
                        for( UInt i = 0, j = idx.y; j < n_end - n_begin; j += blk_size_y, i++ )
                        {
                            B_reg[i] = B_buf[j][l];
                        }
                        for( UInt i = 0, j = idx.x; j < m_end - m_begin; j += blk_size_x, i++ )
                        {
                            A_reg[i] = A_buf[l][j];
                        }
                        #pragma unroll
                        for( UInt j = 0; j < 4; ++j )
                        {
                            #pragma unroll
                            for( UInt i = 0; i < 4; ++i )
                            {
                                C_reg[j][i] += A_reg[i] * B_reg[j];
                            }
                        }
                    }

                    threadgroup_barrier(mem_flags::mem_threadgroup);

                } //k_begin

                //Store element of the C matrix in global memory:
                for(
                    UInt n = n_begin + idx.y, j = 0;
                    n < n_end;
                    n += blk_size_y, ++j
                )
                {
                    for(
                        UInt m = m_begin + idx.x, i = 0;
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

    return;
}



//[[kernel]] void gpu_gemm_sh_reg_nn(
//    device const Real * __restrict__ const A,  // pointer to A matrix data
//    device const Real * __restrict__ const B,  // pointer to B matrix data
//    device       Real * __restrict__ const C,  // pointer to C matrix data
//    device const UInt & M,
//    device const UInt & N,
//    device const UInt & K,          //in: matrix dimensions: C(M,N)+=A(M,k)*B(k,N)
//
//    const uint2 blk_count [[threadgroups_per_grid]],
//    const uint2 blk_size  [[threads_per_threadgroup]],
//    const uint2 blk       [[threadgroup_position_in_grid]],
//    const uint2 idx       [[thread_position_in_threadgroup]]
//)
//{
//    // Important assumptions:
//    // 1.)  tile_M == 4 * blk_size.x
//    // 2.)  tile_N == 4 * blk_size.y
//    // 3.)  4 Byte * ( tile_M * tile_K + tile_N * tile_K ) <= 32768 Byte
//    // 3.a) tile_M * tile_K + tile_N * tile_K <= 8192
//    // 3.b) tile_K <= 8192/(tile_M + tile_N);
//
////    constexpr UInt tile_M = 64;
////    constexpr UInt tile_N = 64;
////    constexpr UInt tile_K = 64;
//
//    constexpr UInt blk_size_x = 16;
//    constexpr UInt blk_size_y = 16;
//
//    constexpr UInt tile_M     = 4 * blk_size_x;
//    constexpr UInt tile_N     = 4 * blk_size_y;
//    constexpr UInt tile_K     = ((32768/sizeof(Real))/(tile_M + tile_N))/2;
//
//    constexpr UInt blk_size_z = tile_K/4;
//
//    threadgroup Real A_buf[tile_K][tile_M];
//    threadgroup Real B_buf[tile_N][tile_K];
//
//    for( UInt n_begin = blk.y * tile_N; n_begin < N; n_begin += blk_count.y * tile_N )
//    {
//        //tile offset in Y dimension
//
//        const UInt n_upper    = n_begin + tile_N;
//        const bool n_complete = n_upper<=N;
//        const UInt n_end      = n_complete ? n_upper : N;
//
//        for( UInt m_begin = blk.x * tile_M; m_begin < M; m_begin += blk_count.x * tile_M )
//        {
//            //tile offset in X dimension
//
//            const UInt m_upper    = m_begin + tile_M;
//            const bool m_complete = m_upper<=M;
//            const UInt m_end      = m_complete ? m_upper : M;
//
//            if( m_complete && n_complete )
//            {
//                //complete tile C(tile_M,tile_N)
//
//                //Initialize registers to zero:
//                thread float4x4 A_reg =  {{static_cast<Real>(0)}};
//                thread float4x4 B_reg =  {{static_cast<Real>(0)}};
//                thread float4x4 C_reg =  {{static_cast<Real>(0)}};
//
//                for( UInt k_begin = 0; k_begin < K; k_begin += tile_K )
//                {
//                    //k_begin is the position of the CUDA thread along the K dimension
//                    const UInt k_upper    = k_begin + tile_K;
//                    const bool k_complete = k_upper<=K;
//                    const UInt k_end      = k_complete ? k_upper : K;
//
//                    // TODO: Maybe swap the two for loops?
//
//                    //Load a tile of matrix A(m_begin:tile_M, k_begin:tile_K):
//                    for( UInt k = k_begin + idx.y; k < k_end; k += blk_size_y )
//                    {
//                        for( UInt m = m_begin + idx.x; m < m_end; m += blk_size_x )
//                        {
//                            A_buf[k - k_begin][m - m_begin] = A[k*M + m];
////                                                                    ^
////                                                                    |
////  In his talk, Liakh points out that it is important for coalescence that m "goes with" idx.x
//                        }
//                    }
//
//                    //Load a tile of matrix B(k_begin:tile_K, n_begin:tile_N):
//                    for( UInt n = n_begin + idx.y; n < n_end; n += blk_size_y )
//                    {
//                        for( UInt k = k_begin + idx.x; k < k_end; k += blk_size_x )
//                        {
//                            B_buf[n - n_begin][k - k_begin] = B[n*K + k];
////                                                                    ^
////                                                                    |
////  In his talk, Liakh points out that it is important for coalescence that k "goes with" idx.x
////  Weird enough, because I thought of a warp (simdgroup) being rather a row of a block.
////  But then again, CUDA seems to be column-major.
////  TODO: Are Metal threadgroups column-major or row-major?
//                        }
//                    }
//
//                    threadgroup_barrier(mem_flags::mem_threadgroup);
//
//                    //Multiply two loaded tiles to produce a tile of matrix C(m_begin:tile_M,n_begin:tile_N):
//                    if( k_complete )
//                    {
//                        #pragma unroll
//                        for( UInt idx_z = 0; idx_z < blk_size_z; ++idx_z )
//                        {
//                            #pragma unroll
//                            for( UInt l = 0; l < 4; ++l )
//                            {
//                                #pragma unroll
//                                for( UInt i = 0; i < 4; ++i )
//                                {
//                                    A_reg[l][i] = A_buf[idx_z + blk_size_z * l][idx.x + blk_size_x*i];
//                                }
//                            }
//
//                            #pragma unroll
//                            for( UInt j = 0; j < 4; ++j )
//                            {
//                                #pragma unroll
//                                for( UInt l = 0; l < 4; ++l )
//                                {
//                                    B_reg[j][l] = B_buf[idx.y + blk_size_y*j][idx_z + blk_size_z * l];
//                                }
//                            }
//
//                            C_reg = A_reg * B_reg;
//                        }
//
//                    }
//                    else
//                    {
//                        // TODO: Fill in the k-noncomplete code.
//                    }
//
//                    threadgroup_barrier(mem_flags::mem_threadgroup);
//
//                } //k_begin
//
//                //Store elements of the C matrix in global memory:
//                #pragma unroll
//                for( UInt j = 0; j < 4; ++j )
//                {
//                    const uint n = (n_begin + idx.y + blk_size_y * j );
//
//                    #pragma unroll
//                    for( UInt i = 0; i < 4; ++i )
//                    {
//                        const uint m = (m_begin + idx.x + blk_size_x * i );
//
//                        C[n*M + m] = C_reg[j][i];
//                    }
//                }
//
//            }
//            else
//            {
//                //incomplete tile of C
//
//                // TODO: Fill in the noncomplete tile code.
//
//                //Initialize registers to zero:
//                float4x4 A_reg = {{static_cast<Real>(0)}};
//                float4x4 B_reg = {{static_cast<Real>(0)}};
//                float4x4 C_reg = {{static_cast<Real>(0)}};
//
//                for( UInt k_begin = 0; k_begin < K; k_begin += tile_K )
//                {
//                    //k_begin is the position of the CUDA thread along the K dimension
//                    const UInt k_upper    = k_begin + tile_K;
//                    const bool k_complete = k_upper<=K;
//                    const UInt k_end      = k_complete ? k_upper : K;
//
//                    //Load a tile of matrix A(m_begin:tile_M, k_begin:tile_K):
//                    for( UInt k = k_begin + idx.y; k < k_end; k += blk_size_y )
//                    {
//                        for( UInt m = m_begin + idx.x; m < m_end; m += blk_size_x )
//                        {
//                            A_buf[k - k_begin][m - m_begin] = A[k*M + m];
//                        }
//                    }
//
//                    //Load a tile of matrix B(k_begin:tile_K, n_begin:tile_N):
//                    for( UInt n = n_begin + idx.y; n < n_end; n += blk_size_y )
//                    {
//                        for( UInt k = k_begin + idx.x; k < k_end; k += blk_size_x )
//                        {
//                            B_buf[n - n_begin][k - k_begin] = B[n*K + k];
//                        }
//                    }
//
//                    threadgroup_barrier(mem_flags::mem_threadgroup);
//
//                    //Multiply two loaded tiles to produce a tile of matrix C(m_begin:tile_M,n_begin:tile_N):
//                    for( UInt l = 0; l < (k_end - k_begin); ++l )
//                    {
//                        for( UInt i = 0, j = idx.x; j < m_end - m_begin; j += blk_size_x, i++ )
//                        {
//                            A_reg[l][i] = A_buf[l][j];
//                        }
//                    }
//                    for( UInt i = 0, j = idx.y; j < n_end - n_begin; j += blk_size_y, i++ )
//                    {
//                        for( UInt l = 0; l < (k_end - k_begin); ++l )
//                        {
//                            B_reg[i][l] = B_buf[j][l];
//                        }
//                    }
//
//                    C_reg += A_reg * B_reg;
//
//                    threadgroup_barrier(mem_flags::mem_threadgroup);
//
//                } //k_begin
//
//                //Store element of the C matrix in global memory:
//                for(
//                    UInt n = n_begin + idx.y, j = 0;
//                    n < n_end;
//                    n += blk_size_y, ++j
//                )
//                {
//                    for(
//                        UInt m = m_begin + idx.x, i = 0;
//                        m < m_end;
//                        m += blk_size_x, ++i
//                    )
//                    {
//                        C[n*M + m] = C_reg[j][i];
//                    }
//                }
//
//            }
//
//        } //m_begin
//
//    } //n_begin
//
//    return;
//}
