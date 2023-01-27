
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

[[kernel]] void GEMM_RM(
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

    const uint i            = thread_position_in_grid.y;        // global row index
    const uint g_i          = threadgroup_position_in_grid.y;   // thread(g)roup row index in grid
    const uint l_i          = thread_position_in_threadgroup.y; // (l)ocal row index in threadgroup
    const uint g_rows       = threads_per_threadgroup.y;        // number of rows in threadgroup
    const uint g_vert_count = threadgroups_per_grid.y;          // number of threadgroups in i-direction
    
    const uint j            = thread_position_in_grid.x;        // global column index
    const uint g_j          = threadgroup_position_in_grid.x;   // thread(g)roup column index in grid
    const uint l_j          = thread_position_in_threadgroup.x; // (l)ocal column index in threadgroup
    const uint g_cols       = threads_per_threadgroup.x;        // number of columns in threadgroup
    const uint g_horz_count = threadgroups_per_grid.x;          // number of threadgroups in j-direction
    
    assert( g_rows == threads_per_threadgroup.y);
    assert( g_cols == threads_per_threadgroup.x);
    
    // Important assumptions:
    // 1.)  panel_M == 4 * g_size.x
    // 2.)  panel_N == 4 * g_size.y
    // 3.)  sizeof(float) * ( panel_M * panel_K + panel_N * panel_K ) <= 32768
    // 3.a) panel_M * panel_K + panel_N * panel_K <= 32768/sizeof(float)
    // 3.b) panel_K <= 8192/(panel_M + panel_N);


    constexpr uint panel_M = 4 * g_rows;
    constexpr uint panel_N = 4 * g_cols;
    constexpr uint panel_K = ((32768/sizeof(float))/(panel_M + panel_N))/2; // TODO: Why division by 2?

    threadgroup float A_shared[panel_M][panel_K];
    threadgroup float B_shared[panel_K][panel_N];
    
    const uint m_step = g_vert_count * panel_M;
    const uint n_step = g_horz_count * panel_N;

    // These are g_vert_count iterations.
    for( uint m_begin = g_i * panel_M; m_begin < M; m_begin += m_step )
    {
        // Tile offset in X dimension

        const uint m_upper    = m_begin + panel_M;
        const bool m_complete = m_upper <= M;
        const uint m_end      = m_complete ? m_upper : M;

        // These are g_horz_count iterations.
        for( uint n_begin = g_j * panel_N; n_begin < N; n_begin += n_step )
        {
            // Tile offset in Y dimension

            const uint n_upper    = n_begin + panel_N;
            const bool n_complete = n_upper <= N;
            const uint n_end      = n_complete ? n_upper : N;
            
            //Initialize registers to zero:
            float A_reg[4]    =  {static_cast<float>(0)};
            float B_reg[4]    =  {static_cast<float>(0)};
            
            // Each thread takes care of a 4 x 4 block.
            float C_reg[4][4] = {{static_cast<float>(0)}};


            for( uint k_begin = 0; k_begin < K; k_begin += panel_K )
            {
                //k_begin is the position of the CUDA thread along the K dimension
                const uint k_upper    = k_begin + panel_K;
                const bool k_complete = k_upper<=K;
                const uint k_end      = k_complete ? k_upper : K;

                // TODO: Maybe swap the two for loops?

                //Load a tile of matrix A(m_begin:panel_M, k_begin:panel_K).
                // Each thread loads 4 x 4 elements into A_buf.
                // But read is scattered for coalescence!
                
                for( uint a = 0; a < 4; ++a )
                {
                    for( uint b = 0; b < 4; ++b )
                    {
                        A_shared[4 * l_i + a][4 * l_j + b] = A[K*(4 * i + a) + b];
                    }
                }
//
//                for( uint m = m_begin + l_i; m < m_end; m += g_rows )
//                {
//                    for( uint k = k_begin + l_j; k < k_end; k += g_cols )
//                    {
//                        A_shared[m - m_begin][k - k_begin] = A[K*m + k];
//                    }
//                }

                //Load a tile of matrix B(k_begin:panel_K, n_begin:panel_N):
                for( uint k = k_begin + l_id.x; k < k_end; k += g_size_x )
                {
                    for( uint n = n_begin + l_id.y; n < n_end; n += g_size_y )
                    {
                        B_shared[k - k_begin][n - n_begin] = B[N*k + n];
                    }
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

                //Multiply two loaded tiles to produce a tile of matrix C(m_begin:panel_M,n_begin:panel_N):
                if( k_complete )
                {
                    for( uint l = 0; l < panel_K; ++l )
                    {
                        for( uint i = 0; i < 4; ++i )
                        {
                            A_reg[i] = A_shared[l][l_id.x + g_size_x*i];
                        }

                        for( uint j = 0; j < 4; ++j )
                        {
                            B_reg[j] = B_shared[l_id.y + g_size_y*j][l];
                        }

                        for( uint j = 0; j < 4; ++j )
                        {
                            for( uint i = 0; i < 4; ++i )
                            {
                                C_reg[j][i] += A_reg[i] * B_reg[j];
                            }
                        }
                    }
                }
                else
                {
//                    // TODO: Write this.
                }

                threadgroup_barrier(mem_flags::mem_threadgroup);

            } //k_begin
            
            //Store elements of the C matrix in global memory:
            for( uint i = 0; i < 4; ++i )
            {
                const uint m = (m_begin + l_id.x + g_size_x * i );

                // TODO: Vectorize the store operation.
                for( uint j = 0; j < 4; ++j )
                {
                    const uint n = (n_begin + l_id.y + g_size_y * j );

                    C[m*N + n] = alpha * C_reg[i][j] + beta * C[m*N + n];
                }
            }


        } //m_begin

    } //n_begin

    return;
}
// FIXME: Comment-in the following line for run-time compilation:
)"
