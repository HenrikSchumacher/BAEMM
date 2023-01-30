R"(
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c Helmholtz.metal -o Helmholtz.air && xcrun -sdk macosx metallib Helmholtz.air -o Helmholtz.metallib

// FIXME: Comment-out the following line for run-time compilation:

// FIXME: We use "block_size" and "n_waves" as "template parameters" for jit-compilation.
// FIXME: Comment-in the following two lines for run-time compilation:
//constant constexpr int block_size   = 64;
//constant constexpr int n_waves      = 32;

#include <metal_stdlib>

using namespace metal;

constant constexpr float zero    = static_cast<float>(0);
constant constexpr float one     = static_cast<float>(1);
constant constexpr float two     = static_cast<float>(2);
//constant     constexpr float one_half = one / two;

constant constexpr float pi      = 3.141592653589793;
constant constexpr float two_pi  = two * pi;
constant constexpr float four_pi = two * two_pi;

//constant     constexpr float one_over_two_pi  = one / two_pi;
constant constexpr float one_over_four_pi = one / four_pi;

using vec_T = float;

[[max_total_threads_per_threadgroup(block_size)]]
[[kernel]] void Helmholtz__Neumann_to_Dirichlet(
    const constant float3 * const mid_points         [[buffer(0)]],
    const constant vec_T  * const Re_B_global        [[buffer(1)]],
    const constant vec_T  * const Im_B_global        [[buffer(2)]],
          device   vec_T  * const Re_C_global        [[buffer(3)]],
          device   vec_T  * const Im_C_global        [[buffer(4)]],
    const constant float  &       kappa              [[buffer(5)]],
    const constant int    &       n                  [[buffer(6)]],
                                   
    const uint thread_position_in_threadgroup  [[thread_position_in_threadgroup]],
    const uint thread_position_in_grid         [[thread_position_in_grid]],
    const uint threads_per_threadgroup         [[threads_per_threadgroup]],
    const uint threadgroup_position_in_grid    [[threadgroup_position_in_grid]]
)
{
    assert( block_size == threads_per_threadgroup );
    
    const int i_loc = thread_position_in_threadgroup;
    const int i     = thread_position_in_grid;
    
    // number of block
    const int block_count = (n + block_size - 1) / block_size;

    constexpr int K = n_waves / (sizeof(vec_T)/sizeof(float));
    
    // each thread in the threadgroup gets one target point x assigned.
    thread float3 x_i;
    
    thread float Re_A_i[block_size]; // stores real of exp( I * kappa * r_ij)/r for j in threadgroup
    thread float Im_A_i[block_size]; // stores imag of exp( I * kappa * r_ij)/r for j in threadgroup

    
    // Each thread maintains one row of the output matrix.
    thread vec_T Re_C_i [K] = {zero};
    thread vec_T Im_C_i [K] = {zero};
    
    // The rows of the input matrix belonging to threadgroup.
    threadgroup vec_T Re_B[block_size][K];
    threadgroup vec_T Im_B[block_size][K];
    
    // Each thread loads the x-data for itself only once.
    x_i = mid_points[i];
    
    for( int block = 0; block < block_count; ++block )
    {
        // Compute Helmholtz kernel for the current tile of size block_size x block_size.
        {
            // Each thread lets its x interact with a bunch of ys loaded into threadgroup.
            threadgroup float3 y[block_size];

            // Each thread in the threadgroup loads 1 entry of y.
            {
                const int j_loc  = i_loc;
                const int j      = block_size * block + j_loc;
                y[j_loc] = mid_points[j];
            }

            // need synchronization after loading data
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for( int j_loc = 0; j_loc < block_size; ++j_loc )
            {
                const int j = block_size * block + j_loc;

                // We ignore the results if i and j coincide or if one of i or j are invalid.
                const float delta = static_cast<float>( (i == j) );

                const float r = distance( x_i, y[j_loc] );

                const float r_inv = one_over_four_pi * (one - delta) / (r + delta);

                float cos_kappa_r;
                float sin_kappa_r = sincos( kappa * r, cos_kappa_r );
                
                Re_A_i[j_loc] = cos_kappa_r * r_inv;
                Im_A_i[j_loc] = sin_kappa_r * r_inv;

            } // for( int j_loc = 0; j_loc < block_size; ++j_loc )
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Now Re_A_i, Im_A_i are pre-computed.
        // Space reserved for the ys is freed again.
        
        // Now we do the actual matrix-matrix multiplication.
        {
//            // Coalesced(?) load.
//            constant    const vec_T * Re_from = &Re_B_global[K * block_size * block];
//            constant    const vec_T * Im_from = &Im_B_global[K * block_size * block];
//
//            threadgroup       vec_T * Re_to   = &Re_B[0][0];
//            threadgroup       vec_T * Im_to   = &Im_B[0][0];
//
//            for( int k = 0; k < block_size * K; k += block_size )
//            {
//                Re_to[k + i_loc] = Re_from[k + i_loc];
//                Im_to[k + i_loc] = Im_from[k + i_loc];
//            }
            
            // TODO: Pad rows of C and B for alignment
            // Each thread in threadgroup loads 1 row of B.
            {
                const int j_loc  = i_loc;
                const int j      = block_size * block + j_loc;

                for( int k = 0; k < K; ++k )
                {
                    Re_B[j_loc][k] = Re_B_global[K*j+k];
                    Im_B[j_loc][k] = Im_B_global[K*j+k];
                }
            }

            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for( int j_loc = 0; j_loc < block_size; ++j_loc )
            {
                for( int k = 0; k < K; ++k )
                {
                    Re_C_i[k] += Re_A_i[j_loc] * Re_B[j_loc][k] - Im_A_i[j_loc] * Im_B[j_loc][k];
                    Im_C_i[k] += Re_A_i[j_loc] * Im_B[j_loc][k] + Im_A_i[j_loc] * Re_B[j_loc][k];
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        

    } // for( int block = 0; block < block_count; ++block )
    
//    // In order to achieve coalesced write, we use threadgroup memory Re_B, Im_B as intermediate buffer.
//    // Each thread writes it row into shared memory.
//    for( int k = 0; k < K; ++k )
//    {
//        Re_B[i_loc][k] = Re_C_i[k];
//        Im_B[i_loc][k] = Im_C_i[k];
//    }
//
//    threadgroup const vec_T * Re_from = &Re_B[0][0];
//    threadgroup const vec_T * Im_from = &Im_B[0][0];
//
//    device            vec_T * Re_to   = &Re_C_global[K * block_size * threadgroup_position_in_grid];
//    device            vec_T * Im_to   = &Im_C_global[K * block_size * threadgroup_position_in_grid];
//
//    // Coalesced(?) write to output.
//    for( int k = 0; k < block_size * K; k += block_size )
//    {
//        Re_to[k + i_loc] = Re_from[k + i_loc];
//        Im_to[k + i_loc] = Im_from[k + i_loc];
//    }
    
    // Finally we can write C_i to C_global; each threads takes care of its own row.
    for( int k = 0; k < K; ++k )
    {
        Re_C_global[K*i+k] = Re_C_i[k];
        Im_C_global[K*i+k] = Im_C_i[k];
    }
    
} // Helmholtz__Neumann_to_Dirichlet

// FIXME: Comment-out the following line for run-time compilation:
)"
