R"(
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c Helmholtz.metal -o Helmholtz.air && xcrun -sdk macosx metallib Helmholtz.air -o Helmholtz.metallib

// FIXME: Comment-out the following line for run-time compilation:

// FIXME: We use "block_size" and "n_waves" as "template parameters" for jit-compilation.
// FIXME: Comment-in the following two lines for run-time compilation:
//constant constexpr uint block_size   = 64;
//constant constexpr uint n_waves      = 32;

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


[[max_total_threads_per_threadgroup(block_size)]]
[[kernel]] void Helmholtz__Neumann_to_Dirichlet4(
    const constant float3   * const mid_points       [[buffer(0)]],
    const constant float4x2 * const g_B              [[buffer(1)]],
          device   float4x2 * const g_C              [[buffer(2)]],
    const constant float    &       kappa            [[buffer(3)]],
    const constant float    &       kappa_step       [[buffer(4)]],
    const constant uint     &       n                [[buffer(5)]],
                                   
    const uint i_loc                    [[thread_position_in_threadgroup]],
    const uint i                        [[thread_position_in_grid]],
    const uint threads_per_threadgroup  [[threads_per_threadgroup]]
)
{
    assert( block_size == threads_per_threadgroup );
    
    // number of block
    const uint block_count = (n + block_size - 1) / block_size;

    constexpr uint K = n_waves >> 2;
    
    // each thread in the threadgroup gets one target point x assigned.
    thread float3 x_i;
    
    thread float2 A_i[block_size]; // stores exp( I * kappa * r_ij)/r for j in threadgroup
    
    // Each thread maintains one row of the output matrix.
    
    thread float4x2 C_i [K] = {{zero}};
    
    // Each thread loads the x-data for itself only once.
    x_i = mid_points[i];
    
    for( uint block = 0; block < block_count; ++block )
    {
        // Compute Helmholtz kernel for the current tile of size block_size x block_size.
        {
            // Each thread lets its x interact with a bunch of ys loaded into threadgroup.
            threadgroup float3 y[block_size];

            // Each thread in the threadgroup loads 1 entry of y.
            {
                const uint j_loc  = i_loc;
                const uint j      = block_size * block + j_loc;
                y[j_loc] = mid_points[j];
            }

            // need synchronization after loading data
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for( uint j_loc = 0; j_loc < block_size; ++j_loc )
            {
                const uint j = block_size * block + j_loc;

                // We ignore the results if i and j coincide or if one of i or j are invalid.
                const float delta = static_cast<float>( (i == j) );

                const float r = distance( x_i, y[j_loc] ) + delta;

                const float r_inv = one_over_four_pi * (one - delta) / r;

                float cos_kappa_r;
                float sin_kappa_r = sincos( kappa * r, cos_kappa_r );
                
                A_i[j_loc][0] = cos_kappa_r * r_inv;
                A_i[j_loc][1] = sin_kappa_r * r_inv;

            } // for( uint j_loc = 0; j_loc < block_size; ++j_loc )
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Now Re_A_i, Im_A_i are pre-computed.
        // Space reserved for the ys is freed again.
        
        // Now we do the actual matrix-matrix multiplication.
        {
            // The rows of the input matrix belonging to threadgroup.
            threadgroup float4x2 B[block_size][K];

            
            // TODO: Pad rows of C and B for alignment
            // Each thread in threadgroup loads 1 row of B.
            {
                const uint j_loc  = i_loc;
                const uint j      = block_size * block + j_loc;
                
                for( uint k = 0; k < K; ++k )
                {
                    B[j_loc][k] = g_B[K*j+k];
                }
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
                    
            
            for( uint j_loc = 0; j_loc < block_size; ++j_loc )
            {
                for( uint k = 0; k < K; ++k )
                {
                    for( uint l = 0; l < 4; ++l )
                    {
                        C_i[k][l][0] +=
                            A_i[j_loc][0] * B[j_loc][k][l][0]
                            -
                            A_i[j_loc][1] * B[j_loc][k][l][0];
                        
                        C_i[k][l][1] +=
                            A_i[j_loc][0] * B[j_loc][k][l][1]
                            +
                            A_i[j_loc][1] * B[j_loc][k][l][1];
                    }
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        

    } // for( uint block = 0; block < block_count; ++block )
    
    // Finally we can write C_i to C_global; each threads takes care of its own row.
    
    for( uint k = 0; k < K; ++k )
    {
        g_C[K*i+k] = C_i[k];
    }
    
} // Helmholtz__Neumann_to_Dirichlet4

// FIXME: Comment-out the following line for run-time compilation:
)"
