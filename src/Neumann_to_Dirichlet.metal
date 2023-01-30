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
[[kernel]] void Helmholtz__Neumann_to_Dirichlet(
    const constant float3 * const mid_points         [[buffer(0)]],
    const constant float4 * const Re_B_global        [[buffer(1)]],
    const constant float4 * const Im_B_global        [[buffer(2)]],
          device   float4 * const Re_C_global        [[buffer(3)]],
          device   float4 * const Im_C_global        [[buffer(4)]],
    const constant float  &       kappa              [[buffer(5)]],
    const constant uint   &       n                  [[buffer(6)]],
                                   
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
    
    thread float Re_A_i[block_size]; // stores real of exp( I * kappa * r_ij)/r for j in threadgroup
    thread float Im_A_i[block_size]; // stores imag of exp( I * kappa * r_ij)/r for j in threadgroup

    
    // Each thread maintains one row of the output matrix.
    thread float4 Re_C_i [K] = {zero};
    thread float4 Im_C_i [K] = {zero};
    
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
                
                Re_A_i[j_loc] = cos_kappa_r * r_inv;
                Im_A_i[j_loc] = sin_kappa_r * r_inv;

            } // for( uint j_loc = 0; j_loc < block_size; ++j_loc )
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Now Re_A_i, Im_A_i are pre-computed.
        // Space reserved for the ys is freed again.
        
        // Now we do the actual matrix-matrix multiplication.
        {
            // The rows of the input matrix belonging to threadgroup.
            threadgroup float4 Re_B[block_size][K];
            threadgroup float4 Im_B[block_size][K];
            
            // TODO: Pad rows of C and B for alignment
            // TODO: Coalesce loading!
            // Each thread in threadgroup loads 1 row of B.
            {
                const uint j_loc  = i_loc;
                const uint j      = block_size * block + j_loc;
                
                for( uint k = 0; k < K; ++k )
                {
                    Re_B[j_loc][k] = Re_B_global[K*j+k];
                    Im_B[j_loc][k] = Im_B_global[K*j+k];
                }
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
                    
            
            for( uint j_loc = 0; j_loc < block_size; ++j_loc )
            {
                for( uint k = 0; k < K; ++k )
                {
                    Re_C_i[k] += Re_A_i[j_loc] * Re_B[j_loc][k] - Im_A_i[j_loc] * Im_B[j_loc][k];
                    Im_C_i[k] += Re_A_i[j_loc] * Im_B[j_loc][k] + Im_A_i[j_loc] * Re_B[j_loc][k];
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        

    } // for( uint block = 0; block < block_count; ++block )
    
    // Finally we can write C_i to C_global; each threads takes care of its own row.
    
    // TODO: Coalesce writing by writing to shared memory first?
    for( uint k = 0; k < K; ++k )
    {
        Re_C_global[K*i+k] = Re_C_i[k];
        Im_C_global[K*i+k] = Im_C_i[k];
    }
    
} // Helmholtz__Neumann_to_Dirichlet

// FIXME: Comment-out the following line for run-time compilation:
)"
