R"(
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c Helmholtz.metal -o Helmholtz.air && xcrun -sdk macosx metallib Helmholtz.air -o Helmholtz.metallib

// FIXME: Comment-out the following line for run-time compilation:

// FIXME: We use "chunk_size" and "n_waves" as "template parameters" for jit-compilation.
// FIXME: Comment-in the following two lines for run-time compilation:
//constant constexpr uint chunk_size   = 64;
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


using Row_T = array<float,n_waves>;

[[max_total_threads_per_threadgroup(chunk_size)]]
[[kernel]] void Helmholtz__Neumann_to_Dirichlet(
    const constant float3 * const mid_points         [[buffer(0)]],
    const constant float4 * const Re_Y_global        [[buffer(1)]],
    const constant float4 * const Im_Y_global        [[buffer(2)]],
          device   float4 * const Re_X_global        [[buffer(3)]],
          device   float4 * const Im_X_global        [[buffer(4)]],
    const constant float  &       kappa              [[buffer(5)]],
    const constant float  &       kappa_step         [[buffer(6)]],
    const constant uint   &       n                  [[buffer(7)]],
                                   
    const uint i_loc                    [[thread_position_in_threadgroup]],
    const uint i                        [[thread_position_in_grid]],
    const uint threads_per_threadgroup  [[threads_per_threadgroup]]
)
{
    assert( chunk_size == threads_per_threadgroup );
    
    // number of chunks
    const uint chunk_count = (n + chunk_size - 1) / chunk_size;

    constexpr uint K = n_waves / 4;
    
    // each thread in the threadgroup gets one target point x assigned.
    thread float3 x_i;
    
    thread float Re_A_i[chunk_size]; // stores real of exp( I * kappa * r_ij)/r for j in threadgroup
    thread float Im_A_i[chunk_size]; // stores imag of exp( I * kappa * r_ij)/r for j in threadgroup
    
    // Each thread maintains one row of the output matrix.
    
    thread float4 Re_X_i [K] = {};
    thread float4 Im_X_i [K] = {};
    
//    thread Row_T Re_X_i = {};
//    thread Row_T Im_X_i = {};
    
    // Each thread loads the x-data for itself only once.
    x_i = mid_points[i];
    
    for( uint chunk = 0; chunk < chunk_count; ++chunk )
    {
        // Compute Helmholtz kernel for the current tile of size chunk_size x chunk_size.
        {
            // Each thread lets its x interact with a bunch of ys loaded into threadgroup.
            threadgroup float3 y[chunk_size];

            // Each thread in the threadgroup loads 1 entry of y.
            {
                const uint j_loc  = i_loc;
                const uint j      = chunk_size * chunk + j_loc;
                y[j_loc] = mid_points[j];
            }

            // need synchronization after loading data
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for( uint j_loc = 0; j_loc < chunk_size; ++j_loc )
            {
                const uint j = chunk_size * chunk + j_loc;

                // We ignore the results if i and j coincide or if one of i or j are invalid.
                const float delta = static_cast<float>( (i == j) );

                const float r = distance( x_i, y[j_loc] ) + delta;

                const float r_inv = one_over_four_pi * (one - delta) / r;

                float cos_kappa_r;
                float sin_kappa_r = sincos( kappa * r, cos_kappa_r );
                
                Re_A_i[j_loc] = cos_kappa_r * r_inv;
                Im_A_i[j_loc] = sin_kappa_r * r_inv;

            } // for( uint j_loc = 0; j_loc < chunk_size; ++j_loc )

        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Now Re_A_i, Im_A_i are pre-computed.
        // Space reserved for the ys is freed again.
        
        // Now we do the actual matrix-matrix multiplication.
        {
            // The rows of the input matrix belonging to threadgroup.
            threadgroup float4 Re_Y[chunk_size][K];
            threadgroup float4 Im_Y[chunk_size][K];
            
//            threadgroup Row_T Re_Y[chunk_size];
//            threadgroup Row_T Im_Y[chunk_size];

            
            // TODO: Vectorize load operations.
            // TODO: Pad rows of X and Y for alignment
            // Each thread in threadgroup loads 1 row of Y.
            {
                const uint j_loc  = i_loc;
                const uint j      = chunk_size * chunk + j_loc;
                
                for( uint k = 0; k < K; ++k )
                {
                    Re_Y[j_loc][k] = Re_Y_global[K*j+k];
                    Im_Y[j_loc][k] = Im_Y_global[K*j+k];
                }
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
                    
            
            for( uint j_loc = 0; j_loc < chunk_size; ++j_loc )
            {
//                for( uint k = 0; k < n_waves; ++k )
                    for( uint k = 0; k < K; ++k )
                {
                    Re_X_i[k] += Re_A_i[j_loc] * Re_Y[j_loc][k] - Im_A_i[j_loc] * Im_Y[j_loc][k];
                    Im_X_i[k] += Re_A_i[j_loc] * Im_Y[j_loc][k] + Im_A_i[j_loc] * Re_Y[j_loc][k];
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        

    } // for( uint chunk = 0; chunk < chunk_count; ++chunk )
    
    // Finally we can write X_i to X_global; each threads takes care of its own row.
    
    for( uint k = 0; k < K; ++k )
    {
        Re_X_global[K*i+k] = Re_X_i[k];
        Im_X_global[K*i+k] = Im_X_i[k];
    }
    
} // Helmholtz__Neumann_to_Dirichlet

// FIXME: Comment-out the following line for run-time compilation:
)"
