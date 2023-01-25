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

inline void cfma(
    const  float   Re_a, const  float   Im_a,
    const  float   Re_b, const  float   Im_b,
    thread float & Re_c, thread float & Im_c
)
{
    Re_c += Re_a * Re_b - Im_a * Im_b;
    Im_c += Re_a * Im_b + Im_a * Re_b;
}

inline void cmulby(
    thread float & Re_a, thread float & Im_a,
    const  float   Re_b, const  float   Im_b
)
{
    const float Re_c = Re_a * Re_b - Im_a * Im_b;
    const float Im_c = Re_a * Im_b + Im_a * Re_b;

    Re_a = Re_c;
    Im_a = Im_c;
}


[[max_total_threads_per_threadgroup(chunk_size)]]
[[kernel]] void Helmholtz__Neumann_to_Dirichlet(
    const device float * const mid_points         [[buffer(0)]],
    const device float * const Re_Y_global        [[buffer(1)]],
    const device float * const Im_Y_global        [[buffer(2)]],
          device float * const Re_X_global        [[buffer(3)]],
          device float * const Im_X_global        [[buffer(4)]],
    const device float &       kappa              [[buffer(5)]],
    const device float &       kappa_step         [[buffer(6)]],
    const device uint  &       n                  [[buffer(7)]],
                                   
    const uint i_loc                    [[thread_position_in_threadgroup]],
    const uint i                        [[thread_position_in_grid]],
    const uint threads_per_threadgroup  [[threads_per_threadgroup]]
)
{
    assert( chunk_size == threads_per_threadgroup );
    
    // number of chunks
    const uint chunk_count = (n + chunk_size - 1)/chunk_size;

    // each thread in the threadgroup gets one target point x assigned.
    thread float x_i [3];
    
    thread float Re_A_i[chunk_size]; // stores real of exp( I * kappa * r_ij)/r for j in threadgroup
    thread float Im_A_i[chunk_size]; // stores imag of exp( I * kappa * r_ij)/r for j in threadgroup
    
    // Each thread maintains one row of the output matrix.
    thread float Re_X_i [n_waves] = {};
    thread float Im_X_i [n_waves] = {};
    
    // Each thread loads the x-data for itself only once.
    const bool i_valid = i < n;

    if( i_valid )
    {
        x_i[0] = mid_points[3*i+0];
        x_i[1] = mid_points[3*i+1];
        x_i[2] = mid_points[3*i+2];
    }
    else
    {
        x_i[0] = zero;
        x_i[1] = zero;
        x_i[2] = zero;
    }
    
    for( uint chunk = 0; chunk < chunk_count; ++chunk )
    {
        // Compute Helmholtz kernel for the current tile of size chunk_size x chunk_size.
        {
            // Each thread lets its x interact with a bunch of ys loaded into threadgroup.
            threadgroup float y[chunk_size][3];

            // Each thread in the threadgroup loads 1 entry of y.
            {
                const uint j_loc  = i_loc;
                const uint j      = chunk_size * chunk + j_loc;

                const bool j_valid = j < n;

                if( j_valid )
                {
                    y[j_loc][0] = mid_points[3*j+0];
                    y[j_loc][1] = mid_points[3*j+1];
                    y[j_loc][2] = mid_points[3*j+2];
                }
                else
                {
                    y[j_loc][0] = zero;
                    y[j_loc][1] = zero;
                    y[j_loc][2] = zero;
                }
            }

            // need synchronization after loading data
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Each thread computes entries of Re_A_i, Im_A_i, Re_B_i, Im_B_i for its i and all js in the chunk.
            for( uint j_loc = 0; j_loc < chunk_size; ++j_loc )
            {
                const uint j = chunk_size * chunk + j_loc;

                const bool j_valid = j < n;

                // We ignore the results if i and j coincide or if one of i or j are invalid.
                const float delta = static_cast<float>( (!i_valid) || (!j_valid) || (i == j) );

                const float z[3] = { y[j_loc][0] - x_i[0], y[j_loc][1] - x_i[1], y[j_loc][2] - x_i[2] };

                const float r = sqrt( z[0] * z[0] + z[1] * z[1] + z[2] * z[2] ) + delta;

                const float r_inv = one_over_four_pi * (one - delta) / r;

                Re_A_i[j_loc] = cos( kappa * r ) * r_inv;
                Im_A_i[j_loc] = sin( kappa * r ) * r_inv;


            } // for( uint j_loc = 0; j_loc < chunk_size; ++j_loc )

        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Now Re_A_i, Im_A_i are pre-computed.
        // Space reserved for the ys is freed again.
        
        // Now we do the actual matrix-matrix multiplication.
        {
            // The rows of the input matrix belonging to threadgroup.
            threadgroup float Re_Y[chunk_size][n_waves];
            threadgroup float Im_Y[chunk_size][n_waves];
            
            // TODO: Vectorize load operations.
            // TODO: Pad rows of X and Y for alignment
            // TODO:  Add some rows to X and Y -> can avoid j_valid and i_valid.
            // Each thread in threadgroup loads 1 row of Y.
            {
                const uint j_loc  = i_loc;
                const uint j      = chunk_size * chunk + j_loc;
                
                const bool j_valid = j < n;
                
                if( j_valid )
                {
                    for( uint k = 0; k < n_waves; ++k )
                    {
                        Re_Y[j_loc][k] = Re_Y_global[n_waves*j+k];
                        Im_Y[j_loc][k] = Im_Y_global[n_waves*j+k];
                    }
                }
                else
                {
                    for( uint k = 0; k < n_waves; ++k )
                    {
                        Re_Y[j_loc][k] = zero;
                        Im_Y[j_loc][k] = zero;
                    }
                }
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for( uint j_loc = 0; j_loc < chunk_size; ++j_loc )
            {
                for( uint k = 0; k < n_waves; ++k )
                {
                    cfma(
                        Re_A_i [j_loc], Im_A_i [j_loc],
                        Re_Y[j_loc][k], Im_Y[j_loc][k],
                        Re_X_i     [k], Im_X_i     [k]
                    );
                }
            }
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        

    } // for( uint chunk = 0; chunk < chunk_count; ++chunk )
    
    // Finally we can write X_i to X_global; each threads takes care of its own row.
    

    
//    // Vectorized store operations.
//    if( i_valid )
//    {
//        thread float4 * Re_X_from = reinterpret_cast<thread float4 *>( &Re_X_i[0] );
//        thread float4 * Im_X_from = reinterpret_cast<thread float4 *>( &Im_X_i[0] );
//
//        device float4 * Re_X_to   = reinterpret_cast<device float4 *>( Re_X_global + n_waves * i);
//        device float4 * Im_X_to   = reinterpret_cast<device float4 *>( Im_X_global + n_waves * i);
//
//        const uint step_count = n_waves/4;
//        for( uint k = 0; k < step_count; ++k )
//        {
//            Re_X_to[k] = Re_X_from[k];
//            Im_X_to[k] = Im_X_from[k];
//        }
//    }
    
    // TODO: Vectorize store operations.
    if( i_valid )
    {
        for( uint k = 0; k < n_waves; ++k )
        {
            Re_X_global[n_waves*i+k] = Re_X_i[k];
            Im_X_global[n_waves*i+k] = Im_X_i[k];
        }
    }
    
} // Helmholtz__Neumann_to_Dirichlet

// FIXME: Comment-out the following line for run-time compilation:
)"
