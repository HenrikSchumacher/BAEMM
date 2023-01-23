
// We have run the following command in the terminal:

//  xcrun -sdk macosx metal -c Helmholtz.metal -o Helmholtz.air && xcrun -sdk macosx metallib Helmholtz.air -o Helmholtz.metallib

#include <metal_stdlib>

using namespace metal;

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


// CAUTION: We require that chunk_size == threads_per_threadgroup!!!
template<uint chunk_size, uint n_waves>
[[max_total_threads_per_threadgroup(chunk_size)]]
[[kernel]] void Helmholtz_Multiply(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                                   
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
)
{
    constexpr float zero     = static_cast<float>(0);
    constexpr float one      = static_cast<float>(1);
//    constexpr float two      = static_cast<float>(2);
//    constexpr float one_half = one / two;
    
    // number of chunks
    const uint chunk_count = (n + chunk_size - 1)/chunk_size;

    // each thread in the threadgroup gets one target point x assigned.
    thread float x_i [3];
    
    thread float Re_A_i[chunk_size]; // stores real of exp( I * kappa * r_ij)/r for j in threadgroup
    thread float Im_A_i[chunk_size]; // stores imag of exp( I * kappa * r_ij)/r for j in threadgroup
    
    thread float Re_B_i[chunk_size];
    thread float Im_B_i[chunk_size];
    
    // Each thread maintains how row of the output matrix.
    thread float Re_X_i [n_waves] = {};
    thread float Im_X_i [n_waves] = {};
    
    // Each thread loads the x-data for itself only once.
    const bool i_valid = i < n;

    if( i_valid )
    {
        #pragma clang unroll(3)
        for( uint k = 0; k < 3; ++k )
        {
            x_i[k] = mid_points[3*i+k];
        }
    }
    else
    {
        #pragma clang unroll(3)
        for( uint k = 0; k < 3; ++k )
        {
            x_i[k] = static_cast<float>(0);
        }
    }
    
    
    for( uint chunk = 0; chunk < chunk_count; ++chunk )
    {
        // CAUTION: We require that chunk_size == threads_per_threadgroup!!!
        
        // Compute Helmholtz kernel the current tile of size chunk_size x chunk_size.
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
                    #pragma clang unroll(3)
                    for( uint k = 0; k < 3; ++k )
                    {
                        y[j_loc][k] = mid_points[3*j+k];
                    }
                }
                else
                {
                    #pragma clang unroll(3)
                    for( uint k = 0; k < 3; ++k )
                    {
                        y[j_loc][k] = zero;
                    }
                }
                
                
                // need synchronization after loading data
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            // Each thread computes entries of Re_A_i, Im_A_i, Re_B_i, Im_B_i for its i and all js in the chunk.
            for( uint j_loc = 0; j_loc < chunk_size; ++j_loc )
            {
                const uint j = chunk_size * chunk + j_loc;
                
                const bool j_valid = j < n;
                
                // We ignore the results if i and j coincide or if one of i or j are invalid.
                const float delta = static_cast<float>( (!i_valid) || (!j_valid) || (i == j) );
                
                const float z[3] = { y[j_loc][0] - x_i[0], y[j_loc][1] - x_i[1], y[j_loc][2] - x_i[2] };
                
                const float r = sqrt( z[0] * z[0] + z[1] * z[1] + z[2] * z[2] ) + delta;
                
                const float r_inv = (one - delta) / r;
                
                Re_A_i[j_loc] = cos( kappa * r ) * r_inv;
                Im_A_i[j_loc] = sin( kappa * r ) * r_inv;
                
                Re_B_i[j_loc] = cos( kappa_step * r );
                Im_B_i[j_loc] = sin( kappa_step * r );
                
            } // for( uint j_loc = 0; j_loc < chunk_size; ++j_loc )
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        
        // Now Re_A_i, Im_A_i, Re_B_i, Im_B_i are computed.
        // Space reserved for the ys is freed again.
        
        // Now we do the actual matrix-matrix multiplication.
        {
            // The rows of the input matrix belonging to threadgroup.
            threadgroup float Re_Y[chunk_size][n_waves];
            threadgroup float Im_Y[chunk_size][n_waves];
            
            // Each thread in threadgroup loads 1 row of Y.
            {
                const uint j_loc  = i_loc;
                const uint j      = chunk_size * chunk + j_loc;
                
                const bool j_valid = j < n;
                
                if( j_valid )
                {
                    #pragma clang unroll(n_waves)
                    for( uint k = 0; k < n_waves; ++k )
                    {
                        Re_Y[j_loc][k] = Re_Y_global[n_waves*j+k];
                        Im_Y[j_loc][k] = Im_Y_global[n_waves*j+k];
                    }
                }
                else
                {
                    #pragma clang unroll(n_waves)
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
                cfma(
                    Re_A_i [j_loc], Im_A_i [j_loc],
                    Re_Y[j_loc][0], Im_Y[j_loc][0],
                    Re_X_i     [0], Im_X_i     [0]
                );
                
                #pragma clang unroll(n_waves-1)
                for( uint k = 1; k < n_waves; ++k )
                {
                    cmulby(
                        Re_A_i[j_loc],  Im_A_i[j_loc],
                        Re_B_i[j_loc],  Im_B_i[j_loc]
                    );
                    
                    cfma(
                        Re_A_i [j_loc], Im_A_i [j_loc],
                        Re_Y[j_loc][k], Im_Y[j_loc][k],
                        Re_X_i     [k], Im_X_i     [k]
                    );
                }
                
            } // for( uint j_loc = 0; j_loc < chunk_size; ++j_loc )
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        

    } // for( uint chunk = 0; chunk < chunk_count; ++chunk )
    
    // Finally we can write X_i to X_global; each threads takes care of its own row.
    
    if( i_valid )
    {
        #pragma clang unroll unroll(n_waves)
        for( uint k = 0; k < n_waves; ++k )
        {
            Re_X_global[n_waves*i+k] = Re_X_i[k];
            Im_X_global[n_waves*i+k] = Im_X_i[k];
        }
    }
    
} // Helmholtz_Multiply





// Templates have to be instantiated here in order to be used.

template [[ host_name("Helmholtz_Multiply_8_1") ]] kernel void Helmholtz_Multiply<8,1>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                                   
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_16_1") ]] kernel void Helmholtz_Multiply<16,1>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                                   
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_32_1") ]] kernel void Helmholtz_Multiply<32,1>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                                   
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_64_1") ]] kernel void Helmholtz_Multiply<64,1>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                            
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_128_1") ]] kernel void Helmholtz_Multiply<128,1>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                            
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_256_1") ]] kernel void Helmholtz_Multiply<256,1>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                            
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);



template [[ host_name("Helmholtz_Multiply_8_8") ]] kernel void Helmholtz_Multiply<8,8>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                                   
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_16_8") ]] kernel void Helmholtz_Multiply<16,8>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                                   
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_32_8") ]] kernel void Helmholtz_Multiply<32,8>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                                   
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_64_8") ]] kernel void Helmholtz_Multiply<64,8>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                            
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_128_8") ]] kernel void Helmholtz_Multiply<128,8>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                            
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_256_8") ]] kernel void Helmholtz_Multiply<256,8>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                            
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);


template [[ host_name("Helmholtz_Multiply_8_16") ]] kernel void Helmholtz_Multiply<8,16>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                                   
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_16_16") ]] kernel void Helmholtz_Multiply<16,16>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                                   
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_32_16") ]] kernel void Helmholtz_Multiply<32,16>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                                   
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_64_16") ]] kernel void Helmholtz_Multiply<64,16>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                            
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_128_16") ]] kernel void Helmholtz_Multiply<128,16>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                            
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_8_32") ]] kernel void Helmholtz_Multiply<8,32>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                                   
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_16_32") ]] kernel void Helmholtz_Multiply<16,32>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                                   
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_32_32") ]] kernel void Helmholtz_Multiply<32,32>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                                   
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_64_32") ]] kernel void Helmholtz_Multiply<64,32>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                            
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);


template [[ host_name("Helmholtz_Multiply_8_64") ]] kernel void Helmholtz_Multiply<8,64>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                                   
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_16_64") ]] kernel void Helmholtz_Multiply<16,64>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                                   
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

template [[ host_name("Helmholtz_Multiply_32_64") ]] kernel void Helmholtz_Multiply<32,64>(
    const device float * __restrict__ const mid_points         [[buffer(0)]],
    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
          device float * __restrict__ const Re_X_global        [[buffer(3)]],
          device float * __restrict__ const Im_X_global        [[buffer(4)]],
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
                                   
    const uint i_loc   [[thread_position_in_threadgroup]],
    const uint i       [[thread_position_in_grid]]
);

//template [[ host_name("Helmholtz_Multiply_64_64") ]] kernel void Helmholtz_Multiply<64,64>(
//    const device float * __restrict__ const mid_points         [[buffer(0)]],
//    const device float * __restrict__ const Re_Y_global        [[buffer(1)]],
//    const device float * __restrict__ const Im_Y_global        [[buffer(2)]],
//          device float * __restrict__ const Re_X_global        [[buffer(3)]],
//          device float * __restrict__ const Im_X_global        [[buffer(4)]],
//    const device float & kappa                                 [[buffer(5)]],
//    const device float & kappa_step                            [[buffer(6)]],
//    const device uint  & n                                     [[buffer(7)]],
//
//
//    const uint i_loc   [[thread_position_in_threadgroup]],
//    const uint i       [[thread_position_in_grid]]
//);

