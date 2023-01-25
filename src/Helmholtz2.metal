R"(
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c Helmholtz2.metal -o Helmholtz2.air && xcrun -sdk macosx metallib Helmholtz2.air -o Helmholtz2.metallib

// FIXME: Comment-out the following line for run-time compilation:

// FIXME: We use "chunk_size" and "n_waves" as "template parameters" for jit-compilation.
// FIXME: Comment-in the following two lines for run-time compilation:

constant constexpr uint M_threads = 8;
constant constexpr uint N_threads = 8;

constant constexpr uint M_tile   = 4;
constant constexpr uint N_tile   = 4;

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
    const device float & kappa                                 [[buffer(5)]],
    const device float & kappa_step                            [[buffer(6)]],
    const device uint  & n                                     [[buffer(7)]],
                                   
    const uint p                        [[thread_position_in_threadgroup]],
    const uint p_global                 [[thread_position_in_grid]],
    const uint q                        [[threadgroup_position_in_grid]],
    const uint threads_per_threadgroup  [[threads_per_threadgroup]]
)
{
    assert( threads_per_threadgroup == 64 );
    
    
    //    p.x in [0,...,8[
    //    p.y in [0,...,8[
    
    threadgroup float Re_A_shared [32][32];
    threadgroup float Im_A_shared [32][32];
    
    threadgroup float Re_B_shared [32][32];
    threadgroup float Im_B_shared [32][32];
    
    // Compute 32 x 32 tile of A_shared.
    {
        // first 32 entries are x
        // last  32 entries are y
        threadgroup float pts_shared [32*2][3];
        
        // Load the 64 points into threadgroup.
        const uint id = 8 * p.x + p.y;
        const uint from = (id < 32) ? 32 * q.x + id : 32 * q.y + (id - 32);
        
        if( from < n )
        {
            pts_shared[id][0] = mid_points[from][0];
            pts_shared[id][1] = mid_points[from][1];
            pts_shared[id][2] = mid_points[from][2];
        }
        else
        {
            pts_shared[id][0] = 0;
            pts_shared[id][1] = 0;
            pts_shared[id][2] = 0;
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        
        // Each thread computes a 4 x 4 tile of A_shared.
        // Note: We store the _transpose_ of A because that will make the matrix-matrix porduct easier!
        for( uint j_ = 0; j_ < 4; ++j_ )
        {
            const uint j = 4 * p.y + i_;
            const bool j_valid = j < n;
            
            thread const float y[3];
            
            y[0] = pts_shared[j][0];
            y[1] = pts_shared[j][1];
            y[2] = pts_shared[j][2];
            
            for( uint i_ = 0; i_ < 4; ++i_ )
            {
                const uint i = 4 * p.x + i_;
                const bool i_valid = i < n;
                
                // We ignore the results if i and j coincide or if one of i or j are invalid.
                const float delta = static_cast<float>( (!i_valid) || (!j_valid) || (i == j) );
                
                thread const float z[3] = {
                    y[0] - pts_shared[32+i][0], y[1] - pts_shared[32+i][1], y[2] - pts_shared[32+i][2]
                };
                
                const float r = sqrt( z[0] * z[0] + z[1] * z[1] + z[2] * z[2] );
                
                const float factor = one_over_four_pi * (one - delta) / (r + delta);
                
                Re_A_shared[j][i] = cos( kappa * r ) * factor;
                Im_A_shared[j][i] = sin( kappa * r ) * factor;
            }
        }
        
        // Each thread loads a 4 x 4 tile of B_shared.
        for( uint i_ = 0; i_ < 4; ++i_ )
        {
            const uint i = 4 * p.x + i_;
            const bool i_valid = i < n;
            
            for( uint j_ = 0; j_ < 4; ++j_ )
            {
                const uint j = 4 * p.j + j_;
                
                Re_B_shared[i][j] = Re_Y_global[32*( +i) +j]
                Im_B_shared[i][j] = Im_Y_global[32*( +i) +j];
            }
        }
     
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
      
    {
        
        
        thread float Re_A_k [4];
        thread float Im_A_k [4];
        
        thread float Re_B_k [4];
        thread float Im_B_k [4];
        
        thread float Re_C [4][4] = {{}};
        thread float Im_C [4][4] = {{}};
        
        
        for( uint k = 0; k < 32; ++k )
        {
            for( uint i = 0; i < 4; ++i )
            {
                A_k[i] = A_shared[4*p.x+i][k];
            }

            for( UInt j = 0; j < 4; ++j )
            {
                B_k[j] = B_shared[k][4*p.y+j];
            }
            
            // C[i][j] += A[i][k] * B[k][j]
            for( uint i = 0; i < 4; ++i )
            {
                for( uint i = 0; i < 4; ++i )
                {
                    cfma(   Re_A_k[i], Im_A_k[i],   Re_B_k[j], Im_B_k[j],   Re_C[i][j], Im_C[i][j]   );
                }
            }
        }
            
        
        
    }
        
    threadgroup_barrier(mem_flags::mem_threadgroup);

    
//    const bool j_valid = j < n;
//
//    // We ignore the results if i and j coincide or if one of i or j are invalid.
//    const float delta = static_cast<float>( (!i_valid) || (!j_valid) || (i == j) );
//
//    const float z[3] = { y[j_loc][0] - x_i[0], y[j_loc][1] - x_i[1], y[j_loc][2] - x_i[2] };
//
//    const float r = sqrt( z[0] * z[0] + z[1] * z[1] + z[2] * z[2] ) + delta;
//
//    const float r_inv = one_over_four_pi * (one - delta) / r;
//
//    Re_A_i[j_loc] = cos( kappa * r ) * r_inv;
//    Im_A_i[j_loc] = sin( kappa * r ) * r_inv;
//
//    assert( chunk_size == threads_per_threadgroup );
//
//    // number of chunks
//    const uint chunk_count = (n + chunk_size - 1)/chunk_size;
//
//    // each thread in the threadgroup gets one target point x assigned.
//    thread float x_i [3];
//
//    thread float Re_A_i[chunk_size]; // stores real of exp( I * kappa * r_ij)/r for j in threadgroup
//    thread float Im_A_i[chunk_size]; // stores imag of exp( I * kappa * r_ij)/r for j in threadgroup
//
//    // Each thread maintains one row of the output matrix.
//    thread float Re_X_i [n_waves] = {};
//    thread float Im_X_i [n_waves] = {};
//
//    // Each thread loads the x-data for itself only once.
//    const bool i_valid = i < n;
//
//    if( i_valid )
//    {
//        x_i[0] = mid_points[3*i+0];
//        x_i[1] = mid_points[3*i+1];
//        x_i[2] = mid_points[3*i+2];
//    }
//    else
//    {
//        x_i[0] = zero;
//        x_i[1] = zero;
//        x_i[2] = zero;
//    }
//
//    for( uint chunk = 0; chunk < chunk_count; ++chunk )
//    {
//        // Compute Helmholtz kernel for the current tile of size chunk_size x chunk_size.
//        {
//            // Each thread lets its x interact with a bunch of ys loaded into threadgroup.
//            threadgroup float y[chunk_size][3];
//
//            // Each thread in the threadgroup loads 1 entry of y.
//            {
//                const uint j_loc  = i_loc;
//                const uint j      = chunk_size * chunk + j_loc;
//
//                const bool j_valid = j < n;
//
//                if( j_valid )
//                {
//                    y[j_loc][0] = mid_points[3*j+0];
//                    y[j_loc][1] = mid_points[3*j+1];
//                    y[j_loc][2] = mid_points[3*j+2];
//                }
//                else
//                {
//                    y[j_loc][0] = zero;
//                    y[j_loc][1] = zero;
//                    y[j_loc][2] = zero;
//                }
//            }
//
//            // need synchronization after loading data
//            threadgroup_barrier(mem_flags::mem_threadgroup);
//
//            // Each thread computes entries of Re_A_i, Im_A_i, Re_B_i, Im_B_i for its i and all js in the chunk.
//            for( uint j_loc = 0; j_loc < chunk_size; ++j_loc )
//            {
//                const uint j = chunk_size * chunk + j_loc;
//
//                const bool j_valid = j < n;
//
//                // We ignore the results if i and j coincide or if one of i or j are invalid.
//                const float delta = static_cast<float>( (!i_valid) || (!j_valid) || (i == j) );
//
//                const float z[3] = { y[j_loc][0] - x_i[0], y[j_loc][1] - x_i[1], y[j_loc][2] - x_i[2] };
//
//                const float r = sqrt( z[0] * z[0] + z[1] * z[1] + z[2] * z[2] ) + delta;
//
//                const float r_inv = one_over_four_pi * (one - delta) / r;
//
//                Re_A_i[j_loc] = cos( kappa * r ) * r_inv;
//                Im_A_i[j_loc] = sin( kappa * r ) * r_inv;
//
//            } // for( uint j_loc = 0; j_loc < chunk_size; ++j_loc )
//
//        }
//
//        threadgroup_barrier(mem_flags::mem_threadgroup);
//
//        // Now Re_A_i, Im_A_i are pre-computed.
//        // Space reserved for the ys is freed again.
//
//        // Now we do the actual matrix-matrix multiplication.
//        {
//            // The rows of the input matrix belonging to threadgroup.
//            threadgroup float Re_Y[chunk_size][n_waves];
//            threadgroup float Im_Y[chunk_size][n_waves];
//
//            // Each thread in threadgroup loads 1 row of Y.
//            {
//                const uint j_loc  = i_loc;
//                const uint j      = chunk_size * chunk + j_loc;
//
//                const bool j_valid = j < n;
//
//                if( j_valid )
//                {
//                    for( uint k = 0; k < n_waves; ++k )
//                    {
//                        Re_Y[j_loc][k] = Re_Y_global[n_waves*j+k];
//                        Im_Y[j_loc][k] = Im_Y_global[n_waves*j+k];
//                    }
//                }
//                else
//                {
//                    for( uint k = 0; k < n_waves; ++k )
//                    {
//                        Re_Y[j_loc][k] = zero;
//                        Im_Y[j_loc][k] = zero;
//                    }
//                }
//            }
//
//            threadgroup_barrier(mem_flags::mem_threadgroup);
//
//            for( uint j_loc = 0; j_loc < chunk_size; ++j_loc )
//            {
//                for( uint k = 0; k < n_waves; ++k )
//                {
//                    cfma(
//                        Re_A_i [j_loc], Im_A_i [j_loc],
//                        Re_Y[j_loc][k], Im_Y[j_loc][k],
//                        Re_X_i     [k], Im_X_i     [k]
//                    );
//                }
//            }
//        }
//
//        threadgroup_barrier(mem_flags::mem_threadgroup);
//
//
//    } // for( uint chunk = 0; chunk < chunk_count; ++chunk )
//
//    // Finally we can write X_i to X_global; each threads takes care of its own row.
//
//    if( i_valid )
//    {
//        for( uint k = 0; k < n_waves; ++k )
//        {
//            Re_X_global[n_waves*i+k] = Re_X_i[k];
//            Im_X_global[n_waves*i+k] = Im_X_i[k];
//        }
//    }
    
} // Helmholtz__Neumann_to_Dirichlet

// FIXME: Comment-out the following line for run-time compilation:
//)"

