R"(
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c Helmholtz.metal -o Helmholtz.air && xcrun -sdk macosx metallib Helmholtz.air -o Helmholtz.metallib

// FIXME: Comment-out the following line for run-time compilation:

// FIXME: We use "block_size" and "wave_chunk_size" as "template parameters" for jit-compilation.
// FIXME: Comment-in the following two lines for run-time compilation:
//constant constexpr int  block_size      = 64;
//constant constexpr int  wave_chunk_size = 32;
//constant constexpr bool single_layer    = true;
//constant constexpr bool double_layer    = true;
//constant constexpr bool adjdbl_layer    = true;

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
    const constant float3 * const mid_points         [[buffer(0)]],
    const constant float3 * const normals            [[buffer(1)]],
    const constant float  * const Re_B_global        [[buffer(2)]],
    const constant float  * const Im_B_global        [[buffer(3)]],
          device   float  * const Re_C_global        [[buffer(4)]],
          device   float  * const Im_C_global        [[buffer(5)]],
    const constant float  &       kappa              [[buffer(6)]],
    const constant float3 &       coeff              [[buffer(7)]],
    const constant int    &       n                  [[buffer(8)]],
    const constant int    &       wave_count         [[buffer(9)]],
                                   
    const uint thread_position_in_threadgroup        [[thread_position_in_threadgroup]],
    const uint thread_position_in_grid               [[thread_position_in_grid]],
    const uint threads_per_threadgroup               [[threads_per_threadgroup]],
    const uint threadgroup_position_in_grid          [[threadgroup_position_in_grid]]
)
{
    
    // Important assumptions:
    // - wave_count and wave_chunk_size are divisible by vec_size
    // - n is divisible by block_size
    // - threads_per_threadgroup == block_size
    
    constexpr int k_chunk_size  = wave_chunk_size;
    const     int k_chunk_count = wave_count / k_chunk_size;
    const     int k_ld          = wave_count;  // Leading dim of B and C.
    
    // number of blocks
    const int block_count = (n + block_size - 1) / block_size;
    
    
    const int i_loc   = thread_position_in_threadgroup;
    const int i       = thread_position_in_grid;
//    const int i_block = threadgroup_position_in_grid;
    
    // each thread in the threadgroup gets one target point x and normal nu assigned.
    thread float3 x_i;
    thread float3 nu_i;
    
    thread float Re_A_i [block_size]; // stores real part of row A_i for j in threadgroup
    thread float Im_A_i [block_size]; // stores imag part of row A_i for j in threadgroup

    
    // The rows of the input matrix belonging to the threadgroup.
    threadgroup float Re_B [block_size][k_chunk_size];
    threadgroup float Im_B [block_size][k_chunk_size];
    
    // Each thread loads the x-data for itself only once.
    x_i  = mid_points[i];
    
    if( adjdbl_layer )
    {
        nu_i = normals[i];
    }
    
    
    // Typically, k_chunk_count is much smaller than block_count.
    // Since loading and writing to device memory is very expensive, it might be a better to make
    // the loop over the k_chunks the outer loop and pay the (small) price of recomputing A multiple times.
    for( int k_chunk = 0; k_chunk < k_chunk_count; ++k_chunk )
    {
        // Each thread maintains one row of the output matrix.
        thread float Re_C_i [k_chunk_size] = {zero};
        thread float Im_C_i [k_chunk_size] = {zero};
        
        for( int j_block = 0; j_block < block_count; ++j_block )
        {
            // Compute the block of A corresponding to
            //  [ block_size * i_block,...,block_size * (i_block+1)[
            //  x
            //  [ block_size * j_block,...,block_size * (j_block+1)[
            {
                // Each thread lets its x interact with a bunch of ys loaded into threadgroup.
                threadgroup float3 y  [block_size];
                threadgroup float3 mu [block_size];
                
                // Each thread in the threadgroup loads 1 entry of y and mu.
                if( double_layer )
                {
                    const int j_loc  = i_loc;
                    const int j      = block_size * j_block + j_loc;
                    y [j_loc] = mid_points[j];
                    mu[j_loc] = normals   [j];
                }
                else
                {
                    const int j_loc  = i_loc;
                    const int j      = block_size * j_block + j_loc;
                    y [j_loc] = mid_points[j];
                }
                
                // need synchronization after loading data
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                for( int j_loc = 0; j_loc < block_size; ++j_loc )
                {
                    const int j = block_size * j_block + j_loc;
                    
                    // We ignore the results if i and j coincide or if one of i or j are invalid.
                    const float delta = static_cast<float>( (i == j) );
                    
                    const float r = distance( x_i, y[j_loc] );
                    
                    const float r_inv = one_over_four_pi * (one - delta) / (r + delta);
                    
                    float cos_kappa_r;
                    float sin_kappa_r = sincos( kappa * r, cos_kappa_r );
                    
                    
                    Re_A_i[j_loc] = 0.f;
                    Im_A_i[j_loc] = 0.f;
                    
                    if( single_layer )
                    {
                        Re_A_i[j_loc] = coeff[0] * cos_kappa_r * r_inv;
                        Im_A_i[j_loc] = coeff[0] * sin_kappa_r * r_inv;
                    }
                    
                } // for( int j_loc = 0; j_loc < block_size; ++j_loc )
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Now Re_A_i, Im_A_i are pre-computed.
            // Space reserved for the ys is freed again.
            
            
            // Load the block of B corresponding to
            //  [ block_size * j_block,...,block_size * (j_block+1)[
            //  x
            //  [ k_chunk_size * k_chunk,...,k_chunk_size * (k_chunk+1)[
                
            // Each thread in threadgroup loads 1 row of B.
            {
                const int j_loc  = i_loc;
                const int j      = block_size * j_block + j_loc;
                
                for( int k = 0; k < k_chunk_size; ++k )
                {
                    Re_B[j_loc][k] = Re_B_global[k_ld * j + k_chunk_size * k_chunk + k];
                    Im_B[j_loc][k] = Im_B_global[k_ld * j + k_chunk_size * k_chunk + k];
                }
            }
            
            //            // Coalesced(?) load.
            //            constant    const float * Re_from = &Re_B_global[k_ld * block_size * j_block];
            //            constant    const float * Im_from = &Im_B_global[k_ld * block_size * j_block];
            //
            //            threadgroup       float * Re_to   = &Re_B[0][0];
            //            threadgroup       float * Im_to   = &Im_B[0][0];
            //
            //            for( int k = 0; k < block_size * k_chunk_count; k += block_size )
            //            {
            //                Re_to[k + i_loc] = Re_from[k + i_loc];
            //                Im_to[k + i_loc] = Im_from[k + i_loc];
            //            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for( int j_loc = 0; j_loc < block_size; ++j_loc )
            {
                for( int k = 0; k < k_chunk_size; ++k )
                {
                    Re_C_i[k] += Re_A_i[j_loc] * Re_B[j_loc][k] - Im_A_i[j_loc] * Im_B[j_loc][k];
                    Im_C_i[k] += Re_A_i[j_loc] * Im_B[j_loc][k] + Im_A_i[j_loc] * Re_B[j_loc][k];
                }
            }
            
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            
        } // for( int j_block = 0; j_block < block_count; ++j_block )
        
        //    // In order to achieve coalesced write, we use threadgroup memory Re_B, Im_B as intermediate buffer.
        //    // Each thread writes it row into shared memory.
        //    for( int k = 0; k < k_chunk_count; ++k )
        //    {
        //        Re_B[i_loc][k] = Re_C_i[k];
        //        Im_B[i_loc][k] = Im_C_i[k];
        //    }
        //
        //    threadgroup const float * Re_from = &Re_B[0][0];
        //    threadgroup const float * Im_from = &Im_B[0][0];
        //
        //    device            float * Re_to   = &Re_C_global[k_ld * block_size * threadgroup_position_in_grid];
        //    device            float * Im_to   = &Im_C_global[k_ld * block_size * threadgroup_position_in_grid];
        //
        //    // Coalesced(?) write to output.
        //    for( int k = 0; k < block_size * k_chunk_count; k += block_size )
        //    {
        //        Re_to[k + i_loc] = Re_from[k + i_loc];
        //        Im_to[k + i_loc] = Im_from[k + i_loc];
        //    }
        
        // Write the block of C corresponding to
        //  [ block_size * i_block,...,block_size * (i_block+1)[
        //  x
        //  [ k_chunk_size * k_chunk,...,k_chunk_size * (k_chunk+1)[
        //
        // Each threads takes care of its own row.
        for( int k = 0; k < k_chunk_size; ++k )
        {
            Re_C_global[k_ld * i + k_chunk_size * k_chunk + k] = Re_C_i[k];
            Im_C_global[k_ld * i + k_chunk_size * k_chunk + k] = Im_C_i[k];
        }
    }
    
} // Helmholtz__Neumann_to_Dirichlet

// FIXME: Comment-out the following line for run-time compilation:
)"
