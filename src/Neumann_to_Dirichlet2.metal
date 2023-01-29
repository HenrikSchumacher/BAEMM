R"(
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c Helmholtz.metal -o Helmholtz.air && xcrun -sdk macosx metallib Helmholtz.air -o Helmholtz.metallib

// FIXME: Comment-out the following line for run-time compilation:

// FIXME: Comment-in the following two lines for run-time compilation:
//constant constexpr unsigned char simd_size = 32;
//constant constexpr unsigned char vec_size  =  1;

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;


using Int   = uint;
using Float = float;
//using Float = half;

using vec_T = array<Float,vec_size>;

constant constexpr Float zero     = static_cast<Float>(0);
constant constexpr Float one      = static_cast<Float>(1);
constant constexpr Float two      = static_cast<Float>(2);
//constant constexpr Float one_half = one / two;

constant constexpr Float pi       = 3.141592653589793;
constant constexpr Float two_pi   = two * pi;
constant constexpr Float four_pi  = two * two_pi;

//constant constexpr Float one_over_two_pi  = one / two_pi;
constant constexpr Float one_over_four_pi = one / four_pi;


[[kernel]] void Helmholtz__Neumann_to_Dirichlet2(
    const constant float3 * const mid_points     [[buffer(0)]],
    const constant vec_T  * const g_Re_B         [[buffer(1)]],
    const constant vec_T  * const g_Im_B         [[buffer(2)]],
          device   vec_T  * const g_Re_C         [[buffer(3)]],
          device   vec_T  * const g_Im_C         [[buffer(4)]],
    const constant Float  &       kappa          [[buffer(5)]],
    const constant uint   &       n              [[buffer(6)]],
    const constant uint   &       n_waves        [[buffer(7)]],
                                                 
    const uint  simdgroup_index_in_threadgroup   [[simdgroup_index_in_threadgroup]],
    const uint  thread_index_in_simdgroup        [[thread_index_in_simdgroup]],
    const uint  i_chunk                          [[threadgroup_position_in_grid]]
)
{
    const unsigned char simd_group  = simdgroup_index_in_threadgroup;
    const unsigned char simd_thread = thread_index_in_simdgroup;
    
    // Each SIMD group manages a row of a simd_size x simd_size matrix.
    const Int i = simd_size * i_chunk + simd_group;
    
    // TODO: Make sure that division is without rest!
    const Int j_chunk_count  = n / simd_size;
    const Int k_chunk_count  = n_waves / (simd_size * vec_size);
    const Int vecs_per_row   = simd_size * k_chunk_count;
    const Int vecs_per_chunk = simd_size * vecs_per_row;
    
    
    thread float3 x_i = mid_points[i];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // We will have n_waves \less n, hence k_chunks will be much lower than j_chunks.
    // Hence it might be a good idea to first loop over the k_chunks and pay the price for recomputing the respective blocks of A repeatedly.
    for( Int k_chunk = 0; k_chunk < k_chunk_count; ++k_chunk )
    {
        // Threadgroup shall compute the matrix block
        // [simd_size * i_chunk,..., simd_size * (i_chunk+1) [
        // x
        // [simd_size * vec_size k_chunk,...,simd_size * vec_size * (k_chunk+1)[
        // of C.
        // Typically we have simd_size = 32. So for vec_size == 4 this would be a 32 x 128 block of C.
        
        // Each thread manages one of the vec_size-vectors of this block in C.
        thread vec_T Re_C_ik {zero};
        thread vec_T Im_C_ik {zero};
        
        for( Int j_chunk = 0; j_chunk < j_chunk_count; ++j_chunk )
        {
            // Each thread computes and holds a single entry of A.
            thread Float Re_A_ij;
            thread Float Im_A_ij;
            
            const Int j = simd_size * j_chunk + simd_thread;
            
            // Each thread loads a y point.
            thread float3 y_j = mid_points[j];
        
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            
            // Threadgroup loads the matrix block
            // [simd_size * j_chunk,..., simd_size * (j_chunk+1) [
            // x
            // [simd_size * k_chunk,...,simd_size * (k_chunk+1)[
            // of Re_B.
            // Typically simd_size = 32. Hence for vec_size = 4 this is a 32 x 128 block.
            // In this case these are exactly 32 KB.
        
            threadgroup vec_T s_Re_B [simd_size][simd_size];
            threadgroup vec_T s_Im_B [simd_size][simd_size];
            
            const Int pos = vecs_per_chunk * j_chunk
                          + simd_size      * k_chunk
                          + vecs_per_row   * simd_group
                          +                  simd_thread;
            
            // Load the real part of B and accumulate its contribution into C.
            s_Re_B[simd_group][simd_thread] = g_Re_B[pos];
            s_Im_B[simd_group][simd_thread] = g_Im_B[pos];
    

            // Threadgroup computes the matrix block
            // [simd_size * i_chunk,..., simd_size * (i_chunk+1) [
            // x
            // [simd_size * j_chunk,...,simd_size * (j_chunk+1)[ of A.
            // Typically we have simd_size = 32, and this is a 32 x 32 block.
            
            // Block to prevent some register spilling.
            {
                //        // We ignore the results if i and j coincide or if one of i or j are invalid.
                const float delta_ij = static_cast<float>( i == j );

                const float r = distance( x_i, y_j );

                const float r_inv = one_over_four_pi * (one - delta_ij) / (r + delta_ij);

                float cos_kappa_r;
                float sin_kappa_r = sincos( kappa * r, cos_kappa_r );

                Re_A_ij = cos_kappa_r * r_inv;
                Im_A_ij = sin_kappa_r * r_inv;
            }
            
            // Now the block of A is finished.
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
//            // Next we accumulate A * B.
//            for( ushort j_loc = 0; j_loc < simd_size; ++j_loc )
//            {
//                // Each simd group handles a single row of A.
//                // We use a broadcast distribute A_ij to avoid fetching from threadgroup memory.
//                const Float Re_a_ij = simd_broadcast( Re_A_ij, j_loc );
//                const Float Im_a_ij = simd_broadcast( Im_A_ij, j_loc );
//
//                const vec_T Re_B_jk = s_B[j_loc][simd_thread];
//
//                // Manual unrolling of for loop.
//                if( vec_size > 0 )
//                {
//                    Re_C_ik[0] = fma( Re_a_ij, Re_B_jk[0], Re_C_ik[0] );
//                    Im_C_ik[0] = fma( Im_a_ij, Re_B_jk[0], Im_C_ik[0] );
//                }
//                if( vec_size > 1 )
//                {
//                    Re_C_ik[1] = fma( Re_a_ij, Re_B_jk[1], Re_C_ik[1] );
//                    Im_C_ik[1] = fma( Im_a_ij, Re_B_jk[1], Im_C_ik[1] );
//                }
//                if( vec_size > 2 )
//                {
//                    Re_C_ik[2] = fma( Re_a_ij, Re_B_jk[2], Re_C_ik[2] );
//                    Im_C_ik[2] = fma( Im_a_ij, Re_B_jk[2], Im_C_ik[2] );
//                }
//                if( vec_size > 3 )
//                {
//                    Re_C_ik[3] = fma( Re_a_ij, Re_B_jk[3], Re_C_ik[3] );
//                    Im_C_ik[3] = fma( Im_a_ij, Re_B_jk[3], Im_C_ik[3] );
//                }
//                if( vec_size > 4 )
//                {
//                    Re_C_ik[4] = fma( Re_a_ij, Re_B_jk[4], Re_C_ik[4] );
//                    Im_C_ik[4] = fma( Im_a_ij, Re_B_jk[4], Im_C_ik[4] );
//                }
//                if( vec_size > 5 )
//                {
//                    Re_C_ik[5] = fma( Re_a_ij, Re_B_jk[5], Re_C_ik[5] );
//                    Im_C_ik[5] = fma( Im_a_ij, Re_B_jk[5], Im_C_ik[5] );
//                }
//                if( vec_size > 6 )
//                {
//                    Re_C_ik[6] = fma( Re_a_ij, Re_B_jk[6], Re_C_ik[6] );
//                    Im_C_ik[6] = fma( Im_a_ij, Re_B_jk[6], Im_C_ik[6] );
//                }
//                if( vec_size > 7 )
//                {
//                    Re_C_ik[7] = fma( Re_a_ij, Re_B_jk[7], Re_C_ik[7] );
//                    Im_C_ik[7] = fma( Im_a_ij, Re_B_jk[7], Im_C_ik[7] );
//                }
//            }
//
//            threadgroup_barrier(mem_flags::mem_threadgroup);
//
//            // Load the imaginary part of B and accumulate its contribution into C.
//            s_B[simd_group][simd_thread] = g_Im_B[pos];
//
//            threadgroup_barrier(mem_flags::mem_threadgroup);
//
//            for( ushort j_loc = 0; j_loc < simd_size; ++j_loc )
//            {
//                // Each simd group handles a single row of A.
//                // We use a broadcast distribute A_ij to avoid fetching from threadgroup memory.
//                const Float Re_a_ij = simd_broadcast( Re_A_ij, j_loc );
//                const Float Im_a_ij = simd_broadcast( Im_A_ij, j_loc );
//
//                const vec_T Im_B_jk = s_B[j_loc][simd_thread];
//
//                // Manual unrolling of for loop.
//                if( vec_size > 0 )
//                {
//                    Re_C_ik[0] += - Im_a_ij * Im_B_jk[0];
//                    Im_C_ik[0] +=   Re_a_ij * Im_B_jk[0];
//                }
//                if( vec_size > 1 )
//                {
//                    Re_C_ik[1] += - Im_a_ij * Im_B_jk[1];
//                    Im_C_ik[1] +=   Re_a_ij * Im_B_jk[1];
//                }
//                if( vec_size > 2 )
//                {
//                    Re_C_ik[2] += - Im_a_ij * Im_B_jk[2];
//                    Im_C_ik[2] +=   Re_a_ij * Im_B_jk[2];
//                }
//                if( vec_size > 3 )
//                {
//                    Re_C_ik[3] += - Im_a_ij * Im_B_jk[3];
//                    Im_C_ik[3] +=   Re_a_ij * Im_B_jk[3];
//                }
//                if( vec_size > 4 )
//                {
//                    Re_C_ik[4] += - Im_a_ij * Im_B_jk[4];
//                    Im_C_ik[4] +=   Re_a_ij * Im_B_jk[4];
//                }
//                if( vec_size > 5 )
//                {
//                    Re_C_ik[5] += - Im_a_ij * Im_B_jk[5];
//                    Im_C_ik[5] +=   Re_a_ij * Im_B_jk[5];
//                }
//                if( vec_size > 6 )
//                {
//                    Re_C_ik[6] += - Im_a_ij * Im_B_jk[6];
//                    Im_C_ik[6] +=   Re_a_ij * Im_B_jk[6];
//                }
//                if( vec_size > 7 )
//                {
//                    Re_C_ik[7] += - Im_a_ij * Im_B_jk[7];
//                    Im_C_ik[7] +=   Re_a_ij * Im_B_jk[7];
//                }
//            }
            
            
            // Next we accumulate A * B.
            for( ushort j_loc = 0; j_loc < simd_size; ++j_loc )
            {
                // Each simd group handles a single row of A.
                // We use a broadcast distribute A_ij to avoid fetching from threadgroup memory.
                const Float Re_a_ij = simd_broadcast( Re_A_ij, j_loc );
                const Float Im_a_ij = simd_broadcast( Im_A_ij, j_loc );

                threadgroup const vec_T & Re_B_jk = s_Re_B[j_loc][simd_thread];
                threadgroup const vec_T & Im_B_jk = s_Im_B[j_loc][simd_thread];

                    // Manual unrolling the for loop.
                    if( vec_size > 0 )
                    {
                        Re_C_ik[0] += Re_a_ij * Re_B_jk[0] - Im_a_ij * Im_B_jk[0];
                        Im_C_ik[0] += Re_a_ij * Im_B_jk[0] + Im_a_ij * Re_B_jk[0];
                    }
                    if( vec_size > 1 )
                    {
                        Re_C_ik[1] += Re_a_ij * Re_B_jk[1] - Im_a_ij * Im_B_jk[1];
                        Im_C_ik[1] += Re_a_ij * Im_B_jk[1] + Im_a_ij * Re_B_jk[1];
                    }
                    if( vec_size > 2 )
                    {
                        Re_C_ik[2] += Re_a_ij * Re_B_jk[2] - Im_a_ij * Im_B_jk[2];
                        Im_C_ik[2] += Re_a_ij * Im_B_jk[2] + Im_a_ij * Re_B_jk[2];
                    }
                    if( vec_size > 3 )
                    {
                        Re_C_ik[3] += Re_a_ij * Re_B_jk[3] - Im_a_ij * Im_B_jk[3];
                        Im_C_ik[3] += Re_a_ij * Im_B_jk[3] + Im_a_ij * Re_B_jk[3];
                    }
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            
        } // for( Int j_chunk = 0; j_chunk < j_chunk_count; ++j_chunk )
        
        const Int pos = vecs_per_chunk * i_chunk
                      + simd_size      * k_chunk
                      + vecs_per_row   * simd_group
                      +                  simd_thread;
        
        // Each thread writes a vector of size vec_size to the output buffer.
        g_Re_C[pos] = Re_C_ik;
        g_Im_C[pos] = Im_C_ik;
        
    } // for( Int j_chunk = 0; j_chunk < j_chunk_count; ++j_chunk )
}

// FIXME: Comment-out the following line for run-time compilation:
)"
