R"(
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c Helmholtz.metal -o Helmholtz.air && xcrun -sdk macosx metallib Helmholtz.air -o Helmholtz.metallib

// FIXME: Comment-out the following line for run-time compilation:

// FIXME: Comment-in the following two lines for run-time compilation:
//constant constexpr unsigned char simd_size        = 32;
//constant constexpr unsigned char frequency_count  =  1;

#include <metal_stdlib>
#include <metal_simdgroup>

using namespace metal;


using Int   = uint;
using Float = float;
//using Float = half;

//using vec_T = array<Float,frequency_count>;

using vec_T = float2;

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
    const constant packed_float3 * const mid_points     [[buffer(0)]],
    const constant vec_T  * const g_Re_B         [[buffer(1)]],
    const constant vec_T  * const g_Im_B         [[buffer(2)]],
          device   vec_T  * const g_Re_C         [[buffer(3)]],
          device   vec_T  * const g_Im_C         [[buffer(4)]],
    const constant vec_T  &       kappa          [[buffer(5)]],
    const constant uint   &       n              [[buffer(6)]],
    const constant uint   &       dir_count      [[buffer(7)]],
                                                 
//    const uint  simdgroup_index_in_threadgroup   [[simdgroup_index_in_threadgroup]],
//    const uint  thread_index_in_simdgroup        [[thread_index_in_simdgroup]],
                                                 
    const uint  simd_group                       [[simdgroup_index_in_threadgroup]],
    const uint  simd_thread                      [[thread_index_in_simdgroup]],
    const uint  i_chunk                          [[threadgroup_position_in_grid]]
)
{
//    const unsigned char simd_group  = simdgroup_index_in_threadgroup;
//    const unsigned char simd_thread = thread_index_in_simdgroup;
    
    // Each SIMD group manages a row of a simd_size x simd_size matrix.
//    const Int i = simd_size * i_chunk + simd_group;
    
    // TODO: Make sure that division is without rest!
//    const Int j_chunk_count  = n / simd_size;
//    const Int k_chunk_count  = dir_count / simd_size;
//    const Int vecs_per_row   = simd_size * k_chunk_count;
//    const Int vecs_per_chunk = simd_size * vecs_per_row;
    
//    thread packed_float3 x_i = mid_points[simd_size * i_chunk + simd_group];
    
//    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // We will have dir_count \less n, hence k_chunks will be much lower than j_chunks.
    // Hence it might be a good idea to first loop over the k_chunks and pay the price for recomputing the respective blocks of A repeatedly.
    for( Int k_chunk = 0; k_chunk < (dir_count / simd_size); ++k_chunk )
    {
        // Threadgroup shall compute the matrix block
        // [simd_size * i_chunk,..., simd_size * (i_chunk+1) [
        // x
        // [simd_size * frequency_count k_chunk,...,simd_size * frequency_count * (k_chunk+1)[
        // of C.
        // Typically we have simd_size = 32. So for frequency_count == 4 this would be a 32 x 128 block of C.
        
        // Each thread manages one of the frequency_count-vectors of this block in C.
        thread vec_T Re_C_ik {zero};
        thread vec_T Im_C_ik {zero};
        
        for( Int j_chunk = 0; j_chunk < (n / simd_size); ++j_chunk )
        {
            // Each thread computes and holds a single entry of A.
            thread vec_T Re_A_ij;
            thread vec_T Im_A_ij;
            
//            const Int j = simd_size * j_chunk + simd_thread;
            
            // Each thread loads a y point.
            thread packed_float3 x_i = mid_points[simd_size * i_chunk + simd_group];
            thread packed_float3 y_j = mid_points[simd_size * j_chunk + simd_thread];
        
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            
            // Threadgroup loads the matrix block
            // [simd_size * j_chunk,..., simd_size * (j_chunk+1) [
            // x
            // [simd_size * k_chunk,...,simd_size * (k_chunk+1)[
            // of Re_B.
            // Typically simd_size = 32. Hence for frequency_count = 4 this is a 32 x 128 block.
            // In this case these are exactly 32 KB.
        
            threadgroup vec_T s_Re_B [simd_size][simd_size];
            threadgroup vec_T s_Im_B [simd_size][simd_size];
            
//            const Int pos = vecs_per_chunk * j_chunk
//                          + simd_size      * k_chunk
//                          + vecs_per_row   * simd_group
//                          +                  simd_thread;
            
            const Int pos = (simd_size * dir_count) * j_chunk
                          +  simd_size              * k_chunk
                          +  dir_count              * simd_group
                          +                           simd_thread;
            
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
                const float delta_ij = static_cast<float>( (i_chunk == j_chunk) && (simd_group ==  simd_thread) );

                const float r = distance( x_i, y_j );

                const float r_inv = one_over_four_pi * (one - delta_ij) / (r + delta_ij);

                float cos_kappa_r;
                float sin_kappa_r;
                
                if( frequency_count > 0 )
                {
                    sin_kappa_r = sincos( kappa[0] * r, cos_kappa_r );
                    
                    Re_A_ij[0] = cos_kappa_r * r_inv;
                    Im_A_ij[0] = sin_kappa_r * r_inv;
                }
                if( frequency_count > 1 )
                {
                    sin_kappa_r = sincos( kappa[1] * r, cos_kappa_r );
                    
                    Re_A_ij[1] = cos_kappa_r * r_inv;
                    Im_A_ij[1] = sin_kappa_r * r_inv;
                }
                if( frequency_count > 2 )
                {
                    sin_kappa_r = sincos( kappa[2] * r, cos_kappa_r );
                    
                    Re_A_ij[2] = cos_kappa_r * r_inv;
                    Im_A_ij[2] = sin_kappa_r * r_inv;
                }
                if( frequency_count > 3 )
                {
                    sin_kappa_r = sincos( kappa[3] * r, cos_kappa_r );
                    
                    Re_A_ij[3] = cos_kappa_r * r_inv;
                    Im_A_ij[3] = sin_kappa_r * r_inv;
                }
            }
            
            // Now the block of A is finished.
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            
            // Next we accumulate A * B.
            for( ushort j_loc = 0; j_loc < simd_size; ++j_loc )
            {
                // Each simd group handles a single row of A.
                // We use a broadcast distribute A_ij to avoid fetching from threadgroup memory.

                thread float Re_a_ij;
                thread float Im_a_ij;
                thread float Re_B_jk;
                thread float Im_B_jk;
                
                // Manual unrolling the for loop.
                if( frequency_count > 0 )
                {
                    Re_a_ij = simd_broadcast( Re_A_ij[0], j_loc );
                    Im_a_ij = simd_broadcast( Im_A_ij[0], j_loc );
                    Re_B_jk = s_Re_B[j_loc][simd_thread][0];
                    Im_B_jk = s_Im_B[j_loc][simd_thread][0];

                    Re_C_ik[0] += Re_a_ij * Re_B_jk - Im_a_ij * Im_B_jk;
                    Im_C_ik[0] += Re_a_ij * Im_B_jk + Im_a_ij * Re_B_jk;
                }
                if( frequency_count > 1 )
                {
                    Re_a_ij = simd_broadcast( Re_A_ij[1], j_loc );
                    Im_a_ij = simd_broadcast( Im_A_ij[1], j_loc );
                    Re_B_jk = s_Re_B[j_loc][simd_thread][1];
                    Im_B_jk = s_Im_B[j_loc][simd_thread][1];
                    
                    Re_C_ik[1] += Re_a_ij * Re_B_jk - Im_a_ij * Im_B_jk;
                    Im_C_ik[1] += Re_a_ij * Im_B_jk + Im_a_ij * Re_B_jk;
                }
                if( frequency_count > 2 )
                {
                    Re_a_ij = simd_broadcast( Re_A_ij[2], j_loc );
                    Im_a_ij = simd_broadcast( Im_A_ij[2], j_loc );
                    Re_B_jk = s_Re_B[j_loc][simd_thread][2];
                    Im_B_jk = s_Im_B[j_loc][simd_thread][2];
                    
                    Re_C_ik[2] += Re_a_ij * Re_B_jk - Im_a_ij * Im_B_jk;
                    Im_C_ik[2] += Re_a_ij * Im_B_jk + Im_a_ij * Re_B_jk;
                }
                if( frequency_count > 3 )
                {
                    Re_a_ij = simd_broadcast( Re_A_ij[3], j_loc );
                    Im_a_ij = simd_broadcast( Im_A_ij[3], j_loc );
                    Re_B_jk = s_Re_B[j_loc][simd_thread][3];
                    Im_B_jk = s_Im_B[j_loc][simd_thread][3];
                    
                    Re_C_ik[3] += Re_a_ij * Re_B_jk - Im_a_ij * Im_B_jk;
                    Im_C_ik[3] += Re_a_ij * Im_B_jk + Im_a_ij * Re_B_jk;
                }
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            
        } // for( Int j_chunk = 0; j_chunk < j_chunk_count; ++j_chunk )
        
//        const Int pos = vecs_per_chunk * i_chunk
//                      + simd_size      * k_chunk
//                      + vecs_per_row   * simd_group
//                      +                  simd_thread;
        
        const Int pos = (simd_size * dir_count) * i_chunk
                      +  simd_size              * k_chunk
                      +  dir_count              * simd_group
                      +                           simd_thread;
        
        
        // Each thread writes a vector of size vec_size to the output buffer.
        g_Re_C[pos] = Re_C_ik;
        g_Im_C[pos] = Im_C_ik;
        
    } // for( Int j_chunk = 0; j_chunk < j_chunk_count; ++j_chunk )
}

// FIXME: Comment-out the following line for run-time compilation:
)"
