R"(
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c Helmholtz.metal -o Helmholtz.air && xcrun -sdk macosx metallib Helmholtz.air -o Helmholtz.metallib

// FIXME: Comment-out the following line for run-time compilation:

// FIXME: Comment-in the following two lines for run-time compilation:
//constant constexpr uint simd_size  = 32;
//constant constexpr uint vec_size   = 1;

#include <metal_stdlib>
#include <metal_simdgroup>

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

using vec_T = array<float,vec_size>;

[[kernel]] void Helmholtz__Neumann_to_Dirichlet2(
    const constant float3 * const mid_points     [[buffer(0)]],
    const constant vec_T  * const g_Re_B         [[buffer(1)]],
    const constant vec_T  * const g_Im_B         [[buffer(2)]],
          device   vec_T  * const g_Re_C         [[buffer(3)]],
          device   vec_T  * const g_Im_C         [[buffer(4)]],
    const constant float  &       kappa          [[buffer(5)]],
    const constant uint   &       n              [[buffer(6)]],
    const constant uint   &       n_waves        [[buffer(7)]],
                                                 
    const uint  simdgroup_index_in_threadgroup   [[simdgroup_index_in_threadgroup]],
    const uint  thread_index_in_simdgroup        [[thread_index_in_simdgroup]],
                                              
    const uint  i_chunk                          [[threadgroup_position_in_grid]],
    const uint  rows                             [[simdgroups_per_threadgroup]],
    const uint  cols                             [[threads_per_simdgroup]]
)
{
    assert( rows == simd_size );
    assert( cols == simd_size );
    
    const int simd_group  = simdgroup_index_in_threadgroup;
    const int simd_thread = thread_index_in_simdgroup;
    
    // Each SIMD group manages a row of a simd_size x simd_size matrix.
    const int i = simd_size * i_chunk + simd_group;
    
    const int local_id = simd_size * simd_group + simd_thread;
    
    // Each threadgroup computes the matrix block
    // [32 * i_chunk,..., 32 * (i_chunk+1) [ x [0,...,32[ of C.
    
    // Each thread manages one of the values of the resulting block of C.
    thread vec_T Re_C_ik {zero};
    thread vec_T Im_C_ik {zero};
    
    int tile_size = simd_size * simd_size;
    
    // Shared memory to load a block of B.
    threadgroup vec_T s_Re_B [simd_size][simd_size];
    threadgroup vec_T s_Im_B [simd_size][simd_size];
    
    thread float Re_A_ij;
    thread float Im_A_ij;

    const int j_chunk_count = n / simd_size;
    
    thread float3 x_i;
    thread float3 y_j;
    
    x_i = mid_points[i];

    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for( int j_chunk = 0; j_chunk < j_chunk_count; ++j_chunk )
    {
        const int j = simd_size * j_chunk + simd_thread;
        
        y_j = mid_points[j];
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
//        // We ignore the results if i and j coincide or if one of i or j are invalid.
        const float delta_ij = static_cast<float>( i == j );

        const float r = distance( x_i, y_j );

        const float r_inv = one_over_four_pi * (one - delta_ij) / (r + delta_ij);

        // Every thread holds exactly one matrix entry.        
        float cos_kappa_r;
        float sin_kappa_r = sincos( kappa * r, cos_kappa_r );
        
        // Each thread stores one entry of A.
        Re_A_ij = cos_kappa_r * r_inv;
        Im_A_ij = sin_kappa_r * r_inv;

        // Each thread loads an entry of a 32 x 32 block of B.
        
        const int pos = tile_size * j_chunk + local_id;
        
        s_Re_B[simd_group][simd_thread] = g_Re_B[pos];
        s_Im_B[simd_group][simd_thread] = g_Im_B[pos];
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for( ushort j_loc = 0; j_loc < simd_size; ++j_loc )
        {
            float Re_a_ij = simd_broadcast( Re_A_ij, j_loc );
            float Im_a_ij = simd_broadcast( Im_A_ij, j_loc );

            const vec_T Re_B_jk = s_Re_B[j_loc][simd_thread];
            const vec_T Im_B_jk = s_Im_B[j_loc][simd_thread];
            
            
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
    }
    
    const int pos = tile_size * i_chunk + local_id;
    g_Re_C[pos] = Re_C_ik;
    g_Im_C[pos] = Im_C_ik;
}

// FIXME: Comment-out the following line for run-time compilation:
)"
