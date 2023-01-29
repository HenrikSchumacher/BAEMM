R"(
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c Helmholtz.metal -o Helmholtz.air && xcrun -sdk macosx metallib Helmholtz.air -o Helmholtz.metallib

// FIXME: Comment-out the following line for run-time compilation:

// FIXME: Comment-in the following two lines for run-time compilation:
//constant constexpr uint simd_size  = 32;

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


template<typename vec_T>
[[kernel]] void Helmholtz__Neumann_to_Dirichlet2(
    const constant float3 * const mid_points     [[buffer(0)]],
    const constant vec_T  * const g_Re_B         [[buffer(1)]],
    const constant vec_T  * const g_Im_B         [[buffer(2)]],
          device   vec_T  * const g_Re_C         [[buffer(3)]],
          device   vec_T  * const g_Im_C         [[buffer(4)]],
    const constant float  &       kappa          [[buffer(5)]],
    const constant float  &       kappa_step     [[buffer(6)]],
    const constant uint   &       n              [[buffer(7)]],
    const constant uint   &       n_waves        [[buffer(8)]],
                                                 
    const uint  simdgroup_index_in_threadgroup   [[simdgroup_index_in_threadgroup]],
    const uint  thread_index_in_simdgroup        [[thread_index_in_simdgroup]],
                                              
    const uint  i_chunk                          [[threadgroup_position_in_grid]],
    const uint  rows                             [[simdgroups_per_threadgroup]],
    const uint  cols                             [[threads_per_simdgroup]]
)
{
    assert( rows == simd_size );
    assert( cols == simd_size );
    
    constexpr int vec_size     = sizeof(vec_T) >> 2;
    constexpr int block_size   = simd_size * simd_size;
    constexpr int k_chunk_size = simd_size * vec_size;
    
    
    const int simd_group  = simdgroup_index_in_threadgroup;
    const int simd_thread = thread_index_in_simdgroup;
    
    // Each SIMD group manages a row of a simd_size x simd_size matrix.
    const int i = simd_size * i_chunk + simd_group;
    
//    const int local_id = simd_size * simd_group + simd_thread;
    
    // Each threadgroup computes the matrix block
    // [32 * i_chunk,..., 32 * (i_chunk+1) [ x [0,...,32[ of C.
    

    
    
    // Shared memory to load a block of B.
    threadgroup vec_T s_Re_B [simd_size][simd_size];
    threadgroup vec_T s_Im_B [simd_size][simd_size];
    
    thread float Re_A_ij;
    thread float Im_A_ij;

    
    // TODO: Make sure that division is without rest!
    const int j_chunk_count  = n / simd_size;
    const int k_chunk_count  = n_waves / k_chunk_size;
    const int vecs_per_chunk = block_size * k_chunk_count;
    const int vecs_per_row   = simd_size  * k_chunk_count;
    
    thread float3 x_i;
    thread float3 y_j;
    
    x_i = mid_points[i];

    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // We will have n_waves \less n, hence k_chunks will be much lower than j_chunks.
    // Hence it might be a good idea to first loop over the k_chunks and pay the price for loading.
    for( int k_chunk = 0; k_chunk < k_chunk_count; ++k_chunk )
    {
        // Each thread manages one of the vec_T values of the resulting block of C.
        thread vec_T Re_C_ik (zero);
        thread vec_T Im_C_ik (zero);
        
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
            
            
            // Matrix block of A now computed.
            
            
            
            // Each thread loads an entry of a simd_size x simd_size block of vec_T in B.
            // E.g., if vec_T == float4 and simd_size == 32, then this is effectively a 32 x 128 block.
            
            const int pos = vecs_per_chunk * j_chunk + simd_size * k_chunk + ( vecs_per_row * simd_group + simd_thread);
            
            s_Re_B[simd_group][simd_thread] = g_Re_B[pos];
            s_Im_B[simd_group][simd_thread] = g_Im_B[pos];
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            for( ushort j_loc = 0; j_loc < simd_size; ++j_loc )
            {
                float Re_a_ij = simd_broadcast( Re_A_ij, j_loc );
                float Im_a_ij = simd_broadcast( Im_A_ij, j_loc );
                
                const vec_T Re_B_jk = s_Re_B[j_loc][simd_thread];
                const vec_T Im_B_jk = s_Im_B[j_loc][simd_thread];
                
                Re_C_ik += Re_a_ij * Re_B_jk - Im_a_ij * Im_B_jk;
                Im_C_ik += Re_a_ij * Im_B_jk + Im_a_ij * Re_B_jk;
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
        } // for( int j_chunk = 0; j_chunk < j_chunk_count; ++j_chunk )
        
        const int pos = vecs_per_chunk * i_chunk + simd_size * k_chunk + ( vecs_per_row * simd_group + simd_thread);
        g_Re_C[pos] = Re_C_ik;
        g_Im_C[pos] = Im_C_ik;
        
    } // for( int j_chunk = 0; j_chunk < j_chunk_count; ++j_chunk )
}

template [[ host_name("Helmholtz__Neumann_to_Dirichlet2_1") ]]
[[kernel]] void Helmholtz__Neumann_to_Dirichlet2<float>(
    const constant float3 * const mid_points     [[buffer(0)]],
    const constant float  * const g_Re_B         [[buffer(1)]],
    const constant float  * const g_Im_B         [[buffer(2)]],
          device   float  * const g_Re_C         [[buffer(3)]],
          device   float  * const g_Im_C         [[buffer(4)]],
    const constant float  &       kappa          [[buffer(5)]],
    const constant float  &       kappa_step     [[buffer(6)]],
    const constant uint   &       n              [[buffer(7)]],
    const constant uint   &       n_waves        [[buffer(8)]],
                                                 
    const uint  simdgroup_index_in_threadgroup   [[simdgroup_index_in_threadgroup]],
    const uint  thread_index_in_simdgroup        [[thread_index_in_simdgroup]],
                                              
    const uint  i_chunk                          [[threadgroup_position_in_grid]],
    const uint  rows                             [[simdgroups_per_threadgroup]],
    const uint  cols                             [[threads_per_simdgroup]]
);

template [[ host_name("Helmholtz__Neumann_to_Dirichlet2_2") ]]
[[kernel]] void Helmholtz__Neumann_to_Dirichlet2<float2>(
    const constant float3 * const mid_points     [[buffer(0)]],
    const constant float2 * const g_Re_B         [[buffer(1)]],
    const constant float2 * const g_Im_B         [[buffer(2)]],
          device   float2 * const g_Re_C         [[buffer(3)]],
          device   float2 * const g_Im_C         [[buffer(4)]],
    const constant float  &       kappa          [[buffer(5)]],
    const constant float  &       kappa_step     [[buffer(6)]],
    const constant uint   &       n              [[buffer(7)]],
    const constant uint   &       n_waves        [[buffer(8)]],
                                                 
    const uint  simdgroup_index_in_threadgroup   [[simdgroup_index_in_threadgroup]],
    const uint  thread_index_in_simdgroup        [[thread_index_in_simdgroup]],
                                              
    const uint  i_chunk                          [[threadgroup_position_in_grid]],
    const uint  rows                             [[simdgroups_per_threadgroup]],
    const uint  cols                             [[threads_per_simdgroup]]
);

template [[ host_name("Helmholtz__Neumann_to_Dirichlet2_4") ]]
[[kernel]] void Helmholtz__Neumann_to_Dirichlet2<float4>(
    const constant float3 * const mid_points     [[buffer(0)]],
    const constant float4 * const g_Re_B         [[buffer(1)]],
    const constant float4 * const g_Im_B         [[buffer(2)]],
          device   float4 * const g_Re_C         [[buffer(3)]],
          device   float4 * const g_Im_C         [[buffer(4)]],
    const constant float  &       kappa          [[buffer(5)]],
    const constant float  &       kappa_step     [[buffer(6)]],
    const constant uint   &       n              [[buffer(7)]],
    const constant uint   &       n_waves        [[buffer(8)]],
                                                         
    const uint  simdgroup_index_in_threadgroup   [[simdgroup_index_in_threadgroup]],
    const uint  thread_index_in_simdgroup        [[thread_index_in_simdgroup]],
                                              
    const uint  i_chunk                          [[threadgroup_position_in_grid]],
    const uint  rows                             [[simdgroups_per_threadgroup]],
    const uint  cols                             [[threads_per_simdgroup]]
);

// FIXME: Comment-out the following line for run-time compilation:
)"
