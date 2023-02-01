R"(
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c Helmholtz.metal -o Helmholtz.air && xcrun -sdk macosx metallib Helmholtz.air -o Helmholtz.metallib

// FIXME: Comment-out the following line for run-time compilation:

// FIXME: Comment-in the following two lines for run-time compilation:
//constant constexpr uint simd_size         = 32;

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


[[kernel]] void Helmholtz__Neumann_to_Dirichlet3(
    const constant float3 * const mid_points     [[buffer(0)]],
    const constant float2 * const g_B            [[buffer(1)]],
          device   float2 * const g_C            [[buffer(2)]],
    const constant float  &       kappa          [[buffer(3)]],
    const constant uint   &       n              [[buffer(4)]],
                                                 
    const uint  simdgroup_index_in_threadgroup   [[simdgroup_index_in_threadgroup]],
    const uint  thread_index_in_simdgroup        [[thread_index_in_simdgroup]],
                                                 
    const uint  i_chunk                          [[threadgroup_position_in_grid]],
    const uint  rows                             [[simdgroups_per_threadgroup]],
    const uint  cols                             [[threads_per_simdgroup]]
)
{
    assert( rows == simd_size );
    assert( cols == simd_size );
    
    
    // Each SIMD group manages a row of a 32 x 32 matrix.
    const int simd_group  = simdgroup_index_in_threadgroup;
    const int simd_thread = thread_index_in_simdgroup;
    
    
    const int i = simd_size * i_chunk + simd_group;
    
    const int local_id = simd_size * simd_group + simd_thread;
    
    // Each threadgroup computes the matrix block
    // [32 * i_chunk,..., 32 * (i_chunk+1) [ x [0,...,32[ of C.
    
    // Each thread manages one of the values of the resulting block of C.
    thread float2 C_ik = zero;
    
    int tile_size = simd_size * simd_size;
    
    // Shared memory to load a block of B.
    threadgroup float2 s_B [simd_size][simd_size];

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
        
        thread float2 A_ij = {cos_kappa_r * r_inv,sin_kappa_r * r_inv};

        // Each thread loads an entry of a 32 x 32 block of B.
        s_B[simd_group][simd_thread] = g_B[tile_size * j_chunk + local_id];
        
        threadgroup_barrier(mem_flags::mem_threadgroup);

        
        for( ushort j_loc_ = 0; j_loc_ < simd_size; ++j_loc_ )
        {
            ushort j_loc = (j_loc_ + simd_group) & (simd_size-1);
            
            float2 a = simd_broadcast( A_ij, j_loc );
            
            const float2 B_jk = s_B[j_loc][simd_thread];
            
            C_ik[0] += a[0] * B_jk[0] - a[1] * B_jk[1];
            C_ik[1] += a[0] * B_jk[1] + a[1] * B_jk[0];
            
//            simdgroup_barrier(mem_flags::mem_none);
        }
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    g_C[tile_size * i_chunk + local_id] = C_ik;
}

// FIXME: Comment-out the following line for run-time compilation:
)"
