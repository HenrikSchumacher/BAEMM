R"(
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c Helmholtz.metal -o Helmholtz.air && xcrun -sdk macosx metallib Helmholtz.air -o Helmholtz.metallib

// FIXME: Comment-out the following line for run-time compilation:

// FIXME: Comment-in the following two lines for run-time compilation:
//constant constexpr uint simd_count  = 32;
//constant constexpr uint simd_size   = 32;

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


[[kernel]] void Helmholtz__Neumann_to_Dirichlet(
    const constant float3 * const mid_points     [[buffer(0)]],
    const constant float  * const g_Re_B         [[buffer(1)]],
    const constant float  * const g_Im_B         [[buffer(2)]],
          device   float  * const g_Re_C         [[buffer(3)]],
          device   float  * const g_Im_C         [[buffer(4)]],
    const constant float  &       kappa          [[buffer(5)]],
    const constant float  &       kappa_step     [[buffer(6)]],
    const constant uint   &       n              [[buffer(7)]],

    const uint  i_loc                            [[simdgroup_index_in_threadgroup]],
    const uint  j_loc                            [[thread_index_in_simdgroup]],
                                                 
    const uint  i_chunk                          [[threadgroup_position_in_grid]],
    const uint  rows                             [[simdgroups_per_threadgroup]],
    const uint  cols                             [[threads_per_simdgroup]]
)
{
//    constexpr uint simd_len = 32;
    
    assert( rows == simd_count );
    assert( cols == simd_size );
    
    
    // Each SIMD group manages a row of a 32 x 32 matrix.
    const uint i = simd_count * i_chunk + i_loc;
    
    // Each threadgroup computes the matrix block
    // [32 * i_chunk,..., 32 * (i_chunk+1) [ x [0,...,32[ of C.
    
    // Each thread stores one of the values of the resulting block.
    thread float Re_C_ik = zero;
    thread float Im_C_ik = zero;
    
    threadgroup float s_Re_BT [simd_count][simd_size] = {{}};
    threadgroup float s_Im_BT [simd_count][simd_size] = {{}};
    
    const uint j_chunk_count = n / simd_size;
    
    thread float3 x_i;
    thread float3 y_j;
    
    // First thread in each simdgroup loads the common x-vector and broadcasts.
    if( j_loc == 0 )
    {
        x_i = mid_points[i];
    }
    
    x_i = simd_broadcast( x_i, 0 );
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    for( uint j_chunk = 0; j_chunk < j_chunk_count; ++j_chunk )
    {
        const uint j = simd_size * j_chunk + j_loc;
        
        y_j = mid_points[j];
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
//        // We ignore the results if i and j coincide or if one of i or j are invalid.
        const float delta_ij = static_cast<float>( i == j );

        const float r = distance( x_i, y_j );

//        const float r = sqrt(
//              (x_i[0] - y_j[0]) * (x_i[0] - y_j[0])
//            + (x_i[1] - y_j[1]) * (x_i[1] - y_j[1])
//            + (x_i[2] - y_j[2]) * (x_i[2] - y_j[2])
//        );

        const float r_inv = one_over_four_pi * (one - delta_ij) / (r + delta_ij);

        // Every thread holds exactly one matrix entry.
        float Re_A_ij = cos( kappa * r ) * r_inv;
        float Im_A_ij = sin( kappa * r ) * r_inv;

        // Each thread loads an entry of a 32 x 32 block of B.
        
        const uint pos = 1024 * j_chunk + 32 * i_loc + j_loc;
        
        s_Re_BT[j_loc][i_loc] = g_Re_B[pos];
        s_Im_BT[j_loc][i_loc] = g_Im_B[pos];
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for( uint k_ = 0; k_ < 32; ++k_ )
        {
            uint k = ( k_ + i_loc );

            k = (k > 31) ? k - 32 : k;

            // BAD: All simdgroups require the same column of B at the same time
            
            const float Re_B_jk = s_Re_BT[k][j_loc];
            const float Im_B_jk = s_Im_BT[k][j_loc];

            const float Re_C_ijjk = Re_A_ij * Re_B_jk - Im_A_ij * Im_B_jk;
            const float Im_C_ijjk = Re_A_ij * Im_B_jk + Im_A_ij * Re_B_jk;

            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            const float Re_sum = simd_sum(Re_C_ijjk);
            const float Im_sum = simd_sum(Im_C_ijjk);
//
//            simdgroup_barrier(mem_flags::mem_threadgroup);
            
            if( j_loc == k )
            {
                Re_C_ik += Re_sum;
                Im_C_ik += Im_sum;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

        } // for( uint k = 0; k < 32; ++k )
    }
    
    const uint pos = 1024 * i_chunk + ( 32 * i_loc + j_loc);
    g_Re_C[pos] = Re_C_ik;
    g_Im_C[pos] = Im_C_ik;
}

// FIXME: Comment-out the following line for run-time compilation:
)"
