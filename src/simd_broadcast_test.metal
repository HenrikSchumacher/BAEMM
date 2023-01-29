R"(
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c Helmholtz.metal -o Helmholtz.air && xcrun -sdk macosx metallib Helmholtz.air -o Helmholtz.metallib

// FIXME: Comment-out the following line for run-time compilation:

// FIXME: Comment-in the following two lines for run-time compilation:
constant constexpr int simd_size = 32;

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


[[kernel]] void simd_broadcast_test(
          device   float  * const g_buffer       [[buffer(0)]],
    const constant uint   &       n              [[buffer(1)]],
                                                 
    const uint  simdgroup_index_in_threadgroup   [[simdgroup_index_in_threadgroup]],
    const uint  thread_index_in_simdgroup        [[thread_index_in_simdgroup]],
                                                 
    const uint  i_chunk                          [[threadgroup_position_in_grid]],
    const uint  rows                             [[simdgroups_per_threadgroup]],
    const uint  cols                             [[threads_per_simdgroup]]
)
{
    assert( rows == simd_size );
    assert( cols == simd_size );
    
    const int simd_thread = thread_index_in_simdgroup;
    const int simd_group  = simdgroup_index_in_threadgroup;
    
    const int i = simd_size * i_chunk + simd_group;
    
    const int local_id = simd_size * simd_group + simd_thread;
    
    thread float threads_value = simd_thread;
    
    thread float value = simd_broadcast( threads_value, simd_group );
//    thread float value = simd_sum( threads_value );
    
    g_buffer[1024 * i_chunk + local_id] = value;
}

// FIXME: Comment-out the following line for run-time compilation:
)"
