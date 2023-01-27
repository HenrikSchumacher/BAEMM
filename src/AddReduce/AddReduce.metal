
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c Helmholtz.metal -o Helmholtz.air && xcrun -sdk macosx metallib Helmholtz.air -o Helmholtz.metallib

// FIXME: Comment-out the following line for run-time compilation:
R"(
// FIXME: Comment-in the following lines for run-time compilation:
//constant constexpr uint threadgroup_size = 1024;
//constant constexpr uint chunk_size       = 256;

#include <metal_stdlib>

using namespace metal;

constant constexpr float zero    = static_cast<float>(0);
constant constexpr float one     = static_cast<float>(1);
constant constexpr float two     = static_cast<float>(2);



using Chunk_T = array<float,chunk_size>;

using UInt = uint;

inline float Reduce( const device array<float,chunk_size> & a )
{
    float sum = 0;
    
    for( uint i = 0; i < chunk_size; ++i )
    {
        sum += a[i];
    }
    
    return sum;
}

[[max_total_threads_per_threadgroup(threadgroup_size)]]
[[kernel]] void AddReduce(
    const device   Chunk_T  * const g_data_in        [[buffer(0)]],
          device   float    * const g_data_out       [[buffer(1)]],
    const constant UInt &   n                        [[buffer(2)]],
                                   
    const uint t_id                     [[thread_position_in_threadgroup]],
    const uint   id                     [[thread_position_in_grid]],
    const uint b_id                     [[threadgroup_position_in_grid]],
                          
    const uint threadgroups_per_grid    [[threadgroups_per_grid]],
    const uint threads_per_grid         [[threads_per_grid]]
)
{
//    thread Chunk_T chunk;
    threadgroup float s_data [threadgroup_size];
    
    const UInt g_count = n / (static_cast<UInt>(sizeof(Chunk_T)/static_cast<UInt>(sizeof(float))));
    
//    const uint vectors_per_thread = n_mats / threads_per_grid;
    
    float sum = {};
    
    // Each threads sums as many results as it can.
    
    // It's an exremely stupid idea to do it like this!
//    const uint i_begin = points_per_thread *  id;
//    const uint i_end   = points_per_thread * (id+1);
//    for( uint i = i_begin; i < i_end; ++i )
//    {
//        sum += g_data_in(g_data_in[i]);
//    }
    
//    // For coalescence we want to do it like this!
    for( UInt i = id; i < g_count; i += threads_per_grid )
    {
        sum += Reduce(g_data_in[i]);
    }
    
    s_data[l_id] = sum;
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // We assume that threadgroup_size is a power of 2.
    
    for( UInt k = threadgroup_size; (k >>= 1) > 0; )
    {
        if( l_id < k A)
        {
            s_data[l_id] += s_data[l_id + k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // TODO: Use simd_sum.
    
//    if(l_id < 32) { WarpReduce<threadgroup_size>(s_data, l_id); }
    
    // Write the threadgroup's net sum to output.
    if(l_id == 0)
    {
        g_data_out[b_id] = s_data[l_id];
    }
    
}


// FIXME: Comment-out the following line for run-time compilation:
)"
