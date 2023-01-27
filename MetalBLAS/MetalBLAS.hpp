//
//  Metal_BLAS.hpp
//  Repulsion
//
//  Created by Henrik on 05.09.22.
//

#pragma once


#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include <map>

#define CLASS MetalBLAS

class CLASS
{
private:
    
    MTL::Device* device;
    
    std::map<std::string, MTL::Function * > functionMap;
    std::map<std::string, MTL::ComputePipelineState * > functionPipelineMap;
    
    MTL::CommandQueue * command_queue;
    
public:
    
    CLASS(MTL::Device* device_)
    :   device( device_ )
    {
        NS::Error *error = nullptr;
        
        // Load the shader files with a .metal file extension in the project
        auto filepath = NS::String::string(
            "/Users/Henrik/github/Repulsion/Example_Metal/BLAS.metallib",
            NS::ASCIIStringEncoding
        );
        
        MTL::Library * lib = device->newLibrary(filepath, &error);
        
        if( lib == nullptr )
        {
            std::cerr << "Failed to load library." << std::endl;
            std::exit(-1);
        }
        
        // Get all function names
        auto function_names = lib->functionNames();
        
        for (NS::UInteger i = 0; i < function_names->count(); i++)
        {
            auto name_nsstring = function_names->object(i)->description();
            auto name_utf8 = name_nsstring->utf8String();
            
//            print(name_utf8);

            // Load function into a map
            functionMap[name_utf8] = (lib->newFunction(name_nsstring));

            // Create pipeline from function
            functionPipelineMap[name_utf8] =
                device->newComputePipelineState(functionMap[name_utf8], &error);

            if (functionPipelineMap[name_utf8] == nullptr)
            {
                std::cout << "Failed to created pipeline state object for "
                          << name_utf8 << ", error "
                          << error->description()->utf8String() << std::endl;
                return;
            }
        }
        
        command_queue = device->newCommandQueue();
        
        if( command_queue == nullptr )
        {
            std::cout << "Failed to find the command queue." << std::endl;
            return;
        }
    }
    
    ~CLASS() = default;

    
    
    void Copy(
        MTL::Buffer * from,
        MTL::Buffer * to,
        const NS::UInteger size
    )
    {
        tic("Copy");
        
        // Create a command buffer to hold commands.
        MTL::CommandBuffer * command_buffer = command_queue->commandBuffer();
        assert( command_buffer != nullptr );

        
        // Encode a blit pass to copy data from the source buffer to the private buffer.
        MTL::BlitCommandEncoder * blit_command_encoder = command_buffer->blitCommandEncoder();
        assert( blit_command_encoder != nullptr );
        blit_command_encoder->copyFromBuffer(from, 0, to, 0, size);

        blit_command_encoder->endEncoding();

        
        command_buffer->addCompletedHandler( ^void( MTL::CommandBuffer * pCmd ){});

        // Private buffer is populated.
        
        command_buffer->commit();
        
//        command_buffer->waitUntilCompleted();
        
        toc("Copy");
    }
    
    
    
    void GEMM_GPU_nn(
              MTL::Buffer * A,
              MTL::Buffer * B,
              MTL::Buffer * C,
        const NS::Integer M,
        const NS::Integer N,
        const NS::Integer K
    )
    {
        // Create a command buffer to hold commands.
        MTL::CommandBuffer* command_buffer = command_queue->commandBuffer();
        assert( command_buffer != nullptr );

        // Create an encoder that translates our command to something the
        // device understands
        MTL::ComputeCommandEncoder* compute_encoder = command_buffer->computeCommandEncoder();
        assert( compute_encoder != nullptr );

        MTL::ComputePipelineState * pipeline = functionPipelineMap["gpu_gemm_nn"];
        assert( pipeline != nullptr );

        // Encode the pipeline state object and its parameters.
        compute_encoder->setComputePipelineState( pipeline );

        // Place data in encoder
        compute_encoder->setBuffer(A, 0, 0);
        compute_encoder->setBuffer(B, 0, 1);
        compute_encoder->setBuffer(C, 0, 2);
        compute_encoder->setBytes(&M, sizeof(NS::Integer), 3);
        compute_encoder->setBytes(&N, sizeof(NS::Integer), 4);
        compute_encoder->setBytes(&K, sizeof(NS::Integer), 5);

//        const NS::Integer max_threads = pipeline->maxTotalThreadsPerThreadgroup();
        
        MTL::Size threads_per_threadgroup = MTL::Size(32, 32, 1);
        MTL::Size threadgroups_per_grid   = MTL::Size(
            (M+threads_per_threadgroup.width-1)/threads_per_threadgroup.width,
            (N+threads_per_threadgroup.height-1)/threads_per_threadgroup.height,
            1
        );

//        valprint( "threads_per_threadgroup.width ", threads_per_threadgroup.width );
//        valprint( "threads_per_threadgroup.height", threads_per_threadgroup.height );
//        valprint( "threadgroups_per_grid.width   ", threadgroups_per_grid.width );
//        valprint( "threadgroups_per_grid.height  ", threadgroups_per_grid.height );

        // Encode the compute command.
        compute_encoder->dispatchThreadgroups(threadgroups_per_grid, threads_per_threadgroup);

        // Signal that we have encoded all we want.
        compute_encoder->endEncoding();

        // Execute the command buffer.
        command_buffer->commit();

        command_buffer->waitUntilCompleted();
    }
    
    void GEMM_GPU_sh_nn(
              MTL::Buffer * A,
              MTL::Buffer * B,
              MTL::Buffer * C,
        const NS::Integer M,
        const NS::Integer N,
        const NS::Integer K
    )
    {
        // Create a command buffer to hold commands.
        MTL::CommandBuffer* command_buffer = command_queue->commandBuffer();
        assert( command_buffer != nullptr );

        // Create an encoder that translates our command to something the
        // device understands
        MTL::ComputeCommandEncoder* compute_encoder = command_buffer->computeCommandEncoder();
        assert( compute_encoder != nullptr );

        MTL::ComputePipelineState * pipeline = functionPipelineMap["gpu_gemm_sh_nn"];
        assert( pipeline != nullptr );

        // Encode the pipeline state object and its parameters.
        compute_encoder->setComputePipelineState( pipeline );

        // Place data in encoder
        compute_encoder->setBuffer(A, 0, 0);
        compute_encoder->setBuffer(B, 0, 1);
        compute_encoder->setBuffer(C, 0, 2);
        compute_encoder->setBytes(&M, sizeof(NS::Integer), 3);
        compute_encoder->setBytes(&N, sizeof(NS::Integer), 4);
        compute_encoder->setBytes(&K, sizeof(NS::Integer), 5);

//        const NS::Integer max_threads = pipeline->maxTotalThreadsPerThreadgroup();
        
        MTL::Size threads_per_threadgroup = MTL::Size(16, 16, 1);
        MTL::Size threadgroups_per_grid   = MTL::Size(
            (M+threads_per_threadgroup.width-1)/threads_per_threadgroup.width,
            (N+threads_per_threadgroup.height-1)/threads_per_threadgroup.height,
            1
        );

//        valprint( "threads_per_threadgroup.width ", threads_per_threadgroup.width );
//        valprint( "threads_per_threadgroup.height", threads_per_threadgroup.height );
//        valprint( "threadgroups_per_grid.width   ", threadgroups_per_grid.width );
//        valprint( "threadgroups_per_grid.height  ", threadgroups_per_grid.height );

        // Encode the compute command.
        compute_encoder->dispatchThreadgroups(threadgroups_per_grid, threads_per_threadgroup);

        // Signal that we have encoded all we want.
        compute_encoder->endEncoding();

        // Execute the command buffer.
        command_buffer->commit();

        command_buffer->waitUntilCompleted();
    }
    
    void GEMM_GPU_sh_reg_nn(
              MTL::Buffer * A,
              MTL::Buffer * B,
              MTL::Buffer * C,
        const NS::Integer M,
        const NS::Integer N,
        const NS::Integer K
    )
    {
        // Create a command buffer to hold commands.
        MTL::CommandBuffer* command_buffer = command_queue->commandBuffer();
        assert( command_buffer != nullptr );

        // Create an encoder that translates our command to something the
        // device understands
        MTL::ComputeCommandEncoder* compute_encoder = command_buffer->computeCommandEncoder();
        assert( compute_encoder != nullptr );

        MTL::ComputePipelineState * pipeline = functionPipelineMap["gpu_gemm_sh_reg_nn"];
        assert( pipeline != nullptr );

        // Encode the pipeline state object and its parameters.
        compute_encoder->setComputePipelineState( pipeline );

        // Place data in encoder
        compute_encoder->setBuffer(A, 0, 0);
        compute_encoder->setBuffer(B, 0, 1);
        compute_encoder->setBuffer(C, 0, 2);
        compute_encoder->setBytes(&M, sizeof(NS::Integer), 3);
        compute_encoder->setBytes(&N, sizeof(NS::Integer), 4);
        compute_encoder->setBytes(&K, sizeof(NS::Integer), 5);

//        const NS::Integer max_threads = pipeline->maxTotalThreadsPerThreadgroup();
        
        MTL::Size threads_per_threadgroup = MTL::Size(16, 16, 1);
        MTL::Size threadgroups_per_grid   = MTL::Size(
            (M+threads_per_threadgroup.width-1)/threads_per_threadgroup.width,
            (N+threads_per_threadgroup.height-1)/threads_per_threadgroup.height,
            1
        );

//        valprint( "threads_per_threadgroup.width ", threads_per_threadgroup.width );
//        valprint( "threads_per_threadgroup.height", threads_per_threadgroup.height );
//        valprint( "threadgroups_per_grid.width   ", threadgroups_per_grid.width );
//        valprint( "threadgroups_per_grid.height  ", threadgroups_per_grid.height );

        // Encode the compute command.
        compute_encoder->dispatchThreadgroups(threadgroups_per_grid, threads_per_threadgroup);

        // Signal that we have encoded all we want.
        compute_encoder->endEncoding();

        // Execute the command buffer.
        command_buffer->commit();

        command_buffer->waitUntilCompleted();
    }
};

#undef CLASS



//[[kernel]] void gpu_gemm_sh_reg_nn(
//    device const Real * __restrict__ const A,  // pointer to A matrix data
//    device const Real * __restrict__ const B,  // pointer to B matrix data
//    device       Real * __restrict__ const C,  // pointer to C matrix data
//    device const UInt & M,
//    device const UInt & N,
//    device const UInt & K,          //in: matrix dimensions: C(M,N)+=A(M,k)*B(k,N)
//
//    const uint2 blk_count [[threadgroups_per_grid]],
//    const uint2 blk_size  [[threads_per_threadgroup]],
//    const uint2 blk       [[threadgroup_position_in_grid]],
//    const uint2 idx       [[thread_position_in_threadgroup]]
//)
//{
//    constexpr UInt tile_M = 16;
//    constexpr UInt tile_N = 16;
//    constexpr UInt tile_K = 16;
//
//
//    // Caution: I think this is column-major matrix format!
//    // _sh  -> uses shared memory = threadgroup memory
//    // _reg -> uses register = thread memory
//    // nn   -> both A and B are not transposed, i.e. CblasNoTrans
//
//    // TODO: Does the order of the loads from A and B matter at all?
//
//    threadgroup Real A_tile[tile_K][tile_M];
//    threadgroup Real B_tile[tile_N][tile_K];
//
//    for( UInt n_begin = blk.y * tile_N; n_begin < N; n_begin += blk_count.y * tile_N )
//    {
//        //tile offset in Y dimension
//
//        const UInt n_upper    = n_begin + tile_N;
//        const bool n_complete = n_upper<=N;
//        const UInt n_end      = n_complete ? N : n_upper;
//
//        for( UInt m_begin = blk.x * tile_M; m_begin < M; m_begin += blk_count.x * tile_M )
//        {
//            //tile offset in X dimension
//
//            const UInt m_upper    = m_begin + tile_M;
//            const bool m_complete = m_upper<=M;
//            const UInt m_end      = m_complete ? M : m_upper;
//
//            if( n_complete && m_complete )
//            {
//                // tile C(tile_M,tile_N) is complete
//
//                //Initialize registers to zero:
//                thread Real A_reg[4]    =  {static_cast<Real>(0.0)};
//                thread Real B_reg[4]    =  {static_cast<Real>(0.0)};
//                thread Real C_reg[4][4] = {{static_cast<Real>(0.0)}};
//
//                for( UInt k_begin = 0; k_begin < K; k_begin += tile_K )
//                {
//                    //k_begin is the position of the CUDA thread along the K dimension
//
//                    const UInt k_upper    = k_begin + tile_K;
//                    const bool k_complete = k_upper<=K;
//                    const UInt k_end      = k_complete ? K : k_upper;
//
//
//                    // TODO: That does not make sense to me. Why is the faster index k in the outer dimension of A?
//
//                    //Load a tile of matrix A(m_begin:m_end, k_begin:k_end):
//                    for(
//                        UInt m = m_begin + idx.x, m_loc = idx.x;
//                        m < m_end;
//                        m += blk_size.x, ++m_loc
//                    )
//                    {
//                        for(
//                            UInt k = k_begin + idx.y, k_loc = idx.y;
//                            k < k_end;
//                            k += blk_size.y, ++k_loc
//                            )
//                        {
//                            A_tile[k_loc][m_loc] = A[k * M + m];
//                        }
//                    }
//
//                    //Load a tile of matrix B(k_begin:k_end, n_begin:n_end):
//                    for(
//                        UInt n = n_begin + idx.y, n_loc = idx.y;
//                        n < n_end;
//                        n += blk_size.y, ++n_loc
//                        )
//                    {
//                        for(
//                            UInt k = k_begin + idx.x, k_loc = idx.x;
//                            k < k_end;
//                            k += blk_size.x, ++k_loc
//                        )
//                        {
//                            B_tile[n_loc][k_loc] = B[n*K + k];
//                        }
//                    }
//
//                    threadgroup_barrier(mem_flags::mem_threadgroup);
//
//                    //Multiply two loaded tiles to produce a tile of matrix C(m_begin:m_end,n_begin:n_end):
//                    if( k_complete )
//                    {
//                        // k-dimension is complete which means we know the length of the outer loop. Let's unroll!
//
//                        #pragma unroll
//                        for( UInt k_loc = 0; k_loc < tile_K; ++k_loc )
//                        {
//                            #pragma unroll
//                            for( UInt j = 0; j < 4; ++j )
//                            {
//                                B_reg[j] = B_tile[idx.y + blk_size.y*j][k_loc];
//                            }
//
//                            #pragma unroll
//                            for( UInt i = 0; i < 4; ++i )
//                            {
//                                A_reg[i] = A_tile[k_loc][idx.x + blk_size.x*i];
//                            }
//
//                            // add outer product!
//                            #pragma unroll
//                            for( UInt j = 0; j < 4; ++j )
//                            {
//                                #pragma unroll
//                                for( UInt i = 0; i < 4; ++i )
//                                {
//                                    C_reg[j][i] += A_reg[i] * B_reg[j];
//                                }
//                            }
//                        }
//                    }
//                    else
//                    {
//                        // Separate code for incomplete k-dimension since we cannot unroll here.
//
//                        for( UInt k_loc = 0; k_loc < (k_end - k_begin); ++k_loc )
//                        {
//                            // Load row of matrix A_tile to registers.
//                            #pragma unroll
//                            for( UInt i = 0; i < 4; ++i)
//                            {
//                                A_reg[i] = A_tile[k_loc][idx.x + blk_size.x*i];
//                            }
//
//                            // Load column of matrix B_tile to registers.
//                            #pragma unroll
//                            for( UInt j = 0; j < 4; ++j)
//                            {
//                                B_reg[j] = B_tile[idx.y + blk_size.y*j][k_loc];
//                            }
//
//                            // add outer product!
//                            #pragma unroll
//                            for( UInt j = 0; j < 4; ++j )
//                            {
//                                #pragma unroll
//                                for( UInt i = 0; i < 4; ++i )
//                                {
//                                    C_reg[j][i] += A_reg[i] * B_reg[j];
//                                }
//                            }
//                        }
//                    }
//                    threadgroup_barrier(mem_flags::mem_threadgroup);
//
//                } //k_begin
//
//                //Store elements of the C matrix in global memory:
//                #pragma unroll
//                for( UInt j = 0; j < 4; ++j )
//                {
//                    #pragma unroll
//                    for( UInt i = 0; i < 4; ++i )
//                    {
//                        C[(n_begin + idx.y + blk_size.y*j)*M + (m_begin + idx.x + blk_size.x*i)] += C_reg[j][i];
//                    }
//                }
//
//            }
//            else
//            { //incomplete tile of C
//
//                //Initialize registers to zero:
//                thread Real C_reg[4][4] = {{static_cast<Real>(0.0)}};
//                thread Real B_reg[4]    =  {static_cast<Real>(0.0)};
//                thread Real A_reg[4]    =  {static_cast<Real>(0.0)};
//
//                for( UInt k_begin = 0; k_begin < K; k_begin += tile_K )
//                {
//                    //k_begin is the position of the CUDA thread along the K dimension
//                    const UInt k_upper    = k_begin + tile_K;
//                    const bool k_complete = k_upper<=K;
//                    const UInt k_end      = k_complete ? K : k_upper;
//
//                    //Load a tile of matrix A(m_begin:m_end, k_begin:k_end):
//                    for(
//                        UInt m = m_begin + idx.x, m_loc = idx.x;
//                        m < m_end;
//                        m += blk_size.x, ++m_loc
//                    )
//                    {
//                        for(
//                            UInt k = k_begin + idx.y, k_loc = idx.y;
//                            k < k_end;
//                            k += blk_size.y, ++k_loc
//                            )
//                        {
//                            A_tile[k_loc][m_loc] = A[k * M + m];
//                        }
//                    }
//
//                    //Load a tile of matrix B(k_begin:k_end, n_begin:n_end):
//                    for(
//                        UInt n = n_begin + idx.y, n_loc = idx.y;
//                        n < n_end;
//                        n += blk_size.y, ++n_loc
//                        )
//                    {
//                        for(
//                            UInt k = k_begin + idx.x, k_loc = idx.x;
//                            k < k_end;
//                            k += blk_size.x, ++k_loc
//                        )
//                        {
//                            B_tile[n_loc][k_loc] = B[n*K + k];
//                        }
//                    }
//
//                    threadgroup_barrier(mem_flags::mem_threadgroup);
//
//                    //Multiply two loaded tiles to produce a tile of matrix C(m_begin:m_end,n_begin:n_end):
//                    for( UInt k_loc = 0; k_loc < (k_end - k_begin); ++k_loc )
//                    {
//
//                        // TODO: According to the code above this should be a loop of length at most 4. Why?
//                        for(
//                            UInt i_loc = idx.x, i = 0;
//                            i_loc < m_end - m_begin;
//                            i_loc += blk_size.x, ++i
//                        )
//                        {
//                            A_reg[i] = A_tile[k_loc][i_loc];
//                        }
//
//                        // TODO: According to the code above this should be a loop of length at most 4. Why?
//                        for(
//                            UInt j_loc = idx.y, j = 0;
//                            j_loc < n_end - n_begin;
//                            j_loc += blk_size.y, ++j
//                        )
//                        {
//                            B_reg[j] = B_tile[j_loc][k_loc];
//                        }
//
//                        #pragma unroll
//                        for( UInt j = 0; j < 4; ++j )
//                        {
//                            #pragma unroll
//                            for( UInt i = 0; i < 4; ++i )
//                            {
//                                C_reg[j][i] += A_reg[i] * B_reg[j];
//                            }
//                        }
//                    }
//
//                    threadgroup_barrier(mem_flags::mem_threadgroup);
//
//                } //k_begin
//
//                //Store element of the C matrix in global memory:
//                for(
//                    UInt n = n_begin + idx.y, j = 0;
//                    n < n_end;
//                    n += blk_size.y, ++j
//                )
//                {
//                    for(
//                        UInt m = m_begin + idx.x, i = 0;
//                        m < m_end;
//                        m += blk_size.x, ++i
//                    )
//                    {
//                        C[n*M + m] = C_reg[j][i];
//                    }
//                }
//
//            }
//
//        } //m_begin
//
//    } //n_begin
//
//    return;
//
//} // gpu_sh_reg_nn
