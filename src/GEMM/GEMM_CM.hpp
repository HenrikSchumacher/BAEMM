public:
    
    void GEMM_CM(
        const uint M,
        const uint N,
        const uint K,
        const float alpha,
        const MTL::Buffer * A,
        const MTL::Buffer * B,
        const float beta,
              MTL::Buffer * C,
        const uint g_rows,
        const uint g_cols,
        const bool wait = true
    )
    {
        MTL::ComputePipelineState * pipeline = GetPipelineState(
              "GEMM_CM",
              std::string(
#include "GEMM_CM.metal"
              ),
              {"uint","uint"},
              {"g_rows","g_cols"},
              {ToString(g_rows),ToString(g_cols)}
          );
        
        assert( pipeline != nullptr );
        
        // Now we can proceed to set up the MTL::CommandBuffer.

        // Create a command buffer to hold commands.
        MTL::CommandBuffer * command_buffer = command_queue->commandBuffer();
        assert( command_buffer != nullptr );

        // Create an encoder that translates our command to something the
        // device understands
        MTL::ComputeCommandEncoder * compute_encoder = command_buffer->computeCommandEncoder();
        assert( compute_encoder != nullptr );

        // Encode the pipeline state object and its parameters.
        compute_encoder->setComputePipelineState( pipeline );
        
        // Place data in encoder
        compute_encoder->setBytes(&M,          sizeof(uint),  0);
        compute_encoder->setBytes(&N,          sizeof(uint),  1);
        compute_encoder->setBytes(&K,          sizeof(uint),  2);
        compute_encoder->setBytes(&alpha,      sizeof(float), 3);
        compute_encoder->setBuffer(A, 0 ,4 );
        compute_encoder->setBuffer(B, 0 ,5 );
        compute_encoder->setBytes(&beta,       sizeof(float), 6);
        compute_encoder->setBuffer(C, 0 ,7 );

//        const NS::Integer max_threads = pipeline->maxTotalThreadsPerThreadgroup();
        
        MTL::Size threads_per_threadgroup ( g_rows, g_cols, 1 );
        MTL::Size threadgroups_per_grid  (
            DivideRoundUp(M, 4 * static_cast<Int>(threads_per_threadgroup.width )),
            DivideRoundUp(N, 4 * static_cast<Int>(threads_per_threadgroup.height)),
            1
        );
        
        
//        valprint("M",M);
//        valprint("N",N);
//        valprint("K",K);
//        valprint("g_rows",g_rows);
//        valprint("g_cols",g_cols);
//        print("threadgroups_per_grid = { "
//              +ToString(threadgroups_per_grid.width)
//              +","
//              +ToString(threadgroups_per_grid.height)
//              +","
//              +ToString(threadgroups_per_grid.depth)
//              +" }."
//        );
//        print("threads_per_threadgroup + { "
//            +ToString(threads_per_threadgroup.width)
//            +","
//            +ToString(threads_per_threadgroup.height)
//            +","
//            +ToString(threads_per_threadgroup.depth)
//            +" }."
//        );
//        valprint("SIMD group size", pipeline->threadExecutionWidth() );

        // Encode the compute command.
        compute_encoder->dispatchThreadgroups(threadgroups_per_grid, threads_per_threadgroup);
        
        // Signal that we have encoded all we want.
        compute_encoder->endEncoding();
        
        // Encode synchronization of return buffers.
        MTL::BlitCommandEncoder * blit_command_encoder = command_buffer->blitCommandEncoder();
        assert( blit_command_encoder != nullptr );
        
        blit_command_encoder->synchronizeResource(C);
        blit_command_encoder->endEncoding();
        
        
        // Execute the command buffer.
        command_buffer->commit();
        if( wait )
        {
            command_buffer->waitUntilCompleted();
        }
    }
