public:

    void simd_broadcast_test(
              MTL::Buffer * buffer,
        const uint N,
        const bool wait = true
    )
    {
        tic(ClassName()+"::simd_broadcast_test");
        
        MTL::ComputePipelineState * pipeline = GetPipelineState(
            "simd_broadcast_test",
            std::string(
#include "simd_broadcast_test.metal"
            ),
            {},
            {},
            {}
          );
        
        constexpr uint simd_size = 32;
        
        assert( pipeline != nullptr );
        
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
        compute_encoder->setBuffer(buffer, 0 ,0 );
        compute_encoder->setBytes(&N, sizeof(NS::Integer), 1);

        
//        const NS::Integer max_threads = pipeline->maxTotalThreadsPerThreadgroup();
        
//        valprint("max_threads", max_threads );

        
        MTL::Size threads_per_threadgroup ( simd_size * simd_size, 1, 1);
        MTL::Size threadgroups_per_grid   (
            DivideRoundUp( N, static_cast<UInt>(simd_size) ), 1, 1
        );
        
        if( pipeline->threadExecutionWidth() != simd_size )
        {
            wprint("pipeline->threadExecutionWidth() != simd_size");
        }
        
//        valprint("n",n);
//        valprint("threadgroups_per_grid", threadgroups_per_grid.width );
//        valprint("threads_per_threadgroup",threads_per_threadgroup.width);
//        valprint("SIMD group size", pipeline->threadExecutionWidth() );

        // Encode the compute command.
        compute_encoder->dispatchThreadgroups(threadgroups_per_grid, threads_per_threadgroup);
        
        // Signal that we have encoded all we want.
        compute_encoder->endEncoding();
        
        // Encode synchronization of return buffers.
        MTL::BlitCommandEncoder * blit_command_encoder = command_buffer->blitCommandEncoder();
        assert( blit_command_encoder != nullptr );
        
        blit_command_encoder->synchronizeResource(buffer);
        blit_command_encoder->endEncoding();
        
        
        // Execute the command buffer.
        command_buffer->commit();
        if( wait )
        {
            command_buffer->waitUntilCompleted();
        }
        
        toc(ClassName()+"::simd_broadcast_test");
    }
