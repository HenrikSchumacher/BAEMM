public:

    void AddReduce(
        const MTL::Buffer * a,
              MTL::Buffer * c,
        const UInt n,
        const UInt threadgroup_count,
        const UInt threadgroup_size,
        const UInt chunk_size = 256
    )
    {
        tic(ClassName()+"::AddReduce(..., threadgroup_count = "+ToString(threadgroup_count)+", threadgroup_size = "+ToString(threadgroup_size)+")");
        
        
        MTL::ComputePipelineState * pipeline = GetPipelineState(
            "AddReduce",
            std::string(
#include "AddReduce.metal"
            ),
            {"uint","uint"},
            {"threadgroup_size","chunk_size"},
            {ToString(threadgroup_size),ToString(chunk_size)}
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
        compute_encoder->setBuffer(a,             0, 0 );
        compute_encoder->setBuffer(c,             0, 1 );
        compute_encoder->setBytes (&n, sizeof(UInt), 2 );

        const NS::Integer max_threads = pipeline->maxTotalThreadsPerThreadgroup();
        
        if(threadgroup_size != max_threads)
        {
            wprint(ClassName()+"::Neumann_to_Dirichlet: threadgroup_size != max_threads");
        }
        
        MTL::Size threads_per_threadgroup (threadgroup_size,  1, 1);
        MTL::Size threadgroups_per_grid   (threadgroup_count, 1, 1);

        // Encode the compute command.
        compute_encoder->dispatchThreadgroups(threadgroups_per_grid, threads_per_threadgroup);
        
        // Signal that we have encoded all we want.
        compute_encoder->endEncoding();
        
        // Encode synchronization of return buffers.
        MTL::BlitCommandEncoder * blit_command_encoder = command_buffer->blitCommandEncoder();
        assert( blit_command_encoder != nullptr );
        
        blit_command_encoder->synchronizeResource(c);
        blit_command_encoder->endEncoding();
        
        
        // Execute the command buffer.
        command_buffer->commit();
        command_buffer->waitUntilCompleted();
        
        toc(ClassName()+"::AddReduce(..., threadgroup_count = "+ToString(threadgroup_count)+", threadgroup_size = "+ToString(threadgroup_size)+")");
    }
