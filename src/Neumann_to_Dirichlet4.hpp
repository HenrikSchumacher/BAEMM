public:

    void Neumann_to_Dirichlet4(
        const MTL::Buffer * B,
              MTL::Buffer * C,
        const float kappa,
        const float kappa_step,
        const uint block_size,
        const uint n_waves,
        const bool wait = true
    )
    {
        tic(ClassName()+"::Neumann_to_Dirichlet4(...,"+ToString(block_size)+","+ToString(n_waves)+")");
        
        
        MTL::ComputePipelineState * pipeline = GetPipelineState(
            "Helmholtz__Neumann_to_Dirichlet4",
            std::string(
#include "Neumann_to_Dirichlet4.metal"
            ),
            {"int","int"},
            {"block_size","n_waves"},
            {ToString(block_size),ToString(n_waves)}
          );
        
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
        compute_encoder->setBuffer(mid_points, 0 ,0 );
        compute_encoder->setBuffer(B,          0, 1 );
        compute_encoder->setBuffer(C,          0, 2 );
        
        compute_encoder->setBytes(&kappa,      sizeof(float),       3);
        compute_encoder->setBytes(&kappa_step, sizeof(float),       4);
        compute_encoder->setBytes(&n,          sizeof(NS::Integer), 5);
        
        MTL::Size threads_per_threadgroup ( block_size, 1, 1);
        MTL::Size threadgroups_per_grid   (
            DivideRoundUp( n, static_cast<UInt>(block_size) ), 1, 1
        );
        
        if( pipeline->maxTotalThreadsPerThreadgroup() != block_size )
        {
            wprint("pipeline->maxTotalThreadsPerThreadgroup() != block_size");
        }
        
        valprint("n",n);
        valprint("threadgroups_per_grid", threadgroups_per_grid.width );
        valprint("threads_per_threadgroup",threads_per_threadgroup.width);
        valprint("SIMD group size", pipeline->threadExecutionWidth() );

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
        
        toc(ClassName()+"::Neumann_to_Dirichlet4(...,"+ToString(block_size)+","+ToString(n_waves)+")");
    }
