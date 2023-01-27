public:
    
    void Neumann_to_Dirichlet(
        const MTL::Buffer * Re_Y,
        const MTL::Buffer * Im_Y,
              MTL::Buffer * Re_X,
              MTL::Buffer * Im_X,
        const float kappa,
        const float kappa_step,
        const uint chunk_size,
        const uint n_waves,
        const bool wait = true
    )
    {
        tic(ClassName()+"::Neumann_to_Dirichlet(...,"+ToString(chunk_size)+","+ToString(n_waves)+")");
        
        MTL::ComputePipelineState * pipeline = GetPipelineState(
              "Helmholtz__Neumann_to_Dirichlet",
              std::string(
#include "Neumann_to_Dirichlet.metal"
              ),
              {"uint","uint"},
              {"chunk_size","n_waves"},
              {ToString(chunk_size),ToString(n_waves)}
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
        compute_encoder->setBuffer(mid_points, 0 ,0 );
        compute_encoder->setBuffer(Re_Y,       0, 1 );
        compute_encoder->setBuffer(Im_Y,       0, 2 );
        compute_encoder->setBuffer(Re_X,       0, 3 );
        compute_encoder->setBuffer(Im_X,       0, 4 );
        
        compute_encoder->setBytes(&kappa,      sizeof(float),       5);
        compute_encoder->setBytes(&kappa_step, sizeof(float),       6);
        compute_encoder->setBytes(&n,          sizeof(NS::Integer), 7);

        const NS::Integer max_threads = pipeline->maxTotalThreadsPerThreadgroup();
        
        if(chunk_size != max_threads)
        {
            wprint(ClassName()+"::Neumann_to_Dirichlet: chunk_size != max_threads");
        }
        
        MTL::Size threads_per_threadgroup (max_threads, 1, 1);
        MTL::Size threadgroups_per_grid  (
            DivideRoundUp(n, static_cast<Int>(threads_per_threadgroup.width)), 1, 1
        );
        
        
        valprint("n",n);
        valprint("threadgroups_per_grid",threadgroups_per_grid.width);
        valprint("threads_per_threadgroup",threads_per_threadgroup.width);
        valprint("SIMD group size", pipeline->threadExecutionWidth() );

        // Encode the compute command.
        compute_encoder->dispatchThreadgroups(threadgroups_per_grid, threads_per_threadgroup);
        
        // Signal that we have encoded all we want.
        compute_encoder->endEncoding();
        
        // Encode synchronization of return buffers.
        MTL::BlitCommandEncoder * blit_command_encoder = command_buffer->blitCommandEncoder();
        assert( blit_command_encoder != nullptr );
        
        blit_command_encoder->synchronizeResource(Re_X);
        blit_command_encoder->synchronizeResource(Im_X);
        blit_command_encoder->endEncoding();
        
        
        // Execute the command buffer.
        command_buffer->commit();
        if( wait )
        {
            command_buffer->waitUntilCompleted();
        }
        
        toc(ClassName()+"::Neumann_to_Dirichlet(...,"+ToString(chunk_size)+","+ToString(n_waves)+")");
    }
