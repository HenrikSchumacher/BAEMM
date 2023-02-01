public:

    void Neumann_to_Dirichlet2(
        const MTL::Buffer * Re_B,
        const MTL::Buffer * Im_B,
              MTL::Buffer * Re_C,
              MTL::Buffer * Im_C,
        const float kappa,
        const uint n_waves,
        const uint simd_size,
        const uint vec_size,
        const bool wait = true
    )
    {
        tic(ClassName()+"::Neumann_to_Dirichlet2(...,"+ToString(n_waves)+","+ToString(simd_size)+")");
            
        if( n_waves != vec_size * simd_size )
        {
            eprint(ClassName()+"::Neumann_to_Dirichlet2: n_waves != v * simd_size");
        }
        
//        if( n_waves != simd_size )
//        {
//            eprint(ClassName()+"::Neumann_to_Dirichlet2: n_waves != simd_size");
//        }
        
        MTL::ComputePipelineState * pipeline = GetPipelineState(
            "Helmholtz__Neumann_to_Dirichlet2",
            std::string(
#include "Neumann_to_Dirichlet2.metal"
            ),
            {"int","int"},
            {"simd_size","vec_size"},
            {ToString(simd_size),ToString(vec_size)}
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
        compute_encoder->setBuffer(Re_B,       0, 1 );
        compute_encoder->setBuffer(Im_B,       0, 2 );
        compute_encoder->setBuffer(Re_C,       0, 3 );
        compute_encoder->setBuffer(Im_C,       0, 4 );
        
        compute_encoder->setBytes(&kappa,      sizeof(float),       5);
        compute_encoder->setBytes(&n,          sizeof(NS::Integer), 6);
        compute_encoder->setBytes(&n_waves,    sizeof(NS::Integer), 7);

        
//        const NS::Integer max_threads = pipeline->maxTotalThreadsPerThreadgroup();
        
//        valprint("max_threads", max_threads );

        
        MTL::Size threads_per_threadgroup ( simd_size * simd_size, 1, 1);
        MTL::Size threadgroups_per_grid   (
            DivideRoundUp( n, static_cast<UInt>(simd_size) ), 1, 1
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
        
        blit_command_encoder->synchronizeResource(Re_C);
        blit_command_encoder->synchronizeResource(Im_C);
//        blit_command_encoder->synchronizeResource(Re_A);
//        blit_command_encoder->synchronizeResource(Im_A);
        blit_command_encoder->endEncoding();
        
        
        // Execute the command buffer.
        command_buffer->commit();
        if( wait )
        {
            command_buffer->waitUntilCompleted();
        }
        
        toc(ClassName()+"::Neumann_to_Dirichlet2(...,"+ToString(n_waves)+","+ToString(simd_size)+")");
    }
