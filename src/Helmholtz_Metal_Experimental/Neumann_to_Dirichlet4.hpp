public:

    void Neumann_to_Dirichlet4(
        const MTL::Buffer * Re_B,
        const MTL::Buffer * Im_B,
              MTL::Buffer * Re_C,
              MTL::Buffer * Im_C,
        const float kappa,
        const float single_layer_coeff,
        const float double_layer_coeff,
        const float adjdbl_layer_coeff,
        const int wave_count,
        const int block_size,
        const int wave_chunk_size,
        const bool wait = true
    )
    {
        bool single_layer = single_layer_coeff != 0.f;
        bool double_layer = double_layer_coeff != 0.f;
        bool adjdbl_layer = adjdbl_layer_coeff != 0.f;
        
        tic(ClassName()+"::Neumann_to_Dirichlet4(...,"+ToString(block_size)+","+ToString(wave_chunk_size)+","+ToString(single_layer)+","+ToString(double_layer)+","+ToString(adjdbl_layer)+")");
        
        MTL::ComputePipelineState * pipeline = GetPipelineState(
              "Helmholtz__Neumann_to_Dirichlet4",
              std::string(
#include "Neumann_to_Dirichlet4.metal"
              ),
              {"int","int","bool","bool","bool"},
              {"block_size","wave_chunk_size","single_layer","double_layer","adjdbl_layer"},
              {ToString(block_size),ToString(wave_chunk_size),ToString(single_layer),ToString(double_layer),ToString(adjdbl_layer)}
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

        // Beware: float3 has size 16 Byte due to paddding. So we have to pad here, too.
        float coeff [4] = { single_layer_coeff, double_layer_coeff, adjdbl_layer_coeff, 0.f};
        
        // Place data in encoder
        compute_encoder->setBuffer(mid_points, 0 ,0 );
        compute_encoder->setBuffer(normals   , 0 ,1 );
        compute_encoder->setBuffer(Re_B,       0, 2 );
        compute_encoder->setBuffer(Im_B,       0, 3 );
        compute_encoder->setBuffer(Re_C,       0, 4 );
        compute_encoder->setBuffer(Im_C,       0, 5 );
        
        compute_encoder->setBytes(&kappa,          sizeof(float), 6);
        compute_encoder->setBytes(&coeff[0],   4 * sizeof(float), 7);
        compute_encoder->setBytes(&n,              sizeof(int  ), 8);
        compute_encoder->setBytes(&wave_count,     sizeof(int  ), 9);

        
        
        const NS::Integer max_threads = pipeline->maxTotalThreadsPerThreadgroup();

//        if(block_size != max_threads)
//        {
//            wprint(ClassName()+"::Neumann_to_Dirichlet4: block_size != max_threads");
//        }
        
        MTL::Size threads_per_threadgroup (max_threads, 1, 1);
        MTL::Size threadgroups_per_grid  (
            CeilDivide(n, static_cast<Int>(threads_per_threadgroup.width)), 1, 1
        );
        
//        valprint("n",n);
//        valprint("threadgroups_per_grid",threadgroups_per_grid.width);
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
        blit_command_encoder->endEncoding();
        
        
        // Execute the command buffer.
        command_buffer->commit();
        if( wait )
        {
            command_buffer->waitUntilCompleted();
        }
        
        toc(ClassName()+"::Neumann_to_Dirichlet4(...,"+ToString(block_size)+","+ToString(wave_chunk_size)+","+ToString(single_layer)+","+ToString(double_layer)+","+ToString(adjdbl_layer)+")");
    }
