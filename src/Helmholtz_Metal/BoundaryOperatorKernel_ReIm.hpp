public:

    void BoundaryOperatorKernel_ReIm(
        const MTL::Buffer * Re_B,
        const MTL::Buffer * Im_B,
              MTL::Buffer * Re_C,
              MTL::Buffer * Im_C,
        const std::vector<Real>      & kappa,
        const std::array <Complex,3> & coeff,
        const int  wave_count,
        const int  block_size,
        const int  wave_chunk_size,
        const bool wait = true
    )
    {
        std::string name ( "BoundaryOperatorKernel_ReIm" );
        
        bool single_layer = std::abs(coeff[0]) != zero;
        bool double_layer = std::abs(coeff[1]) != zero;
        bool adjdbl_layer = std::abs(coeff[2]) != zero;
        
        tic(ClassName()+"::"+name+"(...,"+ToString(block_size)+","+ToString(wave_chunk_size)+","+ToString(single_layer)+","+ToString(double_layer)+","+ToString(adjdbl_layer)+")");
        
        MTL::ComputePipelineState * pipeline = GetPipelineState(
              name,
              std::string(
#include "BoundaryOperatorKernel_ReIm.metal"
              ),
              {"int","int","bool","bool","bool"},
              {"block_size","wave_chunk_size","single_layer","double_layer","adjdbl_layer"},
              {ToString(block_size),ToString(wave_chunk_size),ToString(single_layer),ToString(double_layer),ToString(adjdbl_layer)}
          );
        
        const int chunk_count = wave_count / wave_chunk_size;
        
        if( chunk_count != kappa.size() )
        {
            eprint(ClassName()+"::"+name+": kappa.size() != wave_count / wave_chunk_size. Aborting.");
            return;
        }
        
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

        LoadParameters(coeff);
        
        // Place data in encoder
        compute_encoder->setBuffer(mid_points,   0, 0 );
        compute_encoder->setBuffer(normals   ,   0, 1 );
        compute_encoder->setBuffer(Re_B,         0, 2 );
        compute_encoder->setBuffer(Im_B,         0, 3 );
        compute_encoder->setBuffer(Re_C,         0, 4 );
        compute_encoder->setBuffer(Im_C,         0, 5 );
        compute_encoder->setBytes(kappa.data(), kappa.size() * sizeof(Real ), 6 );
        compute_encoder->setBytes(&coeff_over_four_pi[0],  8 * sizeof(Real ), 7 );
        compute_encoder->setBytes(&simplex_count,              sizeof(int  ), 8 );
        compute_encoder->setBytes(&wave_count,                 sizeof(int  ), 9 );

        
        const NS::Integer max_threads = pipeline->maxTotalThreadsPerThreadgroup();

        
        MTL::Size threads_per_threadgroup (max_threads, 1, 1);
        MTL::Size threadgroups_per_grid  (
            CeilDivide(simplex_count, static_cast<Int>(threads_per_threadgroup.width)), 1, 1
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
        
        toc(ClassName()+"::"+name+"(...,"+ToString(block_size)+","+ToString(wave_chunk_size)+","+ToString(single_layer)+","+ToString(double_layer)+","+ToString(adjdbl_layer)+")");
    }
