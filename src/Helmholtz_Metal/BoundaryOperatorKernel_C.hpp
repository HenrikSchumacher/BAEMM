public:

    void BoundaryOperatorKernel_C( const std::vector<Real> & kappa )
    {
        std::string name ("BoundaryOperatorKernel_C");
        
        if( !B_loaded )
        {
            wprint(ClassName()+"::BoundaryOperatorKernel_C: No values loaded into B. doing nothing.");
        }
        
        std::string tag = ClassName()+"::"+name+"(...,"+ToString(block_size)+","+ToString(wave_chunk_size)+","+ToString(single_layer)+","+ToString(double_layer)+","+ToString(adjdbl_layer)+")";
        
        ptic(tag);
        tic(tag);
        
        MTL::ComputePipelineState * pipeline = GetPipelineState(
            name,
            std::string(
#include "BoundaryOperatorKernel_C.metal"
            ),
            {"int","int","bool","bool","bool"},
            {"block_size","wave_chunk_size","single_layer","double_layer","adjdbl_layer"},
            {
              ToString(block_size),
              ToString(wave_chunk_size),
              ToString(single_layer),
              ToString(double_layer),
              ToString(adjdbl_layer)
            }
        );
        
        if( kappa.size() != wave_count / wave_chunk_size )
        {
            dump(kappa.size());
            dump(wave_count);
            dump(wave_chunk_size);
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
        
        // Place data in encoder
        compute_encoder->setBuffer(mid_points,   0, 0 );
        compute_encoder->setBuffer(normals   ,   0, 1 );
        compute_encoder->setBuffer(B_buf,        0, 2 );
        compute_encoder->setBuffer(C_buf,        0, 3 );
        compute_encoder->setBytes(kappa.data(), kappa.size() * sizeof(Real), 4 );
        compute_encoder->setBytes(&c[0][0],                8 * sizeof(Real), 5 );
        compute_encoder->setBytes(&simplex_count,              sizeof(int ), 6 );
        compute_encoder->setBytes(&wave_count,                 sizeof(int ), 7 );

        const NS::Integer max_threads = pipeline->maxTotalThreadsPerThreadgroup();

//        if(block_size != max_threads)
//        {
//            wprint(ClassName()+"::ApplyBoundaryOperators: block_size != max_threads");
//        }
        
        MTL::Size threads_per_threadgroup (max_threads, 1, 1);
        MTL::Size threadgroups_per_grid  (
            n_rounded / static_cast<Int>(threads_per_threadgroup.width), 1, 1
        );
                
//        valprint("simplex_count",simplex_count);
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
        
        blit_command_encoder->synchronizeResource(C_buf);
        blit_command_encoder->endEncoding();
        
        
        // Execute the command buffer.
        command_buffer->commit();
        command_buffer->waitUntilCompleted();
        
        toc(tag);
        ptoc(tag);
    }
