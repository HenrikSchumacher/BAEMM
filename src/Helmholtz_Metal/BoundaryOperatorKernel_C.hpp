public:

    void BoundaryOperatorKernel_C(
        const WaveNumberContainer_T  & kappa_,
        const CoefficientContainer_T & c_
    )
    {
        // This is to blame.
        
        zerofy_buffer( B_ptr, int_cast<LInt>(rows_rounded) * int_cast<LInt>(ldB) );
        zerofy_buffer( C_ptr, int_cast<LInt>(rows_rounded) * int_cast<LInt>(ldC) );
        
        ModifiedB();
        ModifiedC();
        
        std::string name ("BoundaryOperatorKernel_C");
        
        if( !B_loaded )
        {
            wprint(ClassName()+"::BoundaryOperatorKernel_C: No values loaded into B. doing nothing.");
        }
        
        std::string tag = ClassName()+"::"+name+"(...,"+ToString(block_size)+","+ToString(wave_chunk_size)
            + ",{"+ToString(Re_single_layer)+","+ToString(Im_single_layer)
            +"},{"+ToString(Re_double_layer)+","+ToString(Im_double_layer)
            +"},{"+ToString(Re_adjdbl_layer)+","+ToString(Im_adjdbl_layer)
            +"})";
        
        ptic(tag);
//        tic(tag);
        
        NS::SharedPtr<MTL::ComputePipelineState> pipeline = GetPipelineState(
            name,
            std::string(
#include "BoundaryOperatorKernel_C.metal"
            ),
            {"int","int","bool","bool","bool","bool","bool","bool"},
            {"block_size","wave_chunk_size",
                "Re_single_layer","Im_single_layer",
                "Re_double_layer","Im_double_layer",
                "Re_adjdbl_layer","Im_adjdbl_layer"
            },{
              ToString(block_size),
              ToString(wave_chunk_size),
              ToString(Re_single_layer), ToString(Im_single_layer),
              ToString(Re_double_layer), ToString(Im_double_layer),
              ToString(Re_adjdbl_layer), ToString(Im_adjdbl_layer)
            }
        );
        
        assert( pipeline.get() != nullptr );
        
        // Now we can proceed to set up the MTL::CommandBuffer.

        // Create a command buffer to hold commands.
        NS::SharedPtr<MTL::CommandBuffer> command_buffer = NS::TransferPtr(command_queue->commandBuffer());
        assert( command_buffer.get() != nullptr );

        // Create an encoder that translates our command to something the
        // device understands
        NS::SharedPtr<MTL::ComputeCommandEncoder> compute_encoder = NS::TransferPtr(command_buffer->computeCommandEncoder());
        assert( compute_encoder.get() != nullptr );

        // Encode the pipeline state object and its parameters.
        compute_encoder->setComputePipelineState( pipeline.get() );
        
        const Int wave_count_ = kappa_.Size() * wave_chunk_size;
        
        // Place data in encoder
        compute_encoder->setBuffer(mid_points.get(), 0, 0 );
        compute_encoder->setBuffer(normals.get()   , 0, 1 );
        compute_encoder->setBuffer(B_buf.get(),      0, 2 );
        compute_encoder->setBuffer(C_buf.get(),      0, 3 );
        
        compute_encoder->setBytes(kappa.data(),   kappa.Size() * sizeof(Real), 4 );
        compute_encoder->setBytes(c.data(),        c.Size() * sizeof(Complex), 5 );
        compute_encoder->setBytes(&simplex_count,                 sizeof(int), 6 );
        compute_encoder->setBytes(&wave_count_,                   sizeof(int), 7 );

        const NS::Integer max_threads = pipeline->maxTotalThreadsPerThreadgroup();

//        if(block_size != max_threads)
//        {
//            wprint(ClassName()+"::ApplyBoundaryOperators: block_size != max_threads");
//        }
        
        MTL::Size threads_per_threadgroup (max_threads, 1, 1);
        MTL::Size threadgroups_per_grid  (
            rows_rounded / static_cast<Int>(threads_per_threadgroup.width), 1, 1
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
        
        blit_command_encoder->synchronizeResource(C_buf.get());
        blit_command_encoder->endEncoding();
        
        
        // Execute the command buffer.
        command_buffer->commit();
        command_buffer->waitUntilCompleted();
        
//        toc(tag);
        ptoc(tag);
    }
