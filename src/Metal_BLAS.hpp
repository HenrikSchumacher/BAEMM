//
//  Metal_BLAS.hpp
//  Repulsion
//
//  Created by Henrik on 05.09.22.
//

#pragma once

#include "Tools.hpp"
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

using namespace Tools;

class MetalBLAS
{
private:
    
    MTL::Device* device;
    
    std::map<std::string, MTL::Function * > functionMap;
    std::map<std::string, MTL::ComputePipelineState * > functionPipelineMap;
    
    MTL::CommandQueue * command_queue;
    
public:
    
    MetalBLAS(MTL::Device* device_)
    :   device( device_ )
    {
        NS::Error *error = nullptr;
        
        // Load the shader files with a .metal file extension in the project
        auto filepath = NS::String::string(
            "/Users/Henrik/github/Repulsion/Example_Metal/BLAS.metallib",
            NS::ASCIIStringEncoding
        );
        
        MTL::Library * lib = device->newLibrary(filepath, &error);
        
        if( lib == nullptr )
        {
            std::cerr << "Failed to load library." << std::endl;
            std::exit(-1);
        }
        
        // Get all function names
        auto function_names = lib->functionNames();
        
        for (NS::UInteger i = 0; i < function_names->count(); i++)
        {
            auto name_nsstring = function_names->object(i)->description();
            auto name_utf8 = name_nsstring->utf8String();
            
//            print(name_utf8);

            // Load function into a map
            functionMap[name_utf8] = (lib->newFunction(name_nsstring));

            // Create pipeline from function
            functionPipelineMap[name_utf8] =
                device->newComputePipelineState(functionMap[name_utf8], &error);

            if (functionPipelineMap[name_utf8] == nullptr)
            {
                std::cout << "Failed to created pipeline state object for "
                          << name_utf8 << ", error "
                          << error->description()->utf8String() << std::endl;
                return;
            }
        }
        
        command_queue = device->newCommandQueue();
        
        if( command_queue == nullptr )
        {
            std::cout << "Failed to find the command queue." << std::endl;
            return;
        }
    }
    
    ~MetalBLAS() = default;

    
    
    void Copy(
        MTL::Buffer * from,
        MTL::Buffer * to,
        const NS::UInteger size
    )
    {
        tic("Copy");
        
        // Create a command buffer to hold commands.
        MTL::CommandBuffer * command_buffer = command_queue->commandBuffer();
        assert( command_buffer != nullptr );

        
        // Encode a blit pass to copy data from the source buffer to the private buffer.
        MTL::BlitCommandEncoder * blit_command_encoder = command_buffer->blitCommandEncoder();
        assert( blit_command_encoder != nullptr );
        blit_command_encoder->copyFromBuffer(from, 0, to, 0, size);

        blit_command_encoder->endEncoding();

        
        command_buffer->addCompletedHandler( ^void( MTL::CommandBuffer * pCmd ){});

        // Private buffer is populated.
        
        command_buffer->commit();
        
//        command_buffer->waitUntilCompleted();
        
        toc("Copy");
    }
    
    void GEMM_GPU_sh_reg_nn(
              MTL::Buffer * A,
              MTL::Buffer * B,
              MTL::Buffer * C,
        const NS::Integer M,
        const NS::Integer N,
        const NS::Integer K
    )
    {
        // Create a command buffer to hold commands.
        MTL::CommandBuffer* command_buffer = command_queue->commandBuffer();
        assert( command_buffer != nullptr );

        // Create an encoder that translates our command to something the
        // device understands
        MTL::ComputeCommandEncoder* compute_encoder = command_buffer->computeCommandEncoder();
        assert( compute_encoder != nullptr );

        MTL::ComputePipelineState * pipeline = functionPipelineMap["gpu_gemm_sh_reg_nn"];
        assert( pipeline != nullptr );

        // Encode the pipeline state object and its parameters.
        compute_encoder->setComputePipelineState( pipeline );

        // Place data in encoder
        compute_encoder->setBuffer(A, 0, 0);
        compute_encoder->setBuffer(B, 0, 1);
        compute_encoder->setBuffer(C, 0, 2);
        compute_encoder->setBytes(&M, sizeof(NS::Integer), 3);
        compute_encoder->setBytes(&N, sizeof(NS::Integer), 4);
        compute_encoder->setBytes(&K, sizeof(NS::Integer), 5);

//        const NS::Integer max_threads = pipeline->maxTotalThreadsPerThreadgroup();
        
        MTL::Size threads_per_threadgroup = MTL::Size(16, 16, 1);
        MTL::Size threadgroups_per_grid   = MTL::Size(
            (M+threads_per_threadgroup.width-1)/threads_per_threadgroup.width,
            (N+threads_per_threadgroup.height-1)/threads_per_threadgroup.height,
            1
        );

//        valprint( "threads_per_threadgroup.width ", threads_per_threadgroup.width );
//        valprint( "threads_per_threadgroup.height", threads_per_threadgroup.height );
//        valprint( "threadgroups_per_grid.width   ", threadgroups_per_grid.width );
//        valprint( "threadgroups_per_grid.height  ", threadgroups_per_grid.height );

        // Encode the compute command.
        compute_encoder->dispatchThreadgroups(threadgroups_per_grid, threads_per_threadgroup);

        // Signal that we have encoded all we want.
        compute_encoder->endEncoding();

        // Execute the command buffer.
        command_buffer->commit();

        command_buffer->waitUntilCompleted();
    }
};
