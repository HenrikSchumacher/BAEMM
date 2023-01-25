#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

namespace BAEMM
{
    //https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/index.html#//apple_ref/doc/uid/TP40016642-CH27-SW1
    

    class Helmholtz_Metal
    {
    public:
        
        using Int        = uint;
        using Real       = float;
        using Complex    = std::complex<Real>;
        
        using NS::StringEncoding::UTF8StringEncoding;
        
        static constexpr Real zero  = 0;
        static constexpr Real one   = 1;
        static constexpr Real two   = 2;
        static constexpr Real three = 3;
        
        static constexpr Real half  = one / two;
        static constexpr Real third = one / three;
        
        static constexpr Real pi      = M_PI;
        static constexpr Real two_pi  = two * pi;
        static constexpr Real four_pi = two * two_pi;
        
//        static constexpr Real one_over_two_pi  = one / two_pi;
        static constexpr Real one_over_four_pi = one / four_pi;
            
        
    private:
        
        MTL::Device* device;

        std::map<std::string, MTL::ComputePipelineState * > pipelines;
        
        MTL::CommandQueue * command_queue;
        
        const Int m;
        const Int n;
        
        Tensor2<Real,Int> vertex_coords;
        Tensor2<Int ,Int> triangles;
        
        MTL::Buffer * areas;
        MTL::Buffer * mid_points;
        MTL::Buffer * normals;
        
    public:
        
        template<typename ExtReal,typename ExtInt>
        Helmholtz_Metal(
            MTL::Device* device_,
            ptr<ExtReal> vertex_coords_, ExtInt vertex_count_,
            ptr<ExtInt>  triangles_    , ExtInt simplex_count_
        )
        :   device( device_ )
        ,   m             ( vertex_count_ )
        ,   n             ( simplex_count_ )
        ,   vertex_coords ( vertex_coords_, vertex_count_,  3 )
        ,   triangles     ( triangles_,     simplex_count_, 3 )
        {
            tic(ClassName());
            const Int size  =     n * sizeof(Real);
            const Int size3 = 3 * n * sizeof(Real);
            
            areas      = device->newBuffer(size,  MTL::ResourceStorageModeManaged);
            mid_points = device->newBuffer(size3, MTL::ResourceStorageModeManaged);
            normals    = device->newBuffer(size3, MTL::ResourceStorageModeManaged);
            
            mut<Real> areas_      = static_cast<Real *>(     areas->contents());
            mut<Real> mid_points_ = static_cast<Real *>(mid_points->contents());
            mut<Real> normals_    = static_cast<Real *>(   normals->contents());
            
            Tiny::Vector<3,Real,uint> x;
            Tiny::Vector<3,Real,uint> y;
            Tiny::Vector<3,Real,uint> z;
            
            Tiny::Vector<3,Real,uint> nu;
            
            for( uint i = 0; i < n; ++i )
            {
                x.Read( vertex_coords.data(triangles(i,0)) );
                y.Read( vertex_coords.data(triangles(i,1)) );
                z.Read( vertex_coords.data(triangles(i,2)) );

                mid_points_[3*i+0] = third * ( x[0] + y[0] + z[0] );
                mid_points_[3*i+1] = third * ( x[1] + y[1] + z[1] );
                mid_points_[3*i+2] = third * ( x[2] + y[2] + z[2] );
                
                y -= x;
                z -= x;

                nu[0] = y[1] * z[2] - y[2] * z[1];
                nu[1] = y[2] * z[0] - y[0] * z[2];
                nu[2] = y[0] * z[1] - y[1] * z[0];

                const Real a = nu.Norm();
                areas_[i] = a;

                nu /= a;

                nu.Write( &normals_[3*i] );
            }
            
                 areas->didModifyRange({0,size });
            mid_points->didModifyRange({0,size3});
               normals->didModifyRange({0,size3});

            command_queue = device->newCommandQueue();
            
            if( command_queue == nullptr )
            {
                std::cout << "Failed to find the command queue." << std::endl;
                return;
            }
            
            toc(ClassName());
        }
        
        ~Helmholtz_Metal() = default;

    private:
        
        void CreatePipelineState(
            const std::string & fun_name,       // name of function in code string
            const std::string & fun_fullname,   // name in std::map Pipelines
            const std::string & code,           // string of actual Metal code
            const std::string * param_types,    // types of compile-time parameters (converted to string)
            const std::string * param_names,    // name of compile-time parameters
            const std::string * param_vals,     // values of compile-time parameters (converted to string)
            uint  param_count                   // number of compile-time parameters
        )
        {
            tic("CreatePipeline(" + fun_fullname + ")");
            
            std::stringstream full_code;
            
            // Create compile-time constant. Will be prependend to code string.
            for( uint i = 0; i < param_count; ++i )
            {
                full_code << "constant constexpr " << param_types[i] << " " << param_names[i] << " = " << param_vals[i] <<";\n";
            }
            
            full_code << code;
            
            NS::String * code_NS_String = NS::String::string(full_code.str().c_str(), UTF8StringEncoding);
            
            NS::Error *error = nullptr;
            
            MTL::Library * lib = device->newLibrary(
                code_NS_String,
                nullptr, // <-- crucial for distinguishing from the function that loads from file
                &error
            );
            
            if( lib == nullptr )
            {
                std::cout << "Failed to compile library from string for function "
                          << fun_fullname << ", error "
                          << error->description()->utf8String() << std::endl;
                std::exit(-1);
            }
            
            bool found = false;
            
            // Go through all functions in the library to find ours.
            for( NS::UInteger i = 0; i < lib->functionNames()->count(); ++i )
            {
                found = true;
                
                auto name_nsstring = lib->functionNames()->object(i)->description();
                
                if( fun_name == name_nsstring->utf8String() )
                {
                    // This MTL::Function object is needed only temporarily.
                    MTL::Function * fun = lib->newFunction(name_nsstring);
    
                    // Create pipeline from function.
                    pipelines[fun_fullname] = device->newComputePipelineState(fun, &error);
    
                    if( pipelines[fun_fullname] == nullptr )
                    {
                        std::cout << "Failed to created pipeline state object for "
                                  << fun_name << ", error "
                                  << error->description()->utf8String() << std::endl;
                        return;
                    }
                }
            }
            
            if( found )
            {
                print(std::string("CreatePipeline: Found Metal kernel ") + fun_name +".");
            }
            else
            {
                eprint(std::string("CreatePipeline: Did not find Metal kernel ") + fun_name +" in source code.");
            }
            
            toc("CreatePipeline(" + fun_fullname + ")");
        }
        
    public:
        
        void Neumann_to_Dirichlet(
            const MTL::Buffer * Re_Y,
            const MTL::Buffer * Im_Y,
                  MTL::Buffer * Re_X,
                  MTL::Buffer * Im_X,
            const float kappa,
            const float kappa_step,
            const uint chunk_size,
            const uint n_waves
        )
        {
            tic(ClassName()+"::Neumann_to_Dirichlet(...,"+ToString(chunk_size)+","+ToString(n_waves)+")");
            
            std::string fun_name = "Helmholtz__Neumann_to_Dirichlet";
            
            const std::string template_types [2] = {"uint","uint"};
            const std::string template_names [2] = {"chunk_size","n_waves"};
            const std::string template_vals  [2] = {ToString(chunk_size),ToString(n_waves)};
            
            std::string fun_fullname = fun_name+"_"+template_vals[0]+"_"+template_vals[1];
            
            if( pipelines.count(fun_fullname) == 0 )
            {
                // Calling this function for the first time; we have to compile it first.
                
                CreatePipelineState(
                    fun_name,
                    fun_fullname,
                    std::string(
#include "Helmholtz.metal"
                    ),
                    template_types,template_names,template_vals,2
                );
            }            
            
            MTL::ComputePipelineState * pipeline = pipelines[fun_fullname];
            
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
            MTL::Size threadgroups_per_grid   (
                (n+threads_per_threadgroup.width-1)/threads_per_threadgroup.width, 1, 1
            );

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
            command_buffer->waitUntilCompleted();
            
            toc(ClassName()+"::Neumann_to_Dirichlet(...,"+ToString(chunk_size)+","+ToString(n_waves)+")");
        }
        
    public:
        
        std::string ClassName() const
        {
            return "Helmholtz_Metal";
        }
        
    };
        
} // namespace BAEMM
