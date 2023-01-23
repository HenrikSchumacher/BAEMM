#pragma once

namespace BAEMM
{

    template<class Mesh_T>
    class HelmholtzOperator_Metal
    {
    public:
        
        using Real       = typename Mesh_T::Real;
        using Int        = typename Mesh_T::Int;
        using SReal      = typename Mesh_T::SReal;
        using Complex    = std::complex<SReal>;
        
        using ExtReal    = typename Mesh_T::ExtReal;
//        using ExtComplex = std::complex<ExtReal>;
        
        static constexpr float one   = 1;
        static constexpr float two   = 2;
        static constexpr float three = 3;
        
        static constexpr float half  = one / two;
        static constexpr float third = one / three;
        
    private:
        
        MTL::Device* device;
        
        std::map<std::string, MTL::Function * > functionMap;
        std::map<std::string, MTL::ComputePipelineState * > functionPipelineMap;
        
        MTL::CommandQueue * command_queue;
        
        Mesh_T & M;
        
        const uint n;
        
        Tensor2<float,uint> vertex_coords;
        Tensor2<uint ,uint> triangles;
        
        MTL::Buffer * areas;
        MTL::Buffer * mid_points;
        MTL::Buffer * normals;
        
    public:
        
        HelmholtzOperator_Metal( MTL::Device* device_, Mesh_T & M_)
        :   device( device_ )
        ,   M             ( M_ )
        ,   n             ( M.SimplexCount() )
        ,   vertex_coords ( M.VertexCoordinates().data(), M.VertexCount(),  3   )
        ,   triangles     ( M.Simplices().data(),         M.SimplexCount(), 3   )
        {
            const uint size  =     n * sizeof(float);
            const uint size3 = 3 * n * sizeof(float);
            
            areas      = device->newBuffer(size,  MTL::ResourceStorageModeManaged);
            mid_points = device->newBuffer(size3, MTL::ResourceStorageModeManaged);
            normals    = device->newBuffer(size3, MTL::ResourceStorageModeManaged);
            
            mut<float> areas_      = static_cast<float *>(     areas->contents());
            mut<float> mid_points_ = static_cast<float *>(mid_points->contents());
            mut<float> normals_    = static_cast<float *>(   normals->contents());
            
            Tiny::Vector<3,float,uint> x;
            Tiny::Vector<3,float,uint> y;
            Tiny::Vector<3,float,uint> z;
            
            Tiny::Vector<3,float,uint> nu;
            
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

                const SReal a = nu.Norm();
                areas_[i] = a;

                nu /= a;

                nu.Write( &normals_[3*i] );
            }
            
                 areas->didModifyRange({0,size });
            mid_points->didModifyRange({0,size3});
               normals->didModifyRange({0,size3});
            
            NS::Error *error = nullptr;
            
            // Load the shader files with a .metallib file extension in the project
            auto filepath = NS::String::string(
                "/Users/Henrik/github/BAEMM/src/Helmholtz.metallib",
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
            
            for( NS::UInteger i = 0; i < function_names->count(); ++i )
            {
                auto name_nsstring = function_names->object(i)->description();
                auto name_utf8 = name_nsstring->utf8String();
                
                valprint("kernel found",name_utf8);

                // Load function into a map
                functionMap[name_utf8] = (lib->newFunction(name_nsstring));

                // Create pipeline from function
                functionPipelineMap[name_utf8] =
                    device->newComputePipelineState(functionMap[name_utf8], &error);

                if( functionPipelineMap[name_utf8] == nullptr )
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
        
        ~HelmholtzOperator_Metal() = default;

        
        
    //    void Copy(
    //        MTL::Buffer * from,
    //        MTL::Buffer * to,
    //        const NS::UInteger size
    //    )
    //    {
    //        tic("Copy");
    //
    //        // Create a command buffer to hold commands.
    //        MTL::CommandBuffer * command_buffer = command_queue->commandBuffer();
    //        assert( command_buffer != nullptr );
    //
    //
    //        // Encode a blit pass to copy data from the source buffer to the private buffer.
    //        MTL::BlitCommandEncoder * blit_command_encoder = command_buffer->blitCommandEncoder();
    //        assert( blit_command_encoder != nullptr );
    //        blit_command_encoder->copyFromBuffer(from, 0, to, 0, size);
    //
    //        blit_command_encoder->endEncoding();
    //
    //
    //        command_buffer->addCompletedHandler( ^void( MTL::CommandBuffer * pCmd ){});
    //
    //        // Private buffer is populated.
    //
    //        command_buffer->commit();
    //
    ////        command_buffer->waitUntilCompleted();
    //
    //        toc("Copy");
    //    }
        
            
        
        
        template<uint chunk_size, uint n_waves>
        void Multiply(
            const MTL::Buffer * Re_Y,
            const MTL::Buffer * Im_Y,
                  MTL::Buffer * Re_X,
                  MTL::Buffer * Im_X,
            const float kappa,
            const float kappa_step
        )
        {   
            tic(ClassName()+"::Multiply<"+ToString(n_waves)+">");
            
            // Create a command buffer to hold commands.
            MTL::CommandBuffer* command_buffer = command_queue->commandBuffer();
            assert( command_buffer != nullptr );

            // Create an encoder that translates our command to something the
            // device understands
            MTL::ComputeCommandEncoder* compute_encoder = command_buffer->computeCommandEncoder();
            assert( compute_encoder != nullptr );

            std::string fun_name = "Helmholtz_Multiply_"+ToString(chunk_size)+"_"+ToString(n_waves);
            dump(fun_name);
            
            MTL::ComputePipelineState * pipeline = functionPipelineMap[fun_name];
            assert( pipeline != nullptr );

            // Encode the pipeline state object and its parameters.
            compute_encoder->setComputePipelineState( pipeline );

            // Place data in encoder
            compute_encoder->setBuffer(mid_points, 0 ,0 );
            compute_encoder->setBuffer(Re_Y, 0, 1);
            compute_encoder->setBuffer(Im_Y, 0, 2);
            compute_encoder->setBuffer(Re_X, 0, 3);
            compute_encoder->setBuffer(Im_X, 0, 4);
            compute_encoder->setBytes(&kappa,      sizeof(float),       5);
            compute_encoder->setBytes(&kappa_step, sizeof(float),       6);
            compute_encoder->setBytes(&n,          sizeof(NS::Integer), 7);

            const NS::Integer max_threads = pipeline->maxTotalThreadsPerThreadgroup();
            
            dump(max_threads);
            
            MTL::Size threads_per_threadgroup = MTL::Size(max_threads, 1, 1);
            MTL::Size threadgroups_per_grid   = MTL::Size(
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
            
            toc(ClassName()+"::Multiply<"+ToString(n_waves)+">");
        }
        
    public:
        
        std::string ClassName() const
        {
            return "HelmholtzOperator_Metal<"+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
        
    };
        
} // namespace BAEMM
