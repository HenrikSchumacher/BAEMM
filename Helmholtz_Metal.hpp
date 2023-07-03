#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <complex>
#include "Repulsor/Repulsor.hpp"
#include "Repulsor/Tensors/Tensors.hpp"

// TODO: Priority 1:
// TODO: Debug wrapper
// TODO: diagonal part of single layer boundary operator
// TODO: adjust block_size
// TODO: Are the i, j >= n treated correctly?
// TODO: Hypersingular operator -> local curl operators?
// TODO: Update vertex-coordinates and simplices without discarding compiled functions

// TODO: Priority 2:
// TODO: single and double layer potential operator for far field.
// TODO: evaluate incoming waves on surface -> Dirichlet and Neumann operators.
// TODO: Manage many waves and many wave directions.

// TODO: Priority 3:
// TODO: Calderon preconditioner ->local curl operators.
// TODO: GMRES on GPU

// DONE: averaging operator
// DONE: internal management of MTL::Buffer (round_up, copy, etc.)
// DONE: mass matrix
// DONE: wrapper

namespace BAEMM
{
    using namespace Tools;
    using namespace Tensors;
    using namespace Repulsor;
    
    //https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/index.html#//apple_ref/doc/uid/TP40016642-CH27-SW1
    

    class Helmholtz_Metal
    {
#include "src/Helmholtz_Common/Definitions.hpp"
            
#include "src/Helmholtz_Common/MemberVariables.hpp"
        
        NS::SharedPtr<MTL::Device> device;
        
        std::map<std::string, NS::SharedPtr<MTL::ComputePipelineState>> pipelines;
        
        NS::SharedPtr<MTL::CommandQueue> command_queue;
        
        NS::SharedPtr<MTL::Buffer> areas;
        NS::SharedPtr<MTL::Buffer> mid_points;
        NS::SharedPtr<MTL::Buffer> normals;
        NS::SharedPtr<MTL::Buffer> single_diag;
        NS::SharedPtr<MTL::Buffer> tri_coords;
        
        NS::SharedPtr<MTL::Buffer> B_buf;
        NS::SharedPtr<MTL::Buffer> C_buf;
        
    public:
        using NS::StringEncoding::UTF8StringEncoding;
        static constexpr auto Managed = MTL::ResourceStorageModeManaged;
        
        template<typename ExtReal,typename ExtInt>
        Helmholtz_Metal(
            NS::SharedPtr<MTL::Device> & device_,
            ptr<ExtReal> vertex_coords_, ExtInt vertex_count_,
            ptr<ExtInt>  triangles_    , ExtInt simplex_count_,
            int CPU_thread_count_
        )
        :   CPU_thread_count ( CPU_thread_count_                    )
        ,   vertex_count     ( int_cast<Int>(vertex_count_)         )
        ,   simplex_count    ( int_cast<Int>(simplex_count_)        )
        ,   vertex_coords    ( vertex_coords_, vertex_count,  3     )
        ,   triangles        ( triangles_,     simplex_count, 3     )
        ,   device           ( device_                              )
        {
//            tic(ClassName());
            
            const uint size     =     simplex_count * sizeof(Real);
            const uint size4    = 4 * simplex_count * sizeof(Real);
            
            areas               = NS::TransferPtr(device->newBuffer(size,    Managed));
            mid_points          = NS::TransferPtr(device->newBuffer(size4,   Managed));
            normals             = NS::TransferPtr(device->newBuffer(size4,   Managed));
            single_diag         = NS::TransferPtr(device->newBuffer(size,    Managed));
            tri_coords          = NS::TransferPtr(device->newBuffer(3*size4, Managed));
            
            areas_ptr           = reinterpret_cast<Real*>(      areas->contents());
            mid_points_ptr      = reinterpret_cast<Real*>( mid_points->contents());
            normals_ptr         = reinterpret_cast<Real*>(    normals->contents());
            single_diag_ptr     = reinterpret_cast<Real*>(single_diag->contents());
            tri_coords_ptr      = reinterpret_cast<Real*>( tri_coords->contents());
            
            command_queue = NS::TransferPtr(device->newCommandQueue());
            
            if( command_queue.get() == nullptr )
            {
                eprint(ClassName()+"::Initialize_Metal: Failed to find the command queue." );
                return;
            }
            
            Initialize();
            
                  areas->didModifyRange({0,areas->length()});
             mid_points->didModifyRange({0,mid_points->length()});
                normals->didModifyRange({0,normals->length()});
            single_diag->didModifyRange({0,single_diag->length()});
            
//            toc(ClassName());
        }
        
        ~Helmholtz_Metal()
        {
//            print("~Helmholtz_Metal()");
            
            pipelines = std::map<std::string, NS::SharedPtr<MTL::ComputePipelineState>> ();
        }
        
#include "src/Helmholtz_Common/Initialize.hpp"

#include "src/Helmholtz_Common/GetSetters.hpp"
        
#include "src/Helmholtz_Common/LoadParameters.hpp"
        
#include "src/Helmholtz_Common/LoadParameters3.hpp"
        
#include "src/Helmholtz_Common/InputOutput.hpp"
        
#include "src/Helmholtz_Metal/GetPipelineState.hpp"

#include "src/Helmholtz_Metal/RequireBuffers.hpp"
        
#include "src/Helmholtz_Common/ApplyOperators.hpp"
        
#include "src/Helmholtz_Common/ApplyBoundaryOperators.hpp"
        
#include "src/Helmholtz_Common/ApplySingleLayerDiagonal.hpp"
        
#include "src/Helmholtz_Metal/BoundaryOperatorKernel_C.hpp"
        
//#include "src/Helmholtz_Metal/BoundaryOperatorKernel_ReIm.hpp
    
    public:
        
        const float * VertexCoordinates() const
        {
            return vertex_coords.data();
        }

        const float * TriangleCoordinates() const
        {
            return reinterpret_cast<const float *>(tri_coords->contents());
        }
        
    public:
        
        std::string ClassName() const
        {
            return "Helmholtz_Metal";
        }
        
    };
        
} // namespace BAEMM
