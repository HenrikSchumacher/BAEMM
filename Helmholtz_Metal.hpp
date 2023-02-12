#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include <complex>
#include "Tensors/Tensors.hpp"

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
    
    //https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/index.html#//apple_ref/doc/uid/TP40016642-CH27-SW1
    

    class Helmholtz_Metal
    {
#include "src/Helmholtz_Common/Definitions.hpp"
            
#include "src/Helmholtz_Common/MemberVariables.hpp"
        
        MTL::Device* device;
        
        std::map<std::string, MTL::ComputePipelineState *> pipelines;
        
        MTL::CommandQueue * command_queue;
        
        MTL::Buffer * areas;
        MTL::Buffer * mid_points;
        MTL::Buffer * normals;

        MTL::Buffer * B_buf;
        MTL::Buffer * C_buf;
        
    public:
        
        template<typename ExtReal,typename ExtInt>
        Helmholtz_Metal(
            MTL::Device * device_,
            ptr<ExtReal> vertex_coords_, ExtInt vertex_count_,
            ptr<ExtInt>  triangles_    , ExtInt simplex_count_,
            int OMP_thread_count_
        )
        :   device           ( device_ )
        ,   OMP_thread_count ( OMP_thread_count_ )
        ,   vertex_count     ( vertex_count_ )
        ,   simplex_count    ( simplex_count_ )
        ,   vertex_coords    ( vertex_coords_, vertex_count_,  3 )
        ,   triangles        ( triangles_,     simplex_count_, 3 )
        {
            tic(ClassName());
            
            Initialize_Metal();
            
            Initialize();
            
                 areas->didModifyRange({0,areas->length()});
            mid_points->didModifyRange({0,mid_points->length()});
               normals->didModifyRange({0,normals->length()});
            
            toc(ClassName());
        }
        
        ~Helmholtz_Metal() = default;

#include "src/Helmholtz_Metal/Initialize_Metal.hpp"
        
#include "src/Helmholtz_Common/Initialize.hpp"

#include "src/Helmholtz_Common/GetSetters.hpp"
        
#include "src/Helmholtz_Common/InputOutput.hpp"
        
#include "src/Helmholtz_Metal/GetPipelineState.hpp"

#include "src/Helmholtz_Metal/RequireBuffers.hpp"
        
#include "src/Helmholtz_Common/ApplyBoundaryOperators.hpp"
        
#include "src/Helmholtz_Metal/BoundaryOperatorKernel_C.hpp"
        
//#include "src/Helmholtz_Metal/BoundaryOperatorKernel_ReIm.hpp
        
    public:
        
        std::string ClassName() const
        {
            return "Helmholtz_Metal";
        }
        
    };
        
} // namespace BAEMM
