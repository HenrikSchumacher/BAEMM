#pragma once

#include <complex>
#include "Tensors/Tensors.hpp"

namespace BAEMM
{
    using namespace Tools;
    using namespace Tensors;
    
    class Helmholtz_CPU
    {
#include "src/Helmholtz_Common/Definitions.hpp"
        
    public:
        
        Helmholtz_CPU() = delete;
        
        
        template<typename ExtReal,typename ExtInt>
        Helmholtz_CPU(
            ptr<ExtReal> vertex_coords_, ExtInt vertex_count_,
            ptr<ExtInt>  triangles_    , ExtInt simplex_count_,
            Int OMP_thread_count_
        )
        :   OMP_thread_count ( OMP_thread_count_                 )
        ,   vertex_count     ( vertex_count_                     )
        ,   simplex_count    ( simplex_count_                    )
        ,   vertex_coords    ( vertex_coords_, vertex_count_,  3 )
        ,   triangles        ( triangles_,     simplex_count_, 3 )
        {
            tic(ClassName());
            
            Initialize_CPU();
            
            Initialize();
            
            toc(ClassName());
        }
        
        ~Helmholtz_CPU() = default;

        
#include "src/Helmholtz_Common/MemberVariables.hpp"
        
        Tensor1<Real,Int> areas;
        Tensor2<Real,Int> mid_points;
        Tensor2<Real,Int> normals;
        
        Tensor2<Complex,Int> B_buf;
        Tensor2<Complex,Int> C_buf;
        
#include "src/Helmholtz_CPU/Initialize_CPU.hpp"
        
#include "src/Helmholtz_Common/Initialize.hpp"
            
#include "src/Helmholtz_Common/InputOutput.hpp"
        
#include "src/Helmholtz_Common/GetSetters.hpp"
        
#include "src/Helmholtz_CPU/RequireBuffers.hpp"

#include "src/Helmholtz_Common/ApplyBoundaryOperators.hpp"
        
#include "src/Helmholtz_CPU/BoundaryOperatorKernel_C.hpp"
        
    public:
        
        std::string ClassName() const
        {
            return "Helmholtz_CPU";
        }
        
    }; // Helmholtz_CPU
    
} // namespace BAEMM

