#pragma once

#include <complex>
#include <cblas.h>
#include <lapack.h>

#include "Repulsor/Repulsor.hpp"

//#include <cblas.h>
#include "Repulsor/Tensors/GMRES.hpp"
#include "Repulsor/Tensors/ConjugateGradient.hpp"

namespace BAEMM
{
    using namespace Tools;
    using namespace Tensors;
    using namespace Repulsor;
    
    class Helmholtz_CPU
    {
#include "src/Helmholtz_Common/Definitions.hpp"
        
    public:
        
        Helmholtz_CPU() = delete;
        
        
        template<typename ExtReal,typename ExtInt>
        Helmholtz_CPU(
            ptr<ExtReal> vertex_coords_, ExtInt vertex_count_,
            ptr<ExtInt>  triangles_    , ExtInt simplex_count_,
            Int CPU_thread_count_
        )
        :   CPU_thread_count ( CPU_thread_count_                )
        ,   vertex_count     ( int_cast<Int>(vertex_count_)     )
        ,   simplex_count    ( int_cast<Int>(simplex_count_)    )
        ,   vertex_coords    ( vertex_coords_, vertex_count,  3 )
        ,   triangles        ( triangles_,     simplex_count, 3 )
        {
//            tic(ClassName());
            
            areas           = Tensor1<Real,Int>( simplex_count    );
            mid_points      = Tensor2<Real,Int>( simplex_count, 4 );
            normals         = Tensor2<Real,Int>( simplex_count, 4 );
            single_diag     = Tensor1<Real,Int>( simplex_count    );
            tri_coords      = Tensor3<Real,Int>( simplex_count, 3, 4 );
            
            areas_ptr       = areas.data();
            mid_points_ptr  = mid_points.data();
            normals_ptr     = normals.data();
            single_diag_ptr = single_diag.data();
            tri_coords_ptr  = tri_coords.data();
            
            Initialize();
            
//            toc(ClassName());
        }
        
        ~Helmholtz_CPU() = default;

        
#include "src/Helmholtz_Common/MemberVariables.hpp"
        
        Tensor1<Real,Int> areas;
        Tensor2<Real,Int> mid_points;
        Tensor2<Real,Int> normals;
        Tensor1<Real,Int> single_diag;
        
        Tensor3<Real,Int> tri_coords;
        
        Tensor2<Complex,Int> B_buf;
        Tensor2<Complex,Int> C_buf;
public:        
#include "src/Helmholtz_Common/Initialize.hpp"
            
#include "src/Helmholtz_Common/InputOutput.hpp"
        
#include "src/Helmholtz_Common/GetSetters.hpp"
        
#include "src/Helmholtz_Common/LoadParameters.hpp"
        
#include "src/Helmholtz_Common/LoadParameters3.hpp"
        
#include "src/Helmholtz_CPU/RequireBuffers.hpp"

#include "src/Helmholtz_Common/ApplyOperators.hpp"
        
#include "src/Helmholtz_Common/ApplyBoundaryOperators.hpp"
        
#include "src/Helmholtz_Common/ApplySingleLayerDiagonal.hpp"
        
#include "src/Helmholtz_CPU/BoundaryOperatorKernel_C.hpp"
        


//#include "src/Helmholtz_CPU/SingleLayer.hpp"

    public:
        
        const float * VertexCoordinates() const
        {
            return vertex_coords.data();
        }

        const float * TriangleCoordinates() const
        {
            return tri_coords.data();
        }
        
    public:
        
        std::string ClassName() const
        {
            return "Helmholtz_CPU";
        }
        
    }; // Helmholtz_CPU
    
} // namespace BAEMM

