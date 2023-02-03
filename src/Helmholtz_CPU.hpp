namespace BAEMM
{
    class Helmholtz_CPU
    {
#include "Helmholtz_Common/Definitions.hpp"
        
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

        
#include "Helmholtz_Common/MemberVariables.hpp"
        
        Tensor1<Real,Int> areas;
        Tensor1<Real,Int> mid_points;
        Tensor1<Real,Int> normals;
        
        Tensor2<Complex,Int> B_buf;
        Tensor2<Complex,Int> C_buf;
        
#include "Helmholtz_CPU/Initialize_CPU.hpp"
        
#include "Helmholtz_Common/Initialize.hpp"
            
#include "Helmholtz_Common/InputOutput.hpp"
        
#include "Helmholtz_Common/GetSetters.hpp"
        
#include "Helmholtz_CPU/RequireBuffers.hpp"

#include "Helmholtz_Common/ApplyBoundaryOperators.hpp"
        
#include "Helmholtz_CPU/BoundaryOperatorKernel_C.hpp"
        
    public:
        
        std::string ClassName() const
        {
            return "Helmholtz_CPU";
        }
        
    }; // Helmholtz_CPU
    
} // namespace BAEMM

