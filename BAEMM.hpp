#pragma once

#include "Repulsor/Repulsor.hpp"

namespace BAEMM
{
    using namespace Tools;
    using namespace Tensors;
    using namespace Repulsor;
    
    template<typename Real>
    force_inline void cfma(
        const Real   Re_a, const Real   Im_a,
        const Real   Re_b, const Real   Im_b,
              Real & Re_c,       Real & Im_c
    )
    {
        Re_c += Re_a * Re_b - Im_a * Im_b;
        Im_c += Re_a * Im_b + Im_a * Re_b;
    }

    template<typename Real>
    force_inline void cmulby(
              Real & Re_a,       Real & Im_a,
        const Real   Re_b, const Real   Im_b
    )
    {
        const Real Re_c = Re_a * Re_b - Im_a * Im_b;
        const Real Im_c = Re_a * Im_b + Im_a * Re_b;

        Re_a = Re_c;
        Im_a = Im_c;
    }
    
//    template<typename Real>
//    force_inline void cfma(
//        const Real   Re_a, const Real   Im_a,
//        const Real   Re_b, const Real   Im_b,
//              Real & Re_c,       Real & Im_c
//    )
//    {
//        const Real prod1 = Re_a * Re_b;
//        const Real prod2 = Im_a * Im_b;
//        const Real prod3 = ( Re_a + Im_a ) * ( Re_b + Im_b );
//
//        Re_c += prod1 - prod2;
//        Im_c += prod3 - (prod1 + prod2);
//    }
    
//    template<typename Real>
//    force_inline void cmulby(
//              Real & Re_a,       Real & Im_a,
//        const Real   Re_b, const Real   Im_b
//    )
//    {
//        const Real prod1 = Re_a * Re_b;
//        const Real prod2 = Im_a * Im_b;
//        const Real prod3 = ( Re_a + Im_a ) * ( Re_b + Im_b );
//
//        Re_a = prod1 - prod2;
//        Im_a = prod3 - (prod1 + prod2);
//    }
    
    
    template<typename Real_1, typename Real_2, typename Int>
    void GetDifference(
        const Tensor2<Real_1,Int>               & Re_X,
        const Tensor2<Real_1,Int>               & Im_X,
        const Tensor2<std::complex<Real_2>,Int> & Y,
              Tensor2<std::complex<Real_2>,Int> & Z
    )
    {
        const Int m = Z.Dimension(0);
        const Int n = Z.Dimension(1);
        
        for( Int i = 0; i < m; ++i )
        {
            for( Int j = 0; j < n; ++j )
            {
                Z(i,j) = std::complex<Real_2>( Re_X(i,j), Im_X(i,j) ) - Y(i,j);
            }
        }
    }
}


#include "src/HelmholtzOperator_SoA.hpp"

#include "src/HelmholtzOperator_AoS.hpp"

#include "src/HelmholtzOperator_Metal.hpp"
