#pragma once

#include "Tensors/Tensors.hpp"

namespace BAEMM
{
    using namespace Tools;
    using namespace Tensors;

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
}

#include "src/Helmholtz_CPU.hpp"

#include "src/Helmholtz_Metal.hpp"
