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
}

#include "src/Helmholtz_CPU.hpp"

#include "src/Helmholtz_Metal.hpp"
