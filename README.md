# BÄMM!
BÄMM! - a brute force approach for the boundary element method

This is a first-order boundary-element solver for the three-dimensional acoustic Helmholtz problem. In the sake of simplicity, the library only features piecewise linear, continuous boundary functions.

Helmholtz_OpenCL, Helmholtz_Metal and Helmholtz_CPU allow to apply the mass matrix, the single-layer boundary operator, the double-layer boundary operator and its adjoint to a piecewise linear, continuous function on a mesh, i.e. a function defined by its valuces at the vertices. For examples look into the respective Example*-folders.

Additionally, in Helmholtz_OpenCL the far field map, the near field map and the herglotz wave function are implemented. This can be adapted to the other libraries as needed.
The far field operator, its directional derivative and the $L^2$-adjoint of its directional derivative are implemented in FarField.hpp (as pointed out above, at this stage only available for Helmholtz_OpenCL). To solve the Helmholtz problem, we use the mixed indirect potential approach. The routine GaussNewton calculates (M + DF*DF)^{-1} for M to be a scaled metric operator.

For a detailed documentation open documentation/annotated in your browser.

# Download/Installation

Either clone with

    git clone --recurse-submodules

or clone as usual and then run the following to connect all submodules to their repos.

    git submodule update --init --recursive

_BAEMM_ is header-only, so you do not have to precompile anything and thus you also find no makefile here. Just include

    #include "Helmholtz_OpenCL.hpp"
or
    #include "Helmholtz_Metal.hpp"
or
    #include "Helmholtz_CPU.hpp"

and tell your compiler where to find it. For the former ones you also need to link the respective libraries to use OpenCL or Metal.

As _BÄMM_ depends on the _Repulsor_ library (https://github.com/HenrikSchumacher/Repulsor.git), we refer to its documentation for further linkages and compiler options needed.