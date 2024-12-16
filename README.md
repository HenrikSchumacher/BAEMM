# BÄMM!
BÄMM! - a brute force approach for the boundary element method

This is a GPU-accelerated (by OpenCL) first-order boundary-element solver for the three-dimensional acoustic Helmholtz problem. In the sake of simplicity, the library only features piecewise linear, continuous boundary functions.

Helmholtz_OpenCL allows to apply the mass matrix, the single-layer boundary operator, the double-layer boundary operator and its adjoint to a piecewise linear, continuous function on a mesh, i.e. a function defined by its values at the vertices.

The input can be a block of vectors of size $2^p\leq 64$. This block can itself be divided into uniform blocks handling distinct wavenumbers. For example, one might choose 2 different wavenumbers and 16 right hand sides per wavenumber for a total of 32 right hand sides.

Additionally, the far field operators, the boundary _potential_ operators, incident plane/radial waves (and their normal derivatives) and the herglotz wave function (and its normal derivatives) are implemented.
For the case of the Helmholtz problem with _homogeneous Dirichlet boundary conditions_, the boundary-to far field map, its directional derivative and the $L^2$-adjoint of its directional derivative are implemented in FarField.hpp. To solve the Helmholtz problem, we use the mixed indirect potential approach.
The routine GaussNewtonSolve calculates (M + DF*DF)^{-1} for M to be a scaled metric operator.
For examples look into the respective Example-folders.

A detailed documentation can, for instance, be generated with doxygen.

# Download/Installation

Either clone with

    git clone --recurse-submodules

or clone as usual and then run the following to connect all submodules to their repos.

    git submodule update --init --recursive

_BAEMM_ is header-only, so you do not have to precompile anything and thus you also find no makefile here. Just include

    #include "Helmholtz_OpenCL.hpp"

and tell your compiler where to find it. You also need to tell your compiler where to find the OpenCL implementation and link it with

    -lOpenCL
    
when using gcc, or

    -framework OpenCL
    
when using clang.

As _BÄMM_ depends on the _Repulsor_ library (https://github.com/HenrikSchumacher/Repulsor.git), we refer to its documentation for further linkages and compiler options needed.
