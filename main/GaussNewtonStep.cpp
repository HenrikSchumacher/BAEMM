#include <sys/types.h>
#include <complex>
#include <iostream>
#include <fstream> 

using uint = unsigned int;

#include "../Helmholtz_OpenCL.hpp"
#include "../ReadWrite/ReadFiles.hpp"
#include "../ReadWrite/WriteFiles.hpp"

using namespace Tools;
using namespace Tensors;

using Int = int;
using Real = float;
using Complex = std::complex<Real>;

int main()
{
    // This routine reads out the data from the files data.txt (which contains data as specified in WriteFiles), simplices.bin, meas_direction.bin and coords.bin.
    // The direction of the derivative is read from B.bin, then the Gauss Newton step is calculated.

    Int vertex_count, simplex_count, meas_count;
    Int wave_chunk_count, wave_chunk_size;
    Int GPU_device;

    Tensor2<Int,Int>        simplices;
    Tensor2<Real,Int>       meas_directions;
    Tensor2<Real,Int>       incident_directions;
    Tensor1<Real,Int>       kappa;
    Tensor2<Real,Int>       coords;
    Tensor2<Real,Int>       B_in;

    ReadFixes(vertex_count, simplex_count, meas_count, wave_chunk_count, wave_chunk_size, GPU_device, simplices, meas_directions, incident_directions, kappa);

    ReadCoordinates(vertex_count, coords);

    Int wave_count = wave_chunk_count * wave_chunk_size;

    Int dim = 3;
    ReadInOut(vertex_count, dim, B_in);

    BAEMM::Helmholtz_OpenCL H (
        coords.data(),    vertex_count,
        simplices.data(), simplex_count, 
        meas_directions.data(), meas_count, GPU_device,
        int_cast<Int>(16)
        );

    H.UseDiagonal(true);
    H.SetBlockSize(64);

    Tensor2<Real,Int>    B_out(  vertex_count, dim  );

    Real cg_tol = static_cast<Real>(0.00001);
    Real gmres_tol = static_cast<Real>(0.0005);

    switch (wave_count)
    {
        case 8:
        {
            H.GaussNewtonStep<Int,Real,Complex,8>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                        B_in.data(), B_out.data(), cg_tol, gmres_tol);
            break;
        }
        case 16:
        {
            H.GaussNewtonStep<Int,Real,Complex,16>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                        B_in.data(), B_out.data(), cg_tol, gmres_tol);
            break;
        }
        case 32:
        {
            H.GaussNewtonStep<Int,Real,Complex,32>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                        B_in.data(), B_out.data(), cg_tol, gmres_tol);
            break;
        }
        case 64:
        {
            H.GaussNewtonStep<Int,Real,Complex,64>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                        B_in.data(), B_out.data(), cg_tol, gmres_tol);
            break;
        }
    }
    
    WriteInOut(vertex_count, dim, B_out);

    return 0;
}