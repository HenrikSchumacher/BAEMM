#include <sys/types.h>
#include <complex>

#include "../ReadWrite/ReadFiles.hpp"
#include "../ReadWrite/WriteFiles.hpp"
#include "../Helmholtz_OpenCL.hpp"

using namespace Tools;
using namespace Tensors;

using Int = int;
using Real = float;
using Complex = std::complex(Real);

int main()
{
    Int vertex_count, simplex_count, meas_count;
    Int wave_chunk_count, wave_chunk_size;

    Tensor2<Int,Int>        simplices;
    Tensor2<Real,Int>       meas_directions;
    Tensor2<Real,Int>       incident_directions;
    Tensor1<Real,Int>       kappa;
    Tensor2<Real,Int>       coords;
    Tensor2<Complex,Int>    B_out;

    ReadFixes(vertex_count, simplex_count, meas_count, wave_chunk_count, wave_chunk_size, simplices, meas_directions, incindet_directions, kappa);

    ReadCoordinates(vertex_count, coords);

    const Int wave_count = wave_chunk_count * wave_chunk_size;

    BAEMM::Helmholtz_OpenCL H (
        coords.data(),    coords.Dimension(0),
        simplices.data(), simplices.Dimension(0), 16
        );

    H.UseDiagonal(true);
    H.SetBlockSize(64);

    H.FarField<Int,Real,Complex,wave_count>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                B_out.data(), 0.000001, 0.0001);
    
    WriteInOut(meas_count, wave_count, B_out);

    return 0;
}