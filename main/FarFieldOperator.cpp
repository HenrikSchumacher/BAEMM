#include <iostream>
#include <fstream> 
#include <sys/types.h>
#include <complex>

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
    // This routine reads out the data from the files data.txt (which contains data as specified in WriteFiles), simplices.bin, meas_direction.bin and coords.bin,
    // calculates the FarField pattern and writes it to B.bin
    // In the case of a radial wave incident_directions are the point source

    Int vertex_count, simplex_count, meas_count;
    Int wave_chunk_count, wave_chunk_size;
    Int GPU_device;
    std::string wave_type;

    Tensor2<Real,Int>       coords;
    Tensor2<Int,Int>        simplices;
    Tensor2<Real,Int>       meas_directions;
    Tensor2<Real,Int>       incident_directions;
    Tensor1<Real,Int>       kappa;

    ReadFixes(vertex_count, simplex_count, meas_count, wave_chunk_count, wave_chunk_size, GPU_device, wave_type, simplices, meas_directions, incident_directions, kappa);

    ReadCoordinates(vertex_count, coords);

    Int wave_count = wave_chunk_count * wave_chunk_size;

    BAEMM::Helmholtz_OpenCL H (
        coords.data(),    vertex_count,
        simplices.data(), simplex_count, 
        meas_directions.data(), meas_count, 
        GPU_device, int_cast<Int>(16)
        );

    H.UseDiagonal(true);
    H.SetBlockSize(64);

    Tensor2<Complex,Int>    B_out(  meas_count, wave_count  );
    
    Real cg_tol = static_cast<Real>(0.000001);
    Real gmres_tol = static_cast<Real>(0.0001);

    switch (wave_type)
    {
        case "Plane":
        {
            switch (wave_count)
            {
                case 8:
                {
                    H.FarField<8>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                                B_out.data(), BAEMM::Helmholtz_OpenCL::WaveType::Plane, cg_tol, gmres_tol);
                    break;
                }
                case 16:
                {
                    H.FarField<16>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                                B_out.data(), BAEMM::Helmholtz_OpenCL::WaveType::Plane, cg_tol, gmres_tol);
                    break;
                }
                case 32:
                {
                    H.FarField<32>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                                B_out.data(), BAEMM::Helmholtz_OpenCL::WaveType::Plane, cg_tol, gmres_tol);
                    break;
                }
                case 64:
                {
                    H.FarField<64>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                                B_out.data(), BAEMM::Helmholtz_OpenCL::WaveType::Plane, cg_tol, gmres_tol);
                    break;
                }
                default:
                {
                    eprint("Non valid wave count.");
                    break;
                }
            }
        }
        case "Radial":
        {
            switch (wave_count)
            {
                case 1:
                {
                    H.FarField<1>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                                B_out.data(), BAEMM::Helmholtz_OpenCL::WaveType::Radial, cg_tol, gmres_tol);
                    break;
                }
                case 2:
                {
                    H.FarField<2>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                                B_out.data(), BAEMM::Helmholtz_OpenCL::WaveType::Radial, cg_tol, gmres_tol);
                    break;
                }
                case 4:
                {
                    H.FarField<4>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                                B_out.data(), BAEMM::Helmholtz_OpenCL::WaveType::Radial, cg_tol, gmres_tol);
                    break;
                }
                case 8:
                {
                    H.FarField<8>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                                B_out.data(), BAEMM::Helmholtz_OpenCL::WaveType::Radial, cg_tol, gmres_tol);
                    break;
                }
                default:
                {
                    eprint("Non valid wave count.");
                    break;
                }
            }
        }
    }

    WriteInOut(meas_count, wave_count, B_out, "B.bin");

    fstream file("NeumannDataScat.bin");
    
    if( file.good() )
    {
        std::remove("NeumannDataScat.bin");
    }

    file.close();

    return 0;
}