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
    // The input for the adjoint of the derivative of the FarField operator is read from B.bin, the operator is calculated and again written to B.bin

    Int vertex_count, simplex_count, meas_count;
    Int wave_chunk_count, wave_chunk_size;
    Int GPU_device;

    Tensor2<Int,Int>        simplices;
    Tensor2<Real,Int>       meas_directions;
    Tensor2<Real,Int>       incident_directions;
    Tensor1<Real,Int>       kappa;
    Tensor2<Real,Int>       coords;
    Tensor2<Complex,Int>    B_in;

    ReadFixes(vertex_count, simplex_count, meas_count, wave_chunk_count, wave_chunk_size, GPU_device, simplices, meas_directions, incident_directions, kappa);

    ReadCoordinates(vertex_count, coords);

    Int wave_count = wave_chunk_count * wave_chunk_size;

    ReadInOut(meas_count, wave_count, B_in, "B.bin");
    
    BAEMM::Helmholtz_OpenCL H (
        coords.data(),    vertex_count,
        simplices.data(), simplex_count, 
        meas_directions.data(), meas_count, 
        GPU_device, int_cast<Int>(16)
        );

    H.UseDiagonal(true);
    H.SetBlockSize(64);

    Int dim = 3;

    Tensor2<Real,Int>    B_out(  vertex_count, dim  );

    Real cg_tol = static_cast<Real>(0.00001);
    Real gmres_tol = static_cast<Real>(0.0005);

    Tensor2<Complex,Int> neumann_data_scat;
    Complex* neumann_data_scat_ptr = NULL;

    std::fstream file ("NeumannDataScat.bin");
    
    if( file.good() )
    {
        ReadInOut(vertex_count, wave_count, neumann_data_scat,"NeumannDataScat.bin");
        neumann_data_scat_ptr = neumann_data_scat.data();
    }

    switch (wave_type)
    {
        case "Plane":
        {
            switch (wave_count)
            {
                case 8:
                {
                    H.AdjointDerivative_FF<8>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                                B_in.data(), B_out.data(), &neumann_data_scat_ptr, BAEMM::Helmholtz_OpenCL::WaveType::Plane, cg_tol, gmres_tol);
                    break;
                }
                case 16:
                {
                    H.AdjointDerivative_FF<16>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                                B_in.data(), B_out.data(), &neumann_data_scat_ptr, BAEMM::Helmholtz_OpenCL::WaveType::Plane, cg_tol, gmres_tol);
                    break;
                }
                case 32:
                {
                    H.AdjointDerivative_FF<32>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                                B_in.data(), B_out.data(), &neumann_data_scat_ptr, BAEMM::Helmholtz_OpenCL::WaveType::Plane, cg_tol, gmres_tol);
                    break;
                }
                case 64:
                {
                    H.AdjointDerivative_FF<64>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                                B_in.data(), B_out.data(), &neumann_data_scat_ptr, BAEMM::Helmholtz_OpenCL::WaveType::Plane, cg_tol, gmres_tol);
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
                    H.AdjointDerivative_FF<1>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                                B_in.data(), B_out.data(), &neumann_data_scat_ptr, BAEMM::Helmholtz_OpenCL::WaveType::Radial, cg_tol, gmres_tol);
                    break;
                }
                case 2:
                {
                    H.AdjointDerivative_FF<2>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                                B_in.data(), B_out.data(), &neumann_data_scat_ptr, BAEMM::Helmholtz_OpenCL::WaveType::Radial, cg_tol, gmres_tol);
                    break;
                }
                case 4:
                {
                    H.AdjointDerivative_FF<4>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                                B_in.data(), B_out.data(), &neumann_data_scat_ptr, BAEMM::Helmholtz_OpenCL::WaveType::Radial, cg_tol, gmres_tol);
                    break;
                }
                case 8:
                {
                    H.AdjointDerivative_FF<8>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                                B_in.data(), B_out.data(), &neumann_data_scat_ptr, BAEMM::Helmholtz_OpenCL::WaveType::Radial, cg_tol, gmres_tol);
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

    WriteInOut(vertex_count, dim, B_out, "B.bin");

    if( !file.good() )
    {        
        neumann_data_scat = Tensor2<Complex,Int>(   vertex_count, wave_count    );
        neumann_data_scat.Read(neumann_data_scat_ptr);

        WriteInOut(vertex_count, wave_count, neumann_data_scat,"NeumannDataScat.bin");
    }
    else
    {       
        free(neumann_data_scat_ptr);
    }

    file.close();

    return 0;
}