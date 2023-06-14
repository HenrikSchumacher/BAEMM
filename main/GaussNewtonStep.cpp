#include <sys/types.h>
#include <complex>
#include <iostream>
#include <fstream> 

using uint = unsigned int;

#include "../Helmholtz_OpenCL.hpp"
#include "../ReadWrite/ReadFiles.hpp"
#include "../ReadWrite/WriteFiles.hpp"
// #include "../../Repulsor"

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
    ReadInOut(meas_count, wave_count, B_in);

    BAEMM::Helmholtz_OpenCL H (
        coords.data(),    vertex_count,
        simplices.data(), simplex_count, 
        meas_directions.data(), meas_count, 
        GPU_device, int_cast<Int>(16)
        );

    constexpr Int ambient_dimension = 3;
    constexpr Int domain_dimension = 2;
    constexpr Int thread_count = 8;

    // std::unique_ptr<SimplicialMeshBase<Real,Int,Real,Real>> M = MakeSimplicialMesh<REAL,INT,REAL,REAL>(
    //     coords.data(),  vertex_count, ambient_dimension,
    //     simplices.data(),  simplex_count, domain_dimension+1,
    //     thread_count
    // );

    // M->cluster_tree_settings.split_threshold                        =  2;
    // M->cluster_tree_settings.thread_count                           =  0; // take as many threads as there are used by SimplicialMesh M
    // M->block_cluster_tree_settings.far_field_separation_parameter   =  0.5;
    // M->adaptivity_settings.theta                                    = 10.0;

    // M->GetClusterTree();
    // M->GetBlockClusterTree();

    // Tensor2<Real,Int> diff ( vertex_count, ambient_dimension );

    // const double alpha  = 6;
    // const double beta   = 12;
    // const double weight = 1;

    // M->SetTangentPointExponents(alpha, beta);
    // M->SetTangentPointWeight(weight);

    // bool add_to = true;
    // auto A = [&]( const R_ext * x, R_ext *y )
    //     {   
            
    //         M->TangentPointEnergy_Differential(x, add_to);
    //         memccpy(y,x,vertex_count*3*sizeof(R_ext));
    //     };

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