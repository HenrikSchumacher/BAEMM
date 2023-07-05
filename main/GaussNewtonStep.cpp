#include <sys/types.h>
#include <complex>
#include <iostream>
#include <fstream> 

using uint = unsigned int;
using LInt = long long;

#include "../Helmholtz_OpenCL.hpp"
#include "../ReadWrite/ReadFiles.hpp"
#include "../ReadWrite/WriteFiles.hpp"
// #include "../../Repulsor"

using namespace Tools;
using namespace Tensors;
using namespace Repulsor;

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
    
    Real regpar;

    constexpr Int thread_count = 16;

    Tensor2<Int,Int>        simplices;
    Tensor2<Real,Int>       meas_directions;
    Tensor2<Real,Int>       incident_directions;
    Tensor1<Real,Int>       kappa;
    Tensor2<Real,Int>       coords;
    Tensor2<Complex,Int>    B_in;

    ReadFixes(vertex_count, simplex_count, meas_count, wave_chunk_count, wave_chunk_size, GPU_device, simplices, meas_directions, incident_directions, kappa);

    ReadCoordinates(vertex_count, coords);

    Int wave_count = wave_chunk_count * wave_chunk_size;

    Int dim = 3;
    ReadInOut(meas_count, wave_count, B_in);

    BAEMM::Helmholtz_OpenCL H (
        coords.data(),    vertex_count,
        simplices.data(), simplex_count, 
        meas_directions.data(), meas_count, 
        GPU_device, thread_count
        );

    H.UseDiagonal(true);
    H.SetBlockSize(64);

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

    //----------------------------------------------------------------------------------------------------------------------------------------------------------------
    using Mesh_T     = SimplicialMesh<2,3,Real,Int,LInt,SReal,ExtReal>;
    using Mesh_Ptr_T = std::shared_ptr<Mesh_T>;

    constexpr Int NRHS = 3;

    Mesh_Ptr_T M = std::make_shared<Mesh_T>(
        coords.data(),  vertex_count,
        simplices.data(),  simplex_count,
        thread_count
        );

    M->cluster_tree_settings.split_threshold                        =  2;
    M->cluster_tree_settings.thread_count                           =  thread_count; // take as many threads as there are used by SimplicialMesh M
    M->block_cluster_tree_settings.far_field_separation_parameter   =  0.125f;
    M->adaptivity_settings.theta                                    = 10.0f;

    const Real q  = 6;
    const Real p  = 12;
    const Real s = (p - 2) / q;

    TangentPointEnergy0<Mesh_T>       tpe        (q,p);
    TangentPointMetric0<Mesh_T>       tpm        (q,p);
    PseudoLaplacian    <Mesh_T,false> pseudo_lap (2-s);

    // The operator for the metric.
    auto A = [&]( ptr<Real> X, mut<Real> Y )
    {
        tpm.MultiplyMetric( *M, regpar, X, Scalar::One<Real>, Y, NRHS );
    };

    Real one_over_regpar = 1/regpar;

    Tensor2<Real,Int> Z_buffer  ( M->VertexCount(), NRHS );

    mut<Real> Z  = Z_buffer.data();

    // The operator for the preconditioner.
    auto P = [&]( ptr<Real> X, mut<Real> Y )
    {
        M->H1Solve( X, Y, NRHS );
        pseudo_lap.MultiplyMetric( *M, one_over_regpar, Y, Scalar::Zero<Real>, Z, NRHS );
        M->H1Solve( Z, Y, NRHS );
    };

    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------

    Real gmres_tol_outer = 0.005;

    Tensor2<Real,Int> DE (vertex_count,3);
    Tensor2<Real,Int> grad (vertex_count,3);
    Tensor2<Real,Int> grad_strong (vertex_count,3);

    tpe.Differential( *M ).Write( DE.data() );
    
    ptr<Real> DE_ptr = DE.data();
    mut<Real> grad_ptr = grad.data();

    switch (wave_count)
    {
        case 8:
        {
            H.AdjointDerivative_FF<8>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                        B_in.data(), grad_strong.data(), &neumann_data_scat_ptr, cg_tol, gmres_tol);
            Mass.Dot(
                static_cast<Real>(1.0), grad_strong.data(), 3,
                static_cast<Real>(0.0), grad_ptr, 3,
                3
            );

            combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Generic>(regpar,DE_ptr,static_cast<Real>(1.0f),grad_ptr,vertex_count * 3, thread_count);

            H.GaussNewtonStep<8>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size, 
                        A, P, grad_ptr, B_out.data(), &neumann_data_scat_ptr, cg_tol, gmres_tol, gmres_tol_outer);
            break;
        }
        case 16:
        {
            H.AdjointDerivative_FF<16>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                        B_in.data(), grad_strong.data(), &neumann_data_scat_ptr, cg_tol, gmres_tol);
            Mass.Dot(
                static_cast<Real>(1.0), grad_strong.data(), 3,
                static_cast<Real>(0.0), grad_ptr, 3,
                3
            );

            combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Generic>(regpar,DE_ptr,static_cast<Real>(1.0f),grad_ptr,vertex_count * 3, thread_count);

            H.GaussNewtonStep<16>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size, 
                        A, P, grad_ptr, B_out.data(), &neumann_data_scat_ptr, cg_tol, gmres_tol, gmres_tol_outer);
            break;
        }
        case 32:
        {
            H.AdjointDerivative_FF<32>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                        B_in.data(), grad_strong.data(), &neumann_data_scat_ptr, cg_tol, gmres_tol);
            Mass.Dot(
                static_cast<Real>(1.0), grad_strong.data(), 3,
                static_cast<Real>(0.0), grad_ptr, 3,
                3
            );

            combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Generic>(regpar,DE_ptr,static_cast<Real>(1.0f),grad_ptr,vertex_count * 3, thread_count);

            H.GaussNewtonStep<32>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size, 
                        A, P, grad_ptr, B_out.data(), &neumann_data_scat_ptr, cg_tol, gmres_tol, gmres_tol_outer);
            break;
        }
        case 64:
        {
            H.AdjointDerivative_FF<64>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                        B_in.data(), grad_strong.data(), &neumann_data_scat_ptr, cg_tol, gmres_tol);
            Mass.Dot(
                static_cast<Real>(1.0), grad_strong.data(), 3,
                static_cast<Real>(0.0), grad_ptr, 3,
                3
            );

            combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Generic>(regpar,DE_ptr,static_cast<Real>(1.0f),grad_ptr,vertex_count * 3, thread_count);

            H.GaussNewtonStep<64>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size, 
                        A, P, grad_ptr, B_out.data(), &neumann_data_scat_ptr, cg_tol, gmres_tol, gmres_tol_outer);
            break;
        }
    }
    
    WriteInOut(vertex_count, dim, B_out);

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

void read_regpar (Real& regpar)
{
    ifstream s ("regpar.txt");
    
    if( !s.good() )
    {
        eprint("ReadFromFile: File regpar.txt could not be opened.");
        
        return;
    }
    
    s >> regpar;
}