#include <sys/types.h>
#include <complex>
#include <iostream>
#include <fstream> 
#include <filesystem>

#define TOOLS_ENABLE_PROFILER

#include "../Helmholtz_OpenCL.hpp"
#include "../ReadWrite/ReadFiles.hpp"
#include "../ReadWrite/WriteFiles.hpp"

using namespace Tools;
using namespace Tensors;
using namespace Repulsor;

using uint = unsigned int;
using LInt = long long;
using Int = int;
using Real = double;
using Complex = std::complex<Real>;

using Wave_T = BAEMM::WaveType;

void ReadRegpar(Real& regpar)
{
    ifstream s ("regpar.txt");
    
    if( !s.good() )
    {
        eprint("ReadFromFile: File regpar.txt could not be opened.");
        
        return;
    }
    
    s >> regpar;
}

void WriteSucceeded(Int& succeeded)
{
    ofstream s ("suc.txt");
    
    if( !s.good() )
    {
        eprint("ReadFromFile: File regpar.txt could not be opened.");
        
        return;
    }
    
    s << succeeded;
}

int main()
{
    // This routine reads out the data from the files data.txt (which contains data as specified in WriteFiles), simplices.bin, meas_direction.bin and coords.bin.
    // The direction of the derivative is read from B.bin, then the Gauss Newton step is calculated.
    // In the case of a radial wave incident_directions are the point sources

    Int vertex_count, simplex_count, meas_count;
    Int wave_chunk_count, wave_chunk_size;
    Int GPU_device;   
    string wave_type;

    Real regpar;

    constexpr Int thread_count = 16;

    Tensor2<Int,Int>        simplices;
    Tensor2<Real,Int>       meas_directions;
    Tensor2<Real,Int>       incident_directions;
    Tensor1<Real,Int>       kappa;
    Tensor2<Real,Int>       coords;
    Tensor2<Real,Int>       B_in;

    ReadFixes(vertex_count, simplex_count, meas_count, wave_chunk_count, wave_chunk_size, GPU_device, wave_type, simplices, meas_directions, incident_directions, kappa);

    ReadCoordinates(vertex_count, coords);

    ReadRegpar(regpar);

    Profiler::Clear("/HOME1/users/guests/jannr/BEM");

    Int wave_count = wave_chunk_count * wave_chunk_size;

    Int dim = 3;
    ReadInOut(vertex_count, dim, B_in, "B.bin");

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
    Real gmres_tol = static_cast<Real>(0.005);

    Tensor2<Complex,Int> neumann_data_scat;
    Complex* neumann_data_scat_ptr = nullptr;

    std::fstream file ("NeumannDataScat.bin");
    
    if( file.good() )
    {
        ReadInOut(vertex_count, wave_count, neumann_data_scat,"NeumannDataScat.bin");
        neumann_data_scat_ptr = neumann_data_scat.data();
    }

    //----------------------------------------------------------------------------------------------------------------------------------------------------------------
    using Mesh_T     = SimplicialMesh<2,3,Real,Int,LInt,Real,Real>;
    using Mesh_Ptr_T = std::shared_ptr<Mesh_T>;

    Mesh_Ptr_T M = std::make_shared<Mesh_T>(
        coords.data(),  vertex_count,
        simplices.data(),  simplex_count,
        thread_count
        );

    M->cluster_tree_settings.split_threshold                        =  2;
    M->cluster_tree_settings.thread_count                           =  thread_count; // take as many threads as there are used by SimplicialMesh M
    M->block_cluster_tree_settings.far_field_separation_parameter   =  static_cast<Real>(0.125);
    M->adaptivity_settings.theta                                    = static_cast<Real>(10.0);

    const Real q  = 6;
    const Real p  = 12;
    const Real s = (p - 2) / q;

    TangentPointMetric0<Mesh_T>       tpm        (q,p);
    
    // TODO: Should be obsolete now because we may use tpm.MultiplyPreconditioner.
    PseudoLaplacian    <Mesh_T,false> pseudo_lap (2-s);

    // The operator for the metric.
    auto A = [&]( cptr<Real> X, mptr<Real> Y )
    {
        tpm.MultiplyMetric( *M, regpar, X, Scalar::One<Real>, Y, dim );
    };

    Real one_over_regpar = Inv<Real>(regpar);

    Tensor2<Real,Int> Z_buffer  ( vertex_count, dim );

    mptr<Real> Z  = Z_buffer.data();

    // The operator for the preconditioner.
    auto P = [&]( cptr<Real> X, mptr<Real> Y )
    {
//        tpm.MultiplyPreconditioner( *M, one_over_regpar, X, Scalar::Zero<Real>, Y, dim );
        
        M->H1Solve( X, Y, dim );
        pseudo_lap.MultiplyMetric( *M, one_over_regpar, Y, Scalar::Zero<Real>, Z, dim );
        M->H1Solve( Z, Y, dim );
    };

    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------

    Real gmres_tol_outer = 0.01;
    Int succeeded;
    
    const Wave_T wt = (wave_type == "Radial") ? Wave_T::Radial : Wave_T::Plane;

    switch (wave_count)
    {
        case 1:
        {
            succeeded = H.GaussNewtonStep<1>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                        A, P, B_in.data(), B_out.data(), &neumann_data_scat_ptr, wt, cg_tol, gmres_tol, gmres_tol_outer);
            break;
        }
        case 2:
        {
            succeeded = H.GaussNewtonStep<2>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                        A, P, B_in.data(), B_out.data(), &neumann_data_scat_ptr, wt, cg_tol, gmres_tol, gmres_tol_outer);
            break;
        }
        case 4:
        {
            succeeded = H.GaussNewtonStep<4>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                        A, P, B_in.data(), B_out.data(), &neumann_data_scat_ptr, wt, cg_tol, gmres_tol, gmres_tol_outer);
            break;
        }
        case 8:
        {
            succeeded = H.GaussNewtonStep<8>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                        A, P, B_in.data(), B_out.data(), &neumann_data_scat_ptr, wt, cg_tol, gmres_tol, gmres_tol_outer);
            break;
        }
        case 16:
        {
            succeeded = H.GaussNewtonStep<16>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                        A, P, B_in.data(), B_out.data(), &neumann_data_scat_ptr, wt, cg_tol, gmres_tol, gmres_tol_outer);
            break;
        }
        case 32:
        {
            succeeded = H.GaussNewtonStep<32>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                        A, P, B_in.data(), B_out.data(), &neumann_data_scat_ptr, wt, cg_tol, gmres_tol, gmres_tol_outer);
            break;
        }
        case 64:
        {
            succeeded = H.GaussNewtonStep<64>( kappa.data(), wave_chunk_count, incident_directions.data(), wave_chunk_size,
                        A, P, B_in.data(), B_out.data(), &neumann_data_scat_ptr, wt, cg_tol, gmres_tol, gmres_tol_outer);
            break;
        }
        default:
        {
            eprint("Non valid wave count.");
            break;
        }
    }

    std::string varAsString = std::to_string(1000000 * regpar);

    std::string path_log = "/HOME1/users/guests/jannr/Tools_Log_GN_iteration_" +  varAsString + ".txt";
    std::string path_profile = "/HOME1/users/guests/jannr/Tools_Log_GN_iteration_" +  varAsString + ".tsv";

    std::filesystem::rename("/HOME1/users/guests/jannr/BEM/Tools_Log.txt",path_log);
    std::filesystem::rename("/HOME1/users/guests/jannr/BEM/Tools_Profile.tsv",path_profile);
    
    WriteInOut(vertex_count, dim, B_out, "B.bin");

    WriteSucceeded(succeeded);

    if( !file.good() )
    {      
        neumann_data_scat = Tensor2<Complex,Int>(   vertex_count, wave_count    );
        neumann_data_scat.Read(neumann_data_scat_ptr); 
        WriteInOut(vertex_count, wave_count, neumann_data_scat,"NeumannDataScat.bin");

        free(neumann_data_scat_ptr);
    }

    file.close();

    return 0;
}
