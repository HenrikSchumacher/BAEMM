#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <pwd.h>
#include <filesystem>


#define TOOLS_ENABLE_PROFILER

//#include "../submodules/Repulsor/submodules/Tensors/submodules/Tools/Tools.hpp"
#include "../Helmholtz_OpenCL.hpp"
#include "../ReadMeshFromFile.hpp"

using namespace Tools;
using namespace Tensors;
using namespace BAEMM;

using Helmholtz_T = BAEMM::Helmholtz_OpenCL;
constexpr BAEMM::WaveType Plane  = BAEMM::WaveType::Plane;
//constexpr BAEMM::WaveType Radial = BAEMM::WaveType::Radial;

using Real      = Real64;
using Complex   = std::complex<Real>;
using Int       = Int64;
using LInt      = Int64;

constexpr Int DIM = 3;

int main()
{
    //  std::filesystem::path this_file { __FILE__ };
    //  std::filesystem::path repo_dir = this_file.parent_path().parent_path();
    std::filesystem::path repo_dir = "/HOME1/users/guests/jannr/github/BAEMM_test";
    std::filesystem::path mesh_dir = repo_dir / "Meshes";
    std::filesystem::path home_dir = HomeDirectory();
        
    Profiler::Clear( home_dir );
    

//    std::string mesh_name { "Triceratops_00081920T" };
    std::string mesh_name { "Triceratops_12_00081920T" };
    std::filesystem::path mesh_file = home_dir / (mesh_name + ".txt");
    
//    std::string mesh_name { "Bunny_00086632T" };
//    std::string mesh_name { "Spot_00005856T" };
//    std::string mesh_name { "Spot_00023424T" };
//    std::string mesh_name { "Spot_00093696T" };
//    std::string mesh_name { "Bob_00042752T" };
//    std::string mesh_name { "Blub_00056832T" };
//    std::string mesh_name { "TorusMesh_00038400T" };

//    std::filesystem::path mesh_file = mesh_dir / (mesh_name + ".txt");
    
    std::filesystem::path meas_file = mesh_dir / ("Sphere_00005120T.txt");
    
    
    {
        std::ifstream f( mesh_file.string() );
        if ( !f.good() )
        {
            eprint("File " + mesh_file.string() + " not found. Exiting.");
            exit(1);
        }
    }
    
    {
        std::ifstream f( meas_file.string() );
        if ( !f.good() )
        {
            eprint("File " + mesh_file.string() + " not found. Exiting.");
            exit(1);
        }
    }
    
    tic("Loading obstacle.");
    Tensor2<Real,Int> coords;
    Tensor2<Int, Int> simplices;
    bool mesh_loadedQ = ReadMeshFromFile<Real,Int>( mesh_file.string(), coords, simplices );

    if( !mesh_loadedQ )
    {
        eprint("Failed to load mesh. Aborting.");
        
        exit(-1);
    }
    toc("Loading obstacle.");
    
    print("");
    
    tic("Loading far field sphere.");
    Tensor2<Real,Int> meas_directions;
    Tensor2<Int, Int> simplices_meas;
    bool meas_loadedQ = ReadMeshFromFile<Real,Int>( meas_file.string(), meas_directions, simplices_meas );
    
    if( !meas_loadedQ )
    {
        eprint("Failed to load far field sphere. Aborting.");
        
        exit(-1);
    }
    toc("Loading far field sphere.");
    
    
    print("");
    
    const Int vertex_count  = coords.Dimension(0);
    const Int simplex_count = simplices.Dimension(0);
    const Int meas_count    = meas_directions.Dimension(0);
    
    Int device = 0;
    Int thread_count = 8;
//    Int thread_count = 1;

    logprint("Initialize Helmholtz object");
    Helmholtz_T H (
        coords.data(),          vertex_count,
        simplices.data(),       simplex_count,
        meas_directions.data(), meas_count,
        device,
        thread_count
    );
    
//    constexpr Int wave_count = 32;
    constexpr Int wave_count = 16;
    constexpr Int wave_chunk_size = 16;
    constexpr Int wave_chunk_count = wave_count / wave_chunk_size;
    
    Tensor1<Real,Int> kappa ( wave_chunk_count     );
    Tensor2<Real,Int> inc   ( wave_chunk_size, DIM );
    
    for (int i = 0 ; i < wave_chunk_count; i++)
    {
//        kappa[i] = ( 1 + 2 * ( i + 2 ) ) * Scalar::Pi<Real>;
        kappa[i] = 8 * Scalar::Pi<Real>;
    }

    inc( 0,0) = 0.22663516023574246;
    inc( 0,1) = 0.4654289185373517;
    inc( 0,2) = 0.8555772472045235;
    
    inc( 1,0) = -0.3474510994575331;
    inc( 1,1) = -0.7864451523456505;
    inc( 1,2) = -0.5106679506663584;
    
    inc( 2,0) = -0.9541836471144971;
    inc( 2,1) = -0.27451273343735705;
    inc( 2,2) = -0.11906438073591856;
    
    inc( 3,0) = 0.43775029208027166;
    inc( 3,1) = -0.8876694056416434;
    inc( 3,2) = 0.1428905457734996;
    
    inc( 4,0) = -0.44907802104219113;
    inc( 4,1) = -0.8139693295211834;
    inc( 4,2) = 0.36848726113078095;
    
    inc( 5,0) = 0.47666884531973414;
    inc( 5,1) = 0.3192753380458682;
    inc( 5,2) = -0.8190543757390276;
    
    inc( 6,0) = -0.02194389227182042;
    inc( 6,1) = 0.8818743114784301;
    inc( 6,2) = -0.47097363444932655;
    
    inc( 7,0) = 0.786615210967289;
    inc( 7,1) = -0.1684579992437276;
    inc( 7,2) = 0.5940188653281036;
    
    inc( 8,0) = 0.42244725439759406;
    inc( 8,1) = -0.5580931680624308;
    inc( 8,2) = -0.7141920841160128;
    
    inc( 9,0) = 0.6792359217126083;
    inc( 9,1) = 0.7269051927489542;
    inc( 9,2) = 0.10122945919953094;
    
    inc(10,0) = -0.3904321861768485;
    inc(10,1) = 0.03454064212408624;
    inc(10,2) = -0.919983506394991;
    
    inc(11,0) = -0.3009524704632311;
    inc(11,1) = 0.8792870632871999;
    inc(11,2) = 0.36916374531886265;
    
    inc(12,0) = -0.006080463353741372;
    inc(12,1) = -0.38604914872328944;
    inc(12,2) = 0.9224581739761577;
    
    inc(13,0) = 0.9742753958166932;
    inc(13,1) = -0.0853848807263007;
    inc(13,2) = -0.20855904499585115;
    
    inc(14,0) = -0.8068074072061289;
    inc(14,1) = 0.5272347230598002;
    inc(14,2) = -0.2666183686046023;
    
    inc(15,0) = -0.7266826985737691;
    inc(15,1) = 0.12599246204091846;
    inc(15,2) = 0.675320779409617;

    H.UseDiagonal(true);
    
    using Mesh_T  = SimplicialMesh<2,3,Real,Int,LInt,Real,Real>;
    using Mesh_Ptr_T = std::shared_ptr<Mesh_T>;
    
    logprint("Initialize mesh");
    Mesh_Ptr_T M = std::make_shared<Mesh_T>(
        coords.data(),    vertex_count,
        simplices.data(), simplex_count,
        thread_count
    );
    
    M->cluster_tree_settings.split_threshold                        =  2;
    M->cluster_tree_settings.thread_count                           =  0; // take as many threads as there are used by SimplicialMesh M
    M->block_cluster_tree_settings.far_field_separation_parameter   =  0.125;
    M->adaptivity_settings.theta                                    = 10.0;

    const Real q = 6;
    const Real p = 12;

    logprint("Initialize energy");
    TangentPointEnergy0<Mesh_T> tpe (q,p);
    
    logprint("Initialize metric");
    TangentPointMetric0<Mesh_T> tpm (q,p);
    
    
    Real cg_tol          = 0.00001;
    Real gmres_tol       = 0.005;
    Real gmres_tol_outer = 0.01;
    
//    Real regpar          = 0.001;
    
    Real regpar = 0.000001 * 0.00687195;
    
    // The operator for the metric.
    auto A = [regpar,&M,&tpm]( cptr<Real> X, mptr<Real> Y )
    {
        // Y = regpar * Metric.X
        tpm.MultiplyMetric( *M,
            regpar,             X, DIM,
            Scalar::Zero<Real>, Y, DIM,
            DIM
        );
    };

    Real one_over_regpar = Inv<Real>(regpar);

    // The operator for the preconditioner.
    auto P = [one_over_regpar,&M,&tpm]( cptr<Real> X, mptr<Real> Y )
    {
        // Y = one_over_regpar * Prec.X
        tpm.MultiplyPreconditioner( *M,
            one_over_regpar,    X, DIM,
            Scalar::Zero<Real>, Y, DIM,
            DIM
        );
    };

    
    // Far field
    Tensor2<Complex,Int> F ( meas_count, wave_count );
    
    // Derivative ofr 1/2 |F|^2.
    Tensor2<Real,Int> FDF  ( vertex_count, DIM );
    
    // Right-hand side for solver.
    Tensor2<Real,Int> B    ( vertex_count, DIM );
    
    // Search direction.
    Tensor2<Real,Int> X    ( vertex_count, DIM ); // search direction.
    
    
    Complex * du_dn = nullptr;
    
    
    logprint("DE");
    tpe.Differential( *M, regpar, Scalar::Zero<Real>, B.data(), B.Dimension(1) );
    
    
    logprint("FDF");
    H.FarField<wave_count>(
        kappa.data(), wave_chunk_count,
        inc.data(),   wave_chunk_size,
        F.data(),
        Plane, cg_tol, gmres_tol
    );

    
    H.AdjointDerivative_FF<wave_count>(
        kappa.data(), wave_chunk_count,
        inc.data(),   wave_chunk_size,
        F.data(), FDF.data(),
        du_dn, BAEMM::WaveType::Plane, cg_tol, gmres_tol
    );
    
    H.MassMatrix().Dot<DIM>(
        Tools::Scalar::One<Real>, FDF.data(), static_cast<Helmholtz_T::Int>(FDF.Dimension(1)),
        Tools::Scalar::One<Real>, B.data(),   static_cast<Helmholtz_T::Int>(B.Dimension(1)),
        DIM
    );
    
    bool succeeded;
    
    print("");
    
    logprint("GaussNewtonSolve");
    
    tic("GaussNewtonSolve");
    succeeded = H.GaussNewtonSolve<wave_count>(
        kappa.data(), wave_chunk_count,
        inc.data(),   wave_chunk_size,
        A, P,
        Scalar::One <Real>, B.data(), B.Dimension(1),
        Scalar::Zero<Real>, X.data(), X.Dimension(1),
        du_dn,
        Plane, cg_tol, gmres_tol, gmres_tol_outer
    );
    toc("GaussNewtonSolve");
    
    dump(succeeded);
    
    
    if( du_dn != nullptr )
    {
        free( du_dn );
    }
    
    return 0;
}
