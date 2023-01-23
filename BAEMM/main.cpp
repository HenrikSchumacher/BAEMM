//
//  main.cpp
//  BÄMM!
//
//  Created by Henrik on 21.01.23.
//

#include <iostream>


#include <iostream>

#include <sys/types.h>
#include <pwd.h>
#include <complex>

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <Accelerate/Accelerate.h>

// We have to toggle which domain dimensions and ambient dimensions shall be supported by runtime polymorphism before we load Repulsor.hpp
// You can activate everything you want, but compile times might increase substatially.
using INT     = int32_t;
using EXTINT  = int64_t;

using REAL    = double;
using SREAL   = float;
using COMP    = std::complex<SREAL>;

using EXTREAL = double;

//#define REMESHER_VERBATIM

#define TOOLS_ENABLE_PROFILER // enable profiler

#include "../BAEMM.hpp"

int main(int argc, const char * argv[])
{
    
    using namespace Repulsor;
    using namespace Tensors;
    using namespace Tools;
    using namespace BAEMM;
    
    const char * homedir = getenv("HOME");

    if( homedir == nullptr)
    {
        homedir = getpwuid(getuid())->pw_dir;
    }
    std::string path ( homedir );
    
    Profiler::Clear( path );

    
    int thread_count = 8;
//    int thread_count = 1;
    omp_set_num_threads(thread_count);
    
    using Mesh_T   = SimplicialMeshBase<REAL,INT,SREAL,EXTREAL>;
    using Energy_T = EnergyBase<REAL,INT,SREAL,EXTREAL>;
    using Metric_T = MetricBase<REAL,INT,SREAL,EXTREAL>;
    
    // Create a factory that can make instances of SimplicialMesh with domain dimension in the range 2,...,2 and ambient dimension in the range 3,...,3.
    SimplicialMesh_Factory<Mesh_T,2,2,3,3> mesh_factory;
    
//    std::string file_name = path + "/github/BAEMM/Meshes/TorusMesh_00153600T.txt";
    std::string file_name = path + "/github/BAEMM/Meshes/TorusMesh_00038400T.txt";
//    std::string file_name = path + "/github/BAEMM/Meshes/TorusMesh_00009600T.txt";
//    std::string file_name = path + "/github/BAEMM/Meshes/TorusMesh_00000600T.txt";
    
    dump(file_name);
    
    tic("Initializing mesh");
    std::unique_ptr<Mesh_T> M_ptr = mesh_factory.Make_FromFile<REAL,INT,SREAL,EXTREAL>(file_name, thread_count);

    Mesh_T & M = *M_ptr;  // I don't like pointers. Give me a reference.

    dump(M.ThreadCount());

    toc("Initializing mesh");

    // Some quite decent settings for 2-dimensional surfaces.
    M.cluster_tree_settings.split_threshold                        =  2;
    M.cluster_tree_settings.thread_count                           =  0; // take as many threads as there are used by SimplicialMesh M
    M.block_cluster_tree_settings.far_field_separation_parameter   =  0.25;
    M.adaptivity_settings.theta                                    = 10.0;

    print("");
    
    const REAL q = 6;
    const REAL p = 12;

    TangentPointEnergy_Factory<Mesh_T,2,2,3,3> TPE_factory;

    std::unique_ptr<Energy_T> tpe_ptr = TPE_factory.Make( 2, 3, q, p );

    const auto & tpe = *tpe_ptr;

    double en;
    // Mesh_T::CotangentVector_T is Tensor2<REAL,INT> in this case. It is a simple container class for heap-allocated matrices.
    Mesh_T::CotangentVector_T diff;

    tic("tpe.Energy(M)");
    en = tpe.Value(M);
    toc("tpe.Energy(M)");

    dump(en);

    tic("tpe.Differential(M)");
    diff = tpe.Differential(M);
    toc("tpe.Differential(M)");

    print("");
    print("");

    static constexpr INT n_waves = 32;
    
    static constexpr INT i_blk   = 4;
    static constexpr INT j_blk   = 2;

    SREAL kappa = 2.;
    SREAL kappa_step = 0.1;
    
    NS::AutoreleasePool* p_pool = NS::AutoreleasePool::alloc()->init();
    
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    
    HelmholtzOperator_AoS   <Mesh_T> H_AoS   ( M );
    HelmholtzOperator_SoA   <Mesh_T> H_SoA   ( M );
    HelmholtzOperator_Metal <Mesh_T> H_Metal ( device, M );
    
    const INT n = M.SimplexCount();
    
    Tensor2<COMP,INT>  X_True ( n, n_waves );
    Tensor2<COMP,INT>  X      ( n, n_waves );
    Tensor2<COMP,INT>  Y      ( n, n_waves );
    Tensor2<COMP,INT>  Z      ( n, n_waves );

    Tensor2<SREAL,INT> Re_X   ( n, n_waves );
    Tensor2<SREAL,INT> Im_X   ( n, n_waves );
    Tensor2<SREAL,INT> Re_Y   ( n, n_waves );
    Tensor2<SREAL,INT> Im_Y   ( n, n_waves );
    
    const uint size = n * n_waves * sizeof(float);
    
    MTL::Buffer * Re_X_Metal = device->newBuffer(size, MTL::ResourceStorageModeManaged);
    MTL::Buffer * Im_X_Metal = device->newBuffer(size, MTL::ResourceStorageModeManaged);
    MTL::Buffer * Re_Y_Metal = device->newBuffer(size, MTL::ResourceStorageModeManaged);
    MTL::Buffer * Im_Y_Metal = device->newBuffer(size, MTL::ResourceStorageModeManaged);
    
    
    mut<float> Re_X_Metal_ptr = static_cast<float*>( Re_X_Metal->contents() );
    mut<float> Im_X_Metal_ptr = static_cast<float*>( Im_X_Metal->contents() );
    mut<float> Re_Y_Metal_ptr = static_cast<float*>( Re_Y_Metal->contents() );
    mut<float> Im_Y_Metal_ptr = static_cast<float*>( Im_Y_Metal->contents() );

    Re_Y.Random();
    Im_Y.Random();
    
    Re_Y.Write( Re_Y_Metal_ptr );   Re_Y_Metal->didModifyRange({0,size});
    Im_Y.Write( Im_Y_Metal_ptr );   Im_Y_Metal->didModifyRange({0,size});
    
    for( INT i = 0; i < n; ++i )
    {
        for( INT k = 0; k < n_waves; ++k )
        {
            Y(i,k) = COMP( Re_Y(i,k), Im_Y(i,k) );
        }
    }
    
    
    valprint("n      ",n      );
    valprint("n_waves",n_waves);
    valprint("i_blk  ",i_blk  );
    valprint("j_blk  ",j_blk  );
    
//    REAL dummy (0);
//
//    X.SetZero();
//    tic("AoS naive");
//    for( INT k = 0; k < n_waves; ++k )
//    {
//        H_AoS.Multiply<1>( Y.data(), X.data(), kappa + k * kappa_step, kappa_step, thread_count );
//    }
//    toc("AoS naive");
//    dummy += X.MaxNorm();
//    dump(dummy);
//
//    print("");
    
    X_True.SetZero();
    H_AoS.Multiply<n_waves>(
        Y.data(), X_True.data(), kappa, kappa_step, thread_count
    );
    
    print("");
//
//    X.SetZero();
//    H_AoS.Multiply_Blocked<i_blk,j_blk,n_waves,true,true,false>(
//        Y.data(), X.data(), kappa, kappa_step, thread_count
//    );
//    Subtract(X_True,X,Z);
//    valprint("error",Z.MaxNorm()/X_True.MaxNorm());
//
//    X.SetZero();
//    H_AoS.Multiply_Blocked<i_blk,j_blk,n_waves,true,false,false>(
//        Y.data(), X.data(), kappa, kappa_step, thread_count
//    );
//    Subtract(X_True,X,Z);
//    valprint("error",Z.MaxNorm()/X_True.MaxNorm());
//
//    X.SetZero();
//    H_AoS.Multiply_Blocked<i_blk,j_blk,n_waves,false,true,false>(
//        Y.data(), X.data(), kappa, kappa_step, thread_count
//    );
//    Subtract(X_True,X,Z);
//    valprint("error",Z.MaxNorm()/X_True.MaxNorm());
//
//    X.SetZero();
//    H_AoS.Multiply_Blocked<i_blk,j_blk,n_waves,false,false,false>(
//        Y.data(), X.data(), kappa, kappa_step, thread_count
//    );
//    Subtract(X_True,X,Z);
//    valprint("error",Z.MaxNorm()/X_True.MaxNorm());


    zerofy_buffer( Re_X_Metal_ptr, n * n_waves );   Re_X_Metal->didModifyRange({0,size});
    zerofy_buffer( Im_X_Metal_ptr, n * n_waves );   Im_X_Metal->didModifyRange({0,size});
    H_Metal.Multiply<16,n_waves>(
        Re_Y_Metal, Im_Y_Metal, Re_X_Metal, Im_X_Metal, kappa, kappa_step
    );
    Re_X.Read( Re_X_Metal_ptr );
    Im_X.Read( Im_X_Metal_ptr );
    
    GetDifference( Re_X, Im_X, X_True, Z );
    valprint("error",Z.MaxNorm()/X_True.MaxNorm());
    
    zerofy_buffer( Re_X_Metal_ptr, n * n_waves );   Re_X_Metal->didModifyRange({0,size});
    zerofy_buffer( Im_X_Metal_ptr, n * n_waves );   Im_X_Metal->didModifyRange({0,size});
    H_Metal.Multiply<32,n_waves>(
        Re_Y_Metal, Im_Y_Metal, Re_X_Metal, Im_X_Metal, kappa, kappa_step
    );
    Re_X.Read( Re_X_Metal_ptr );
    Im_X.Read( Im_X_Metal_ptr );
    GetDifference( Re_X, Im_X, X_True, Z );
    valprint("error",Z.MaxNorm()/X_True.MaxNorm());
    
    zerofy_buffer( Re_X_Metal_ptr, n * n_waves );   Re_X_Metal->didModifyRange({0,size});
    zerofy_buffer( Im_X_Metal_ptr, n * n_waves );   Im_X_Metal->didModifyRange({0,size});
    H_Metal.Multiply<64,n_waves>(
        Re_Y_Metal, Im_Y_Metal, Re_X_Metal, Im_X_Metal, kappa, kappa_step
    );
    Re_X.Read( Re_X_Metal_ptr );
    Im_X.Read( Im_X_Metal_ptr );
    GetDifference( Re_X, Im_X, X_True, Z );
    valprint("error",Z.MaxNorm()/X_True.MaxNorm());

//    for( uint i = 0; i < n; ++i )
//    {
//        for( uint j = 0; j < n_waves; ++j )
//        {
//            if( std::abs(Z(i,j)) > 0.001 )
//            {
//                dump(i);
//                dump(j);
//                dump(Z(i,j));
//                valprint("X(i,j)",COMP( Re_X(i,j),Im_X(i,j)) );
//            }
//        }
//    }
    
    p_pool->release();

    return 0;
}
