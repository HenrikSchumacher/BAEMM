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
using Int     =  int32_t;
using UInt    = uint32_t;
using ExtInt  =  int64_t;

using Real    = float64_t;
using Float   = float32_t;
using Complex = std::complex<Float>;

//#define REMESHER_VERBATIM

#define TOOLS_ENABLE_PROFILER // enable profiler

#include "../BAEMM.hpp"

int main(int argc, const char * argv[])
{
    const char * homedir = getenv("HOME");

    if( homedir == nullptr)
    {
        homedir = getpwuid(getuid())->pw_dir;
    }
    std::string path ( homedir );
    
//    std::string file_name = path + "/github/BAEMM/Meshes/TorusMesh_00153600T.txt";
    std::string file_name = path + "/github/BAEMM/Meshes/TorusMesh_00038400T.txt";
//    std::string file_name = path + "/github/BAEMM/Meshes/TorusMesh_00009600T.txt";
//    std::string file_name = path + "/github/BAEMM/Meshes/TorusMesh_00000600T.txt";
    
    

    using namespace Repulsor;
    using namespace Tensors;
    using namespace Tools;
    using namespace BAEMM;
    

    Profiler::Clear( path );

    
    int thread_count = 8;
//    int thread_count = 1;
    omp_set_num_threads(thread_count);
    
    using Mesh_T = SimplicialMeshBase<Real,Int,Real,Real>;
    
    // Create a factory that can make instances of SimplicialMesh with domain dimension in the range 2,...,2 and ambient dimension in the range 3,...,3.
    SimplicialMesh_Factory<Mesh_T,2,2,3,3> mesh_factory;
    
    dump(file_name);
    
    std::unique_ptr<Mesh_T> M_ptr = mesh_factory.Make_FromFile<Real,Int,Real,Real>(file_name, thread_count);

    Mesh_T & M = *M_ptr;  // I don't like pointers. Give me a reference.

    print("");
    print("");

    static constexpr UInt n_waves = 32;
    
    static constexpr UInt i_blk   = 4;
    static constexpr UInt j_blk   = 2;

    Float kappa = 2.;
    Float kappa_step = 0.1;
    
    NS::AutoreleasePool* p_pool = NS::AutoreleasePool::alloc()->init();
    
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    
    Helmholtz_AoS<Float,UInt> H_AoS(
        M.VertexCoordinates().data(), M.VertexCount(),
        M.Simplices().data(),         M.SimplexCount()
    );
    
    const UInt n = M.SimplexCount();
    
    Tensor2<Complex,Int>  X_True ( n, n_waves );
    Tensor2<Complex,Int>  X      ( n, n_waves );
    Tensor2<Complex,Int>  Y      ( n, n_waves );
    Tensor2<Complex,Int>  Z      ( n, n_waves );

    Tensor2<Float,Int> Re_Y      ( n, n_waves );
    Tensor2<Float,Int> Im_Y      ( n, n_waves );
    

    const UInt size = n * n_waves * sizeof(Float);
    
    MTL::Buffer * Re_X_Metal = device->newBuffer(size, MTL::ResourceStorageModeManaged);
    MTL::Buffer * Im_X_Metal = device->newBuffer(size, MTL::ResourceStorageModeManaged);
    MTL::Buffer * Re_Y_Metal = device->newBuffer(size, MTL::ResourceStorageModeManaged);
    MTL::Buffer * Im_Y_Metal = device->newBuffer(size, MTL::ResourceStorageModeManaged);
    
    auto print_error = [&] ( const auto & X )
    {
        Subtract( X_True, X, Z );
        valprint("error",Z.MaxNorm()/X_True.MaxNorm());
    };

    auto print_error_C = [&] ( const auto & Re_X_Metal, auto & Im_X_Metal )
    {
        X.Read(
            static_cast<Float*>( Re_X_Metal->contents() ),
            static_cast<Float*>( Im_X_Metal->contents() )
        );
        print_error(X);
        Subtract( X_True, X, Z );
    };
                                
    Re_Y.Random();
    Im_Y.Random();
    
    Y.Read( Re_Y.data(), Im_Y.data() );
                                
    Y.Write(
        static_cast<Float*>( Re_Y_Metal->contents() ),
        static_cast<Float*>( Im_Y_Metal->contents() )
    );
    Re_Y_Metal->didModifyRange({0,size});
    Im_Y_Metal->didModifyRange({0,size});
    
    
    valprint("n      ", n       );
    valprint("n_waves", n_waves );
    valprint("i_blk  ", i_blk   );
    valprint("j_blk  ", j_blk   );
    
//    Float dummy (0);
//
//    tic("AoS naive");
//    for( UInt k = 0; k < n_waves; ++k )
//    {
//        H_AoS.Neumann_to_Dirichlet<1,false>(
//            Y.data(), X.data(), kappa + k * kappa_step, kappa_step, thread_count
//        );
//    }
//    toc("AoS naive");
//    dummy += X.MaxNorm();
//    dump(dummy);
//
//    tic("AoS naive");
//    for( UInt k = 0; k < n_waves; ++k )
//    {
//        H_AoS.Neumann_to_Dirichlet_Blocked<i_blk,j_blk,1,false>(
//            Y.data(), X.data(), kappa + k * kappa_step, kappa_step, thread_count
//        );
//    }
//    toc("AoS naive");
//    dummy += X.MaxNorm();
//    dump(dummy);
//
//    print("");

    
    H_AoS.Neumann_to_Dirichlet_Blocked<i_blk,j_blk,n_waves,false>(
        Y.data(), X_True.data(), kappa, kappa_step, thread_count
    );
    
//    H_AoS.Neumann_to_Dirichlet<n_waves,false>(
//        Y.data(), X.data(), kappa, kappa_step, thread_count
//    );
//    print_error(X);
    
    print("");
    
    Helmholtz_Metal H_Metal (
        device,
        M.VertexCoordinates().data(), M.VertexCount(),
        M.Simplices().data(),         M.SimplexCount()
    );
    
    for( UInt k = 0; k < 4; ++k )
    {
        H_Metal.Neumann_to_Dirichlet(
            Re_Y_Metal, Im_Y_Metal, Re_X_Metal, Im_X_Metal, kappa, kappa_step, 64, n_waves
        );
        print_error_C(Re_X_Metal,Im_X_Metal);
    }
    
    p_pool->release();

    return 0;
}
