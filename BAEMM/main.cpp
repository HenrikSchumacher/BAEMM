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



//#define REMESHER_VERBATIM

#define TOOLS_ENABLE_PROFILER // enable profiler

#include "../BAEMM.hpp"

template<typename T>
T * ToPtr( MTL::Buffer * a )
{
    return reinterpret_cast<T*>(a->contents());
}


namespace BAEMM
{
    // We have to toggle which domain dimensions and ambient dimensions shall be supported by runtime polymorphism before we load Repulsor.hpp
    // Bou can activate everything you want, but compile times might increase substatially.
    using Int           =  int32_t;
    using UInt          = uint32_t;
    using ExtInt        =  int64_t;
    
    using Real          = float64_t;
    using Float         = float32_t;
    using Complex       = std::complex<Float>;
    
    using Metal_Float   = float32_t;
    using Metal_Complex = std::complex<Metal_Float>;
}



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
    
//    std::string file_name = path + "/github/BAEMM/Meshes/TorusMesh_00153600T.txt";
    std::string file_name = path + "/github/BAEMM/Meshes/TorusMesh_00038400T.txt";
//    std::string file_name = path + "/github/BAEMM/Meshes/TorusMesh_00009600T.txt";
//    std::string file_name = path + "/github/BAEMM/Meshes/TorusMesh_00000600T.txt";
    
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

    const UInt n = M.SimplexCount();
    static constexpr uint vec_size   =  1;
    static constexpr uint simd_size  = 32;
    static constexpr uint wave_count = 32;
    
//    static constexpr uint vec_size   =  4;
//    static constexpr uint simd_size  = 32;
//    static constexpr uint wave_count    = 32 * 2 * 4;
    
    static constexpr uint i_blk   = 4;
    static constexpr uint j_blk   = 2;

    Float kappa = 2.;

    valprint("n      ", n       );
    valprint("wave_count", wave_count );

    NS::AutoreleasePool* p_pool = NS::AutoreleasePool::alloc()->init();

    MTL::Device* device = MTL::CreateSystemDefaultDevice();

    Helmholtz_AoS<Float,UInt> H_AoS(
        M.VertexCoordinates().data(), M.VertexCount(),
        M.Simplices().data(),         M.SimplexCount()
    );
    
//    Tensor2<Complex,Int>  A      ( n, n );
//    Tensor2<Complex,Int>  A_True ( n, n );
    
    Tensor2<Complex,Int>  C_True ( n, wave_count );
    Tensor2<Complex,Int>  C      ( n, wave_count );
    Tensor2<Complex,Int>  B      ( n, wave_count );
    Tensor2<Complex,Int>  Z      ( n, wave_count );
    

    Tensor2<Float,Int> Re_B      ( n, wave_count );
    Tensor2<Float,Int> Im_B      ( n, wave_count );

    
    constexpr UInt round_to      = 64;
    const     UInt n_rounded     = RoundUpTo( n, round_to );
    dump(n);
    dump(n_rounded);

    const UInt size = n_rounded * wave_count * sizeof(Metal_Float);

    MTL::Buffer * Re_C_Metal = device->newBuffer(size, MTL::ResourceStorageModeManaged);
    MTL::Buffer * Im_C_Metal = device->newBuffer(size, MTL::ResourceStorageModeManaged);
    MTL::Buffer * Re_B_Metal = device->newBuffer(size, MTL::ResourceStorageModeManaged);
    MTL::Buffer * Im_B_Metal = device->newBuffer(size, MTL::ResourceStorageModeManaged);
    
//    MTL::Buffer * Re_A_Metal = device->newBuffer(n*n*sizeof(Float), MTL::ResourceStorageModeManaged);
//    MTL::Buffer * Im_A_Metal = device->newBuffer(n*n*sizeof(Float), MTL::ResourceStorageModeManaged);

    MTL::Buffer * C_Metal    = device->newBuffer(2*size, MTL::ResourceStorageModeManaged);
    MTL::Buffer * B_Metal    = device->newBuffer(2*size, MTL::ResourceStorageModeManaged);

    auto print_error = [&] ( const auto & C )
    {
        Subtract( C_True, C, Z );
        valprint("Error",Z.MaxNorm()/C_True.MaxNorm());
    };

    auto print_error_ReIm = [&] ( const auto & Re_C_Metal, auto & Im_C_Metal )
    {
        C.Read( ToPtr<Metal_Float>(Re_C_Metal), ToPtr<Metal_Float>(Im_C_Metal) );
        print_error(C);
    };

    auto print_error_C = [&] ( const auto & C_Metal )
    {
        C.Read( ToPtr<Metal_Complex>(C_Metal) );
        print_error(C);
        Subtract( C_True, C, Z );
    };

    Re_B.Random();
    Im_B.Random();


    B.Read( Re_B.data(), Im_B.data() );

    B.Write( ToPtr<Metal_Float>(Re_B_Metal), ToPtr<Metal_Float>(Im_B_Metal) );
    Re_B_Metal->didModifyRange({0,size});
    Im_B_Metal->didModifyRange({0,size});
    
    B.Write( ToPtr<Metal_Complex>(B_Metal)  );
    B_Metal->didModifyRange({0,2*size});

//    H_AoS.Neumann_to_Dirichlet_Blocked<i_blk,j_blk,wave_count,false>(
//        B.data(), C_True.data(), kappa, thread_count
//    );
//
//    dump( C_True.MaxNorm());

    print("");

    Helmholtz_Metal H_Metal (
        device,
        M.VertexCoordinates().data(), M.VertexCount(),
        M.Simplices().data(),         M.SimplexCount()
    );

    H_Metal.Neumann_to_Dirichlet(
        Re_B_Metal, Im_B_Metal, Re_C_Metal, Im_C_Metal, kappa, 64, wave_count
    );
    C_True.Read( ToPtr<Float>(Re_C_Metal), ToPtr<Float>(Im_C_Metal) );
    
    H_Metal.Neumann_to_Dirichlet(
        Re_B_Metal, Im_B_Metal, Re_C_Metal, Im_C_Metal, kappa, 64, wave_count
    );
    print_error_ReIm(Re_C_Metal,Im_C_Metal);
    
    print("");

    H_Metal.Neumann_to_Dirichlet2(
        Re_B_Metal, Im_B_Metal, Re_C_Metal, Im_C_Metal, kappa, wave_count, simd_size, vec_size
    );
    print_error_ReIm(Re_C_Metal,Im_C_Metal);
    
    H_Metal.Neumann_to_Dirichlet2(
        Re_B_Metal, Im_B_Metal, Re_C_Metal, Im_C_Metal, kappa, wave_count, simd_size, vec_size
    );
    print_error_ReIm(Re_C_Metal,Im_C_Metal);
    
    print("");
    
    H_Metal.Neumann_to_Dirichlet3(
        B_Metal, C_Metal, kappa, wave_count, simd_size );
    print_error_C(C_Metal);
    
    H_Metal.Neumann_to_Dirichlet3(
        B_Metal, C_Metal, kappa, wave_count, simd_size );
    print_error_C(C_Metal);
    
    
    H_Metal.Neumann_to_Dirichlet4(
        Re_B_Metal, Im_B_Metal, Re_C_Metal, Im_C_Metal, kappa, 1.f, 0.f, 0.f,
        wave_count, 64, 32, true
    );
    print_error_ReIm(Re_C_Metal,Im_C_Metal);
    
    H_Metal.Neumann_to_Dirichlet4(
        Re_B_Metal, Im_B_Metal, Re_C_Metal, Im_C_Metal, kappa, 1.f, 0.f, 0.f,
        wave_count, 64, 32, true
    );
    print_error_ReIm(Re_C_Metal,Im_C_Metal);
    
    H_Metal.Neumann_to_Dirichlet4(
        Re_B_Metal, Im_B_Metal, Re_C_Metal, Im_C_Metal, kappa, 1.f, 0.f, 0.f,
        wave_count, 64, 16, true
    );
    print_error_ReIm(Re_C_Metal,Im_C_Metal);
    
    H_Metal.Neumann_to_Dirichlet4(
        Re_B_Metal, Im_B_Metal, Re_C_Metal, Im_C_Metal, kappa, 1.f, 0.f, 0.f,
        wave_count, 64, 16, true
    );
    print_error_ReIm(Re_C_Metal,Im_C_Metal);
    
    
    p_pool->release();

    return 0;
}
