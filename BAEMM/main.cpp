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
    using Int     =  int32_t;
    using UInt    = uint32_t;
    using ExtInt  =  int64_t;
    
    using Real    = float64_t;
    using Float   = float32_t;
    using Complex = std::complex<Float>;
    

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
    static constexpr uint n_waves = 32;
    const uint simd_count = 32;
    const uint simd_size  = 32;

    static constexpr uint i_blk   = 4;
    static constexpr uint j_blk   = 2;

    Float kappa = 2.;
    Float kappa_step = 0.1;

    valprint("n      ", n       );
    valprint("n_waves", n_waves );

    NS::AutoreleasePool* p_pool = NS::AutoreleasePool::alloc()->init();

    MTL::Device* device = MTL::CreateSystemDefaultDevice();

    Helmholtz_AoS<Float,UInt> H_AoS(
        M.VertexCoordinates().data(), M.VertexCount(),
        M.Simplices().data(),         M.SimplexCount()
    );
    
//    Tensor2<Complex,Int>  A      ( n, n );
//    Tensor2<Complex,Int>  A_True ( n, n );
    
    Tensor2<Complex,Int>  C_True ( n, n_waves );
    Tensor2<Complex,Int>  C      ( n, n_waves );
    Tensor2<Complex,Int>  B      ( n, n_waves );
    Tensor2<Complex,Int>  Z      ( n, n_waves );
    

    Tensor2<Float,Int> Re_B      ( n, n_waves );
    Tensor2<Float,Int> Im_B      ( n, n_waves );

    
    constexpr UInt round_to      = 64;
    const     UInt n_rounded     = RoundUpTo( n, round_to );
    dump(n);
    dump(n_rounded);

    const UInt size = n_rounded * n_waves * sizeof(Float);

    MTL::Buffer * Re_C_Metal = device->newBuffer(size, MTL::ResourceStorageModeManaged);
    MTL::Buffer * Im_C_Metal = device->newBuffer(size, MTL::ResourceStorageModeManaged);
    MTL::Buffer * Re_B_Metal = device->newBuffer(size, MTL::ResourceStorageModeManaged);
    MTL::Buffer * Im_B_Metal = device->newBuffer(size, MTL::ResourceStorageModeManaged);
    
//    MTL::Buffer * Re_A_Metal = device->newBuffer(n*n*sizeof(Float), MTL::ResourceStorageModeManaged);
//    MTL::Buffer * Im_A_Metal = device->newBuffer(n*n*sizeof(Float), MTL::ResourceStorageModeManaged);

//    MTL::Buffer * C_Metal    = device->newBuffer(2*size, MTL::ResourceStorageModeManaged);
//    MTL::Buffer * B_Metal    = device->newBuffer(2*size, MTL::ResourceStorageModeManaged);

    auto print_error = [&] ( const auto & C )
    {
        Subtract( C_True, C, Z );
        valprint("Error",Z.MaxNorm()/C_True.MaxNorm());
    };

    auto print_error_ReIm = [&] ( const auto & Re_C_Metal, auto & Im_C_Metal )
    {
        C.Read( ToPtr<Float>(Re_C_Metal), ToPtr<Float>(Im_C_Metal) );
        print_error(C);
    };

//    auto print_error_C = [&] ( const auto & C_Metal )
//    {
//        C.Read( reinterpret_cast<Complex*>( C_Metal->contents() ) );
//        print_error(C);
//        Subtract( C_True, C, Z );
//    };

    Re_B.Fill(1);
//    Re_B.Random();
//    Im_B.Random();

    Im_B.SetZero();

    B.Read( Re_B.data(), Im_B.data() );

    B.Write( ToPtr<Float>( Re_B_Metal ), ToPtr<Float>( Im_B_Metal ) );
    Re_B_Metal->didModifyRange({0,size});
    Im_B_Metal->didModifyRange({0,size});

    H_AoS.Neumann_to_Dirichlet_Blocked<i_blk,j_blk,n_waves,false>(
        B.data(), C_True.data(), kappa, kappa_step, thread_count
    );

    dump( C_True.MaxNorm());
    
//    H_AoS.Neumann_to_Dirichlet_Assembled<i_blk,j_blk,n_waves,false>(
//        A_True.data(), kappa, kappa_step, thread_count
//    );
    
//
//    H_AoS.Neumann_to_Dirichlet_C<n_waves,false>(
//        ToPtr<Float>(Re_B_Metal), ToPtr<Float>(Im_B_Metal),
//        ToPtr<Float>(Re_C_Metal), ToPtr<Float>(Im_C_Metal),
//        kappa, kappa_step, thread_count
//    );
//    print_error_ReIm(Re_C_Metal,Im_C_Metal);

    print("");

    Helmholtz_Metal H_Metal (
        device,
        M.VertexCoordinates().data(), M.VertexCount(),
        M.Simplices().data(),         M.SimplexCount()
    );

    H_Metal.Neumann_to_Dirichlet(
        Re_B_Metal, Im_B_Metal, Re_C_Metal, Im_C_Metal, kappa, kappa_step, 64, n_waves
    );
    print_error_ReIm(Re_C_Metal,Im_C_Metal);

    H_Metal.Neumann_to_Dirichlet2(
        Re_B_Metal, Im_B_Metal, Re_C_Metal, Im_C_Metal, kappa, kappa_step, n_waves, simd_count, simd_size
    );
    print_error_ReIm(Re_C_Metal,Im_C_Metal);

    H_Metal.Neumann_to_Dirichlet3(
        Re_B_Metal, Im_B_Metal, Re_C_Metal, Im_C_Metal, kappa, kappa_step, n_waves, 32 );
    print_error_ReIm(Re_C_Metal,Im_C_Metal);

    H_Metal.Neumann_to_Dirichlet(
        Re_B_Metal, Im_B_Metal, Re_C_Metal, Im_C_Metal, kappa, kappa_step, 64, n_waves
    );
    print_error_ReIm(Re_C_Metal,Im_C_Metal);

    H_Metal.Neumann_to_Dirichlet2(
        Re_B_Metal, Im_B_Metal, Re_C_Metal, Im_C_Metal, kappa, kappa_step, n_waves, simd_count, simd_size
    );
    print_error_ReIm(Re_C_Metal,Im_C_Metal);

    H_Metal.Neumann_to_Dirichlet3(
        Re_B_Metal, Im_B_Metal, Re_C_Metal, Im_C_Metal, kappa, kappa_step, n_waves, 32
    );
    print_error_ReIm(Re_C_Metal,Im_C_Metal);


    C.Read(
        reinterpret_cast<Float*>( Re_C_Metal->contents() ),
        reinterpret_cast<Float*>( Im_C_Metal->contents() )
    );
    
    std::ofstream fileC ("/Users/Henrik/C.txt");
    fileC << C;
    
    std::ofstream fileATrue ("/Users/Henrik/CTrue.txt");
    fileATrue << C_True;

    
//    Helmholtz_Metal H_Metal (
//        device,
//        M.VertexCoordinates().data(), M.VertexCount(),
//        M.Simplices().data(),         M.SimplexCount()
//    );
    
//    constexpr UInt threadgroup_size  = 1024;
//    constexpr UInt threadgroup_count = 16 * 2;
//    
//    
//    const UInt m = UInt(1024) * UInt(1024) * UInt(1024);
//    
//    print("Allocating "+ToString( double(m * sizeof(Float)) / (std::pow(2.,20)) )+" MB.");
//    MTL::Buffer * a = device->newBuffer(m * sizeof(Float), MTL::ResourceStorageModeManaged);
////    MTL::Buffer * b = device->newBuffer(thread_count_ * sizeof(float), MTL::ResourceStorageModeManaged);
//    print("Allocating "+ToString( double(threadgroup_count * sizeof(Float)) / (std::pow(2.,20)))+" MB.");
//    MTL::Buffer * c = device->newBuffer(threadgroup_count * sizeof(Float), MTL::ResourceStorageModeManaged);
//
////    fill_buffer( reinterpret_cast<float*>(a->contents()), m,                 static_cast<float>(1) );
//    
//    std::random_device r;
//    std::default_random_engine engine ( r() );
//    std::uniform_real_distribution<Float> unif(static_cast<Float>(0.),static_cast<Float>(1.));
//    
//    mut<Float> a_ = reinterpret_cast<Float*>(a->contents());
//    for( UInt i = 0; i < m; ++i )
//    {
//        a_[i] = unif(engine);
//    }
////    fill_buffer( a_, m,  static_cast<Float>(.1) );
//
////    fill_buffer( reinterpret_cast<float*>(b->contents()), thread_count_,     static_cast<float>(0) );
//    fill_buffer( reinterpret_cast<Float*>(c->contents()), threadgroup_count, static_cast<Float>(1) );
//    
//    for( UInt k = 0; k < 8; ++k )
//    {
//        H_Metal.AddReduce(a, c, m, threadgroup_count, threadgroup_size );
//    }
//    
//    dump(m);
//    
//    tic("a_sum");
//    double a_sum = 0;
//    #pragma omp parallel for num_threads( thread_count) reduction( + : a_sum )
//    for( UInt i = 0; i < m; ++i )
//    {
//        a_sum += a_[i];
//    }
//    toc("a_sum");
//    dump(a_sum);
//    
//    ptr<Float> c_ = reinterpret_cast<ptr<Float>>(c->contents());
//    
////    print( ToString( c_, threadgroup_count, 16 ) );
//    
//    double c_sum = 0;
//    for( UInt i = 0; i < threadgroup_count; ++i )
//    {
//        c_sum += c_[i];
//    }
//    dump(c_sum);
    
    p_pool->release();

    return 0;
}





//    B.Write( reinterpret_cast<Complex*>( B_Metal->contents() ) );
//    B_Metal->didModifyRange({0,2*size});


//    Float dummy (0);
//
//    tic("AoS naive");
//    for( UInt k = 0; k < n_waves; ++k )
//    {
//        H_AoS.Neumann_to_Dirichlet<1,false>(
//            B.data(), C.data(), kappa + k * kappa_step, kappa_step, thread_count
//        );
//    }
//    toc("AoS naive");
//    dummy += C.MaxNorm();
//    dump(dummy);
//
//    tic("AoS naive");
//    for( UInt k = 0; k < n_waves; ++k )
//    {
//        H_AoS.Neumann_to_Dirichlet_Blocked<i_blk,j_blk,1,false>(
//            B.data(), C.data(), kappa + k * kappa_step, kappa_step, thread_count
//        );
//    }
//    toc("AoS naive");
//    dummy += C.MaxNorm();
//    dump(dummy);
//
//    print("");
