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
#include "Repulsor.hpp"


using namespace Tools;
using namespace Tensors;

template<typename T>
T * ToPtr( MTL::Buffer * a )
{
    return reinterpret_cast<T*>(a->contents());
}


// We have to toggle which domain dimensions and ambient dimensions shall be supported by runtime polymorphism before we load Repulsor.hpp
// Bou can activate everything you want, but compile times might increase substatially.
using Int           =  Int32;
using UInt          = UInt32;
using ExtInt        =  Int64;

using Real          = Real64;
using Float         = Real32;
using Complex       = std::complex<Float>;

using Metal_Float   = Real32;
using Metal_Complex = std::complex<Metal_Float>;


int main(int argc, const char * argv[])
{
    using namespace Repulsor;
    using namespace Tensors;
    using namespace Tools;
//    using namespace BAEMM;

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

    int OMP_thread_count = 8;
//    int OMP_thread_count = 1;
    omp_set_num_threads(OMP_thread_count);

    using Mesh_T = SimplicialMeshBase<Real,Int,Real,Real>;

    // Create a factory that can make instances of SimplicialMesh with domain dimension in the range 2,...,2 and ambient dimension in the range 3,...,3.
    SimplicialMesh_Factory<Mesh_T,2,2,3,3> mesh_factory;

    dump(file_name);

    std::unique_ptr<Mesh_T> M_ptr = mesh_factory.Make_FromFile<Real,Int,Real,Real>(file_name, OMP_thread_count);

    Mesh_T & M = *M_ptr;  // I don't like pointers. Give me a reference.

    print("");
    print("");

    const UInt n = M.SimplexCount();
    static constexpr uint wave_count       = 64;
    static constexpr uint wave_chunk_size  = 16;
    static constexpr uint wave_chunk_count = wave_count / wave_chunk_size;
    static constexpr uint block_size       = 32;

    constexpr Float kappa = 2.;

    std::vector<Float> kappa_list (wave_chunk_count, kappa);

    std::array<Complex,4> coeff_0 {
        Complex(0.0f,0.0f),
        Complex(1.0f,0.0f),
        Complex(0.0f,0.0f),
        Complex(0.0f,0.0f)
    };

    std::array<Complex,4> coeff_1 {
        Complex(1.0f,0.0f),
        Complex(0.1f,0.2f),
        Complex(2.0f,-1.0f),
        Complex(-0.5f,0.4f)
    };

    const uint n_rounded          = RoundUpTo( n, block_size      );
    const uint wave_count_rounded = RoundUpTo( n, wave_chunk_size );
    dump(n);
    dump(n_rounded);
    dump(wave_count);
    dump(wave_count_rounded);

    NS::AutoreleasePool * p_pool = NS::AutoreleasePool::alloc()->init();

    MTL::Device* device = MTL::CreateSystemDefaultDevice();

    

    Tensor2<Complex,Int>  C_True ( n, wave_count );
    Tensor2<Complex,Int>  C      ( n, wave_count );
    Tensor2<Complex,Int>  B      ( n, wave_count );
    Tensor2<Complex,Int>  Z      ( n, wave_count );

    Tensor2<Float,Int>    Re_B   ( n, wave_count );
    Tensor2<Float,Int>    Im_B   ( n, wave_count );

    auto print_error = [&] ( const auto & C )
    {
        Subtract( C_True, C, Z );
        valprint("Error",Z.MaxNorm()/C_True.MaxNorm());
    };

    Re_B.Random();
    Im_B.Random();

    B.Read( Re_B.data(), Im_B.data() );

    print("");

    print("");
    print("Preparing Helmholtz classes");
    print("");

    BAEMM::Helmholtz_CPU H_CPU(
        M.VertexCoordinates().data(), M.VertexCount(),
        M.Simplices().data(),         M.SimplexCount(),
        OMP_thread_count
    );
    H_CPU.SetWaveChunkSize(wave_chunk_size);

    BAEMM::Helmholtz_Metal H_Metal (
        device,
        M.VertexCoordinates().data(), M.VertexCount(),
        M.Simplices().data(),         M.SimplexCount(),
        OMP_thread_count
    );
    H_Metal.SetWaveChunkSize(wave_chunk_size);

    print("");
    print("Single layer operator");
    print("");

    H_CPU.LoadCoefficients(coeff_0);
    H_CPU.ReadB ( B.data(), wave_count );
    H_CPU.BoundaryOperatorKernel_C(kappa_list);
    H_CPU.WriteC( C_True.data(), wave_count );

    H_Metal.LoadCoefficients(coeff_0);
    H_Metal.ReadB ( B.data(), wave_count );
    H_Metal.BoundaryOperatorKernel_C(kappa_list);
    H_Metal.WriteC( C.data(), wave_count );
    print_error(C);

    print("");
    print("General operator");
    print("");

    H_CPU.LoadCoefficients(coeff_1);
    H_CPU.ReadB ( B.data(), wave_count );
    H_CPU.BoundaryOperatorKernel_C(kappa_list);
    H_CPU.WriteC( C_True.data(), wave_count );
    dump( C_True.MaxNorm() );

    H_Metal.LoadCoefficients(coeff_1);
    H_Metal.ReadB ( B.data(), wave_count );
    H_Metal.BoundaryOperatorKernel_C(kappa_list);
    H_Metal.WriteC( C.data(), wave_count );
    print_error(C);

    print("");
    print("From vertices");
    print("");

    
    
    
    Tensor2<Complex,Int> X       ( H_Metal.VertexCount(), wave_count);
    Tensor2<Real   ,Int> Re_X    ( H_Metal.VertexCount(), wave_count);
    Tensor2<Real   ,Int> Im_X    ( H_Metal.VertexCount(), wave_count);
    Tensor2<Complex,Int> Y_CPU   ( H_Metal.VertexCount(), wave_count);
    Tensor2<Complex,Int> Y_Metal ( H_Metal.VertexCount(), wave_count);

    Re_X.Random(8);
    Im_X.Random(8);
    
    X.Read( Re_X.data(), Im_X.data() );

    H_CPU.ApplyBoundaryOperators_PL(
        Complex(1), X.data(),       wave_count,
        Complex(0), Y_CPU.data(),   wave_count,
        kappa_list, coeff_1,        wave_count
    );
    
    H_Metal.ApplyBoundaryOperators_PL(
        Complex(1), X.data(),       wave_count,
        Complex(0), Y_Metal.data(), wave_count,
        kappa_list, coeff_1,        wave_count
    );
    
    {
        Float error = 0;
        for( Int i = 0; i < H_Metal.VertexCount(); ++i )
        {
            for( Int k = 0; k < wave_count; ++k )
            {
                error = std::max( error, std::abs( Y_CPU(i,k) - Y_Metal(i,k) ) );
            }
        }
        
        valprint("Error", error / Y_CPU.MaxNorm() );
    }
    
    p_pool->release();

    return 0;
}
