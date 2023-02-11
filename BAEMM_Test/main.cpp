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
using Int           = int;
using Real          = double;
using Complex       = std::complex<Real>;


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

    static constexpr uint wave_count       = 64;
    static constexpr uint wave_chunk_size  = 16;
    static constexpr uint wave_chunk_count = wave_count / wave_chunk_size;
//    static constexpr uint block_size       = 64;


    print("");
    print("Preparing Helmholtz classes");
    print("");

    BAEMM::Helmholtz_CPU H_CPU(
        M.VertexCoordinates().data(), M.VertexCount(),
        M.Simplices().data(),         M.SimplexCount(),
        OMP_thread_count
    );
    H_CPU.SetWaveChunkSize(wave_chunk_size);
    
    // Create an object that handles GPU acces via Metal.
    
    // Some pool to handle reference-counted objects associated to Metal (namespace MTL).
    NS::AutoreleasePool * p_pool = NS::AutoreleasePool::alloc()->init();

    // Request the GPU device.
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    
    BAEMM::Helmholtz_Metal H_Metal (
        device,
        M.VertexCoordinates().data(),  // pointer to an array of doubles or floats
        M.VertexCount(),               // number of vertices
        M.Simplices().data(),          // pointer to an array of ints or long ints
        M.SimplexCount(),              // number of simplices
        OMP_thread_count               // number of OpenMP threads to use.
    );
    H_Metal.SetWaveChunkSize(wave_chunk_size);


    // Some matrices to hold data.
    Tensor2<Complex,Int> X     ( H_Metal.VertexCount(), wave_count);
    Tensor2<Complex,Int> Y_CPU ( H_Metal.VertexCount(), wave_count);
    Tensor2<Complex,Int> Y     ( H_Metal.VertexCount(), wave_count);
    
    // Generate some random input data.
    X.Random( 8 );
    // Loading a buffer would also work
    // X.Read( some_buffer );
    
    
    // Two real factors
    const Complex alpha = 1;
    const Complex beta  = 0;
    
    // Prepare wave numbers (all equal to 2 in this example).
    constexpr Real kappa = 2.;

    std::vector<Real> kappa_list (wave_chunk_count, kappa);
    
    // Set the coefficients for the operators
    std::array<Complex,4> coeff {
        Complex(0.0f,0.0f), // coefficient of mass matrix
        Complex(1.0f,0.0f), // coefficient of single layer op
        Complex(0.0f,0.0f), // coefficient of double layer op
        Complex(0.0f,0.0f)  // coefficient of adjoint double layer op
    };

    const Int ldX = X.Dimension(1);
    const Int ldY = Y.Dimension(1);

    H_CPU.ApplyBoundaryOperators_PL(
        alpha,      X.data(),       ldX,
        beta,       Y_CPU.data(),   ldY,
        kappa_list, coeff,          wave_count
    );
    
    // Compute Y = alpha * A * X + beta * Y
    // where A =   coeff[0] * [mass]
    //           + coeff[1] * [single layer op]
    //           + coeff[2] * [double layer op]
    //           + coeff[3] * [adjoint double layer op]
    
    H_Metal.ApplyBoundaryOperators_PL(
        alpha,
        X.data(),       // pointer to complex floating point type
        ldX,            // "leading dimension": number of columns in buffer X
        beta,
        Y.data(),       // pointer to complex floating point type
        ldY,            // "leading dimension": number of columns in buffer Y
        kappa_list,     // List of wave numbers;
        coeff,
        wave_count      // number of waves to process; wave_count <= ldX and wave_count <= dY
    );
    
    // Check the correctness against CPU implementation (which might also be wrong!!!)
    const Real error = RelativeMaxError(Y_CPU,Y);
    valprint("Error", error );

    
    
    // Destruct the pool for managing Metal's reference counted pointers.
    p_pool->release();
    
    return 0;
}
