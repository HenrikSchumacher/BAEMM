#include <iostream>

#include <sys/types.h>
#include <pwd.h>
#include <complex>

//#define OBJC_DEBUG_MISSING_POOLS = YES

#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION


#define TOOLS_ENABLE_PROFILER // enable profiler

#include "../Helmholtz_CPU.hpp"
#include "../Helmholtz_Metal.hpp"

using namespace Tools;
using namespace Tensors;

template<typename T>
T * ToPtr( MTL::Buffer * a )
{
    return reinterpret_cast<T*>(a->contents());
}


template<typename Real, typename Int>
void ReadFromFile(
    const std::string & file_name,
    Tensor2<Real,Int> & coords,
    Tensor2<Int,Int>  & simplices
)
{
    print("Reading mesh from file "+file_name+".");
    
    std::ifstream s (file_name);
    
    if( !s.good() )
    {
        eprint("ReadFromFile: File "+file_name+" could not be opened.");
        
        return;
    }
    
    std::string str;
    Int amb_dim;
    Int dom_dim;
    Int vertex_count;
    Int simplex_count;
    s >> str;
    s >> dom_dim;
    valprint("dom_dim",dom_dim);
    s >> str;
    s >> amb_dim;
    valprint("amb_dim",amb_dim);
    s >> str;
    s >> vertex_count;
    valprint("vertex_count",vertex_count);
    s >> str;
    s >> simplex_count;
    valprint("simplex_count",simplex_count);
    
    const Int simplex_size = dom_dim+1;
    
    valprint("simplex_size",simplex_size);
    
    coords    = Tensor2<Real,Int>(vertex_count, amb_dim     );
    simplices = Tensor2<Int, Int>(simplex_count,simplex_size);
    
    mut<Real> V = coords.data();
    mut<Int>     S = simplices.data();
    
    
    for( Int i = 0; i < vertex_count; ++i )
    {
        for( Int k = 0; k < amb_dim; ++k )
        {
            s >> V[amb_dim * i + k];
        }
    }
    
    for( Int i = 0; i < simplex_count; ++i )
    {
        for( Int k = 0; k < simplex_size; ++k )
        {
            s >> S[simplex_size * i + k];
        }
    }
}

// We have to toggle which domain dimensions and ambient dimensions shall be supported by runtime polymorphism before we load Repulsor.hpp
// Bou can activate everything you want, but compile times might increase substatially.
using Int           = long long;
using Real          = double;
using Complex       = std::complex<Real>;


int main(int argc, const char * argv[])
{
    //    using namespace Repulsor;
    //    using namespace BAEMM;
    
    
    
    //Loading vertex coordinates and simplices from file.
    
    const char * homedir = getenv("HOME");
    
    if( homedir == nullptr)
    {
        homedir = getpwuid(getuid())->pw_dir;
    }
    std::string path ( homedir );
    
//        std::string file_name = path + "/github/BAEMM/Meshes/TorusMesh_00153600T.txt";
//        std::string file_name = path + "/github/BAEMM/Meshes/TorusMesh_00038400T.txt";
    std::string file_name = path + "/github/BAEMM/Meshes/TorusMesh_00009600T.txt";
//    std::string file_name = path + "/github/BAEMM/Meshes/TorusMesh_00000600T.txt";
    
    
    Tensor2<Real,Int> coords;
    Tensor2<Int ,Int> simplices;
    ReadFromFile(file_name, coords, simplices);
    
    print("");
    
    // Clear file used by the profiler.
    Profiler::Clear( path );
    
    print("");
    print("");
    
    int CPU_thread_count = 8;
    //    int CPU_thread_count = 1;
    omp_set_num_threads(CPU_thread_count);
    
    valprint("number of CPU threads",CPU_thread_count);
    
    print("");
    print("");
    
    static constexpr Int wave_count       = 16;
    static constexpr Int wave_chunk_size  = 16;
    static constexpr Int wave_chunk_count = wave_count / wave_chunk_size;
    
    
    print("");
    print("Preparing Helmholtz classes");
    print("");
    
    BAEMM::Helmholtz_CPU H_CPU (
        coords.data(),    coords.Dimension(0),
        simplices.data(), simplices.Dimension(0),
        CPU_thread_count
    );
    
    
    // Create an object that handles GPU access via Metal.
    
    // Some pool to handle reference-counted objects associated to Metal (namespace MTL).
    //    NS::SharedPtr<NS::AutoreleasePool> auto_pool
    //        = NS::TransferPtr( NS::AutoreleasePool::alloc()->init() );
    
    // Request the GPU device.
    //    NS::SharedPtr<MTL::Device> device =
    //        reinterpret_cast<MTL::Device *>( MTL::CopyAllDevices()->object(0) )
    //    );
    
    NS::SharedPtr<MTL::Device> device = NS::TransferPtr(
        reinterpret_cast<MTL::Device *>( MTL::CopyAllDevices()->object(0) )
    );
    
    BAEMM::Helmholtz_Metal H_Metal (
        device,
        coords.data(),              // pointer to an array of doubles or floats
        coords.Dimension(0),        // number of vertices
        simplices.data(),           // pointer to an array of ints or long ints
        simplices.Dimension(0),     // number of simplices
        CPU_thread_count            // number of CPU threads to use.
    );
    
    // Some matrices to hold data.
    Tensor2<Complex,Int> X     ( H_Metal.VertexCount(), wave_count );
    Tensor2<Complex,Int> Y_CPU ( H_Metal.VertexCount(), wave_count );
    Tensor2<Complex,Int> Y     ( H_Metal.VertexCount(), wave_count );
    
    // Generate some random input data.
    X.Random(CPU_thread_count);
    // Loading a buffer would also work
    // X.Read( some_buffer );
    
    
    // Prepare wave numbers (all equal to 2 in this example).
    constexpr Real kappa = 2.;
    
    Tensor1<Real,Int> kappa_list (wave_chunk_count, kappa);
    
    // Set the coefficients for the operators
    Tensor2<Complex,Int> coeff_list (wave_chunk_count,4);
    
    coeff_list.Random();
//    for( Int k = 0; k < wave_chunk_count; ++k )
//    {
//        coeff_list[k][0] = Complex(4.0f,1.2f); // coefficient of mass matrix
//        coeff_list[k][1] = Complex(0.4f,1.3f); // coefficient of single layer op
//        coeff_list[k][2] = Complex(1.2f,0.9f); // coefficient of double layer op
//        coeff_list[k][3] = Complex(1.0f,0.5f); // coefficient of adjoint double layer op
//    }
    
    
    // Two complex factors
    const Complex alpha = 1; // typically alpha = 1
    const Complex beta  = 0; // beta  = 0 means "overwrite Y", beta = 1  means "add-into Y"
    
    
    const Int ldX = X.Dimension(1);
    const Int ldY = Y.Dimension(1);

    const Complex * X_ptr = X.data();
          Complex * Y_ptr = Y.data();
    const Real    * kappa_ptr = kappa_list.data();
    const Complex * coeff_ptr = coeff_list.data();
    
    // Compute Y = alpha * A * X + beta * Y
    // where A =   coeff[0] * [mass]
    //           + coeff[1] * [single layer op]
    //           + coeff[2] * [double layer op]
    //           + coeff[3] * [adjoint double layer op]
    
    H_CPU.ApplyBoundaryOperators_PL(
        alpha,
        X.data(),           // pointer to complex floating point type
        ldX,                // "leading dimension": number of columns in buffer X
        beta,
        Y_CPU.data(),       // pointer to complex floating point type
        ldY,                // "leading dimension": number of columns in buffer Y
        kappa_list.data(),  // the wave numbers to use; 1 for each chunk
        coeff_list.data(),  // the coefficients to use; 4 for each chunk
        wave_count,         // how many right hand sides to process
        wave_chunk_size     // do it in chunks of size wave_chunk_size
    );

    

    
    // IMPORTANT:
    // 1. Real and Complex have to be of same precision
    // 2. type Real has to be real
    // 3. type Complex has to be complex
    //      e.g. Real = float and Complex = std::complex<float> would work.
    //      e.g. Real = double and Complex = std::complex<double> would work, too.
    // 4. X_ptr is array of size >= vertex_count * ldX
    // 5. Y_ptr is array of size >= vertex_count * ldY
    // 6. kappa_ptr is an array of size (wave_count + wave_chunk_size - 1) / wave_chunk_size
    // 7. coeff_ptr is an array of size 4 * (wave_count + wave_chunk_size - 1) / wave_chunk_size.
    // 8. wave_count <= ldX, ldY
    // 9. wave_chunk_size = 16 or a nearby power of 2 should work best.
    
    H_Metal.ApplyBoundaryOperators_PL(
        alpha,
        X_ptr,              // pointer to complex floating point type
        ldX,                // "leading dimension": number of columns in buffer X
        beta,
        Y_ptr,              // pointer to complex floating point type
        ldY,                // "leading dimension": number of columns in buffer Y
        kappa_ptr,          // the wave numbers to use; 1 for each chunk
        coeff_ptr,          // the coefficients to use; 4 for each chunk
        wave_count,         // how many right hand sides to process
        wave_chunk_size     // do it in chunks of size wave_chunk_size
    );
    
    // Check the correctness against CPU implementation (which might also be wrong!!!)
    {
        const Real error = RelativeMaxError(Y_CPU,Y);
        valprint("Error", error );
    }
    
    
    print("");
    print("Second run to factor-out one-time costs.");
    print("");
    
    tic("H_CPU.ApplyBoundaryOperators_PL");
    H_CPU.ApplyBoundaryOperators_PL(
        alpha, X.data(),     ldX,
        beta,  Y_CPU.data(), ldY,
        kappa_list.data(), coeff_list.data(), wave_count, wave_chunk_size
    );
    toc("H_CPU.ApplyBoundaryOperators_PL");
    
    tic("H_Metal.ApplyBoundaryOperators_PL");
    H_Metal.ApplyBoundaryOperators_PL(
        alpha, X.data(),     ldX,
        beta,  Y.data(),     ldY,
        kappa_list.data(), coeff_list.data(), wave_count, wave_chunk_size
    );
    toc("H_Metal.ApplyBoundaryOperators_PL");
    
    // Check the correctness against CPU implementation (which might also be wrong!!!)
    {
        const Real error = RelativeMaxError(Y_CPU,Y);
        valprint("Error", error );
    }
    
    // Set the coefficients for the operators
    
    Tensor2<Complex,Int> coeff_0 (wave_chunk_count,4);
    for( Int k = 0; k < wave_chunk_count; ++k )
    {
        coeff_0[k][0] = Complex(0.0f,0.0f);
        coeff_0[k][1] = Complex(1.0f,0.0f);
        coeff_0[k][2] = Complex(0.0f,0.0f);
        coeff_0[k][3] = Complex(0.0f,0.0f);
    }
    
    Tensor2<Complex,Int> coeff_1 (wave_chunk_count,4);
    for( Int k = 0; k < wave_chunk_count; ++k )
    {
        coeff_1[k][0] = Complex(0.0f,0.0f);
        coeff_1[k][1] = Complex(1.9f,0.0f);
        coeff_1[k][2] = Complex(0.0f,1.1f);
        coeff_1[k][3] = Complex(1.2f,0.0f);
    }
    
    Tensor2<Complex,Int> coeff_2 (wave_chunk_count,4);
    for( Int k = 0; k < wave_chunk_count; ++k )
    {
        coeff_2[k][0] = Complex(1.0f,1.0f);
        coeff_2[k][1] = Complex(1.9f,0.2f);
        coeff_2[k][2] = Complex(0.7f,1.1f);
        coeff_2[k][3] = Complex(1.2f,0.9f);
    }
    
    print("");
    print("");
    print("Checking dependence of runtime on number of nonzero coeffiencts.");
    print("");
    print("");
    print("One nonzero coefficients.");

    tic("H_Metal.ApplyBoundaryOperators_PL");
    H_Metal.ApplyBoundaryOperators_PL(
        alpha,      X.data(),  ldX,
        beta,       Y.data(),  ldY,
        kappa_list.data(), coeff_0.data(), wave_count, wave_chunk_size
    );
    toc("H_Metal.ApplyBoundaryOperators_PL");
    
    tic("H_Metal.ApplyBoundaryOperators_PL");
    H_Metal.ApplyBoundaryOperators_PL(
        alpha,      X.data(),  ldX,
        beta,       Y.data(),  ldY,
        kappa_list.data(), coeff_0.data(), wave_count, wave_chunk_size
    );
    toc("H_Metal.ApplyBoundaryOperators_PL");
    
    print("");
    print("");
    print("Three nonzero coefficients.");
    tic("H_Metal.ApplyBoundaryOperators_PL");
    H_Metal.ApplyBoundaryOperators_PL(
        alpha,      X.data(),  ldX,
        beta,       Y.data(),  ldY,
        kappa_list.data(), coeff_1.data(), wave_count, wave_chunk_size
    );
    toc("H_Metal.ApplyBoundaryOperators_PL");
    
    tic("H_Metal.ApplyBoundaryOperators_PL");
    H_Metal.ApplyBoundaryOperators_PL(
        alpha,      X.data(),  ldX,
        beta,       Y.data(),  ldY,
        kappa_list.data(), coeff_1.data(), wave_count, wave_chunk_size
    );
    toc("H_Metal.ApplyBoundaryOperators_PL");

    print("");
    print("");
    print("Six nonzero coefficients.");
    tic("H_Metal.ApplyBoundaryOperators_PL");
    H_Metal.ApplyBoundaryOperators_PL(
        alpha,      X.data(),  ldX,
        beta,       Y.data(),  ldY,
        kappa_list.data(), coeff_2.data(), wave_count, wave_chunk_size
    );
    toc("H_Metal.ApplyBoundaryOperators_PL");
    
    tic("H_Metal.ApplyBoundaryOperators_PL");
    H_Metal.ApplyBoundaryOperators_PL(
        alpha,      X.data(),  ldX,
        beta,       Y.data(),  ldY,
        kappa_list.data(), coeff_2.data(), wave_count, wave_chunk_size
    );
    toc("H_Metal.ApplyBoundaryOperators_PL");
    
    
    print("");
    print("");
    
    return 0;
}
