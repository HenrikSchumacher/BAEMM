#include <iostream> 
#include <fstream> 
#include <cblas.h>
#include "read.hpp"

using Complex = std::complex<Real>;
using LInt = long long;
using SReal = Real;
using ExtReal = Real;

using namespace Tools;
using namespace Tensors;
using namespace Repulsor;

int main()
{
    BAEMM::Helmholtz_OpenCL H = read_OpenCL("/github/BAEMM/Meshes/Sphere_00040560T.txt");
    // BAEMM::Helmholtz_CPU H_CPU = read_CPU("/github/BAEMM/Meshes/TorusMesh_00153600T.txt");
    
    Int n = H.VertexCount();
    Int m = H.GetMeasCount();
    const Int wave_count = 16;
    constexpr Int wave_chunk_size = 16;
    constexpr Int wave_chunk_count = wave_count/wave_chunk_size;
    Complex* B = (Complex*)malloc(16 * m * sizeof(Complex));
    Complex* C = (Complex*)malloc(16 * m * sizeof(Complex));

    Int thread_count = 16;

    // using namespace Tensors;
    // using namespace Tools;

    // // const std::string path = "/HOME1/users/guests/jannr/github/BAEMM/Meshes/TorusMesh_00038400T.txt";
    // // std::string file_name = path;
    // // Tensor2<Real, Int> coords;
    // // Tensor2<Int, Int> simplices;

    // // ReadFromFile<Real, Int>(file_name, coords, simplices);

    Real * kappa = (Real*)malloc(wave_chunk_count * sizeof(Real));
    Real* inc = (Real*)malloc(wave_chunk_size * 3 * sizeof(Real));
    // Complex * coeff = (Complex*)malloc(4 * wave_chunk_count * sizeof(Complex));
    Complex * wave_coeff = (Complex*)malloc(4 * wave_chunk_count * sizeof(Complex));

    // for(Int i = 0 ; i < wave_chunk_count ; i++)
    // {
    //     coeff[4 * i + 0] = 0.0f;
    //     coeff[4 * i + 1] = 0.0f;
    //     coeff[4 * i + 2] = 0.0f;
    //     coeff[4 * i + 3] = 1.0f;
    // }

    for(Int i = 0 ; i < wave_chunk_count ; i++)
    {
        wave_coeff[4 * i + 0] = 0.0f;
        wave_coeff[4 * i + 1] = 1.0f;
        wave_coeff[4 * i + 2] = 0.0f;
        wave_coeff[4 * i + 3] = 0.0f;
    }

    for (int i = 0 ; i < wave_chunk_count; i++)
    {
        kappa[i] = Scalar::Pi<Real>;
    }

    // Real* C = (Real*)malloc(3 * n * sizeof(Real));

    // for (int i = 0 ; i < 4; i++)
    // {
    //     inc[12*i + 0] = 1.0f;
    //     inc[12*i + 1] = 0.0f;
    //     inc[12*i + 2] = 0.0f;
    //     inc[12*i + 3] = 0.0f;
    //     inc[12*i + 4] = 1.0f;
    //     inc[12*i + 5] = 0.0f;
    //     inc[12*i + 6] = 0.0f;
    //     inc[12*i + 7] = 0.0f;
    //     inc[12*i + 8] = 1.0f;
    //     inc[12*i + 9] = 1/std::sqrt(3.0f);
    //     inc[12*i + 10] = 1/std::sqrt(3.0f);
    //     inc[12*i + 11] = 1/std::sqrt(3.0f);
    // }

    H.UseDiagonal(true);

    Real cg_tol = static_cast<Real>(0.00001);
    Real gmres_tol = static_cast<Real>(0.0001);

    Complex* neumann_data_scat_ptr = NULL;

    // const Real* B = H.VertexCoordinates();
    for (int i = 0; i < 16 * m; i++)
    {
        B[i] = Complex(1.0f,0.0f);
    }
    // H.CreateIncidentWave_PL(Complex(1.0f,0.0f), inc, wave_chunk_size,
    //                         Complex(0.0f,0.0f), C, wave_count,
    //                         kappa, wave_coeff, wave_count, wave_chunk_size,
    //                         BAEMM::Helmholtz_OpenCL::WaveType::Plane
    //                         );
    // H.ApplyMassInverse<wave_count>(C,B,wave_count,cg_tol);

    BAEMM::Helmholtz_OpenCL::kernel_list list = H.LoadKernel(kappa,coeff,wave_count,wave_chunk_size);                        
    // tic("FF");
    // for (Int i = 0 ; i < 10; i++)
    // {
        H.ApplyBoundaryOperators_PL(
                        wave_count, Complex(1.0f,0.0f),B,Complex(0.0f,0.0f),C
                        );
    // }
    // toc("FF");

    H.DestroyKernel(&list);

    Real error = 0.0f;
    Complex a = Complex(0.0f,-1/(2*kappa[0]));
    a *= std::exp(Complex(0.0f,2*kappa[0]))-Complex(1.0f,0.0f);

    for(int i = 0; i < n ; i++)
    {
        for(int j = 0; j < 16 ; j++)
        {
            Real e = std::abs(C[i + 16*j] -a);
            if (e > error)
            {
                error = e;
            }
        }
    }
    std::cout << error << std::endl;
 

    std::ofstream fout_r("data_real.txt");
    // std::ofstream fout_i("data_imag.txt");
    if(fout_r.is_open() && fout_r.is_open())
	{
		for(int i = 0; i < n ; i++)
		{
            for(int j = 0; j < 3 ; j++)
            {
                fout_r << C[i * 3 + j] << " "; 
                // fout_i << C[i * wave_count + j].imag() << " "; 
            }
            fout_r << "\n";
            // fout_i << "\n";
		}
	}            

    free(B);
    free(C);
    free(inc);
    free(kappa);
    // free(coeff);
    free(wave_coeff);
    return 0;
}