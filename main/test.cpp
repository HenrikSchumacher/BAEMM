#include <iostream> 
#include <fstream> 
#include <cblas.h>
#include "read.hpp"
#include "../Tensors/GMRES.hpp"
#include "../Tensors/ConjugateGradient.hpp"

using Complex = std::complex<float>;

int main()
{
    BAEMM::Helmholtz_OpenCL H_GPU = read_OpenCL("/github/BAEMM/Meshes/Sphere_00081920T.txt");
    // BAEMM::Helmholtz_CPU H_CPU = read_CPU("/github/BAEMM/Meshes/TorusMesh_00153600T.txt");
    
    Int n = H_GPU.VertexCount();
    const Int wave_count = 16;
    constexpr Int wave_chunk_size = 16;
    constexpr Int wave_chunk_count = wave_count/wave_chunk_size;

    Real* B = (Real*)malloc(3 * n * sizeof(Real));

    for (int i = 0; i < n; i++)
    {
        //B[i] = std::exp(Complex(0.0f,(float)i));
        // B[i] = 1.0f;
        // B[i] = 0.0f;
        // B[i] = 0.0f;
    }
    {
        using namespace Tensors;
        using namespace Tools;

        const std::string path = "/HOME1/users/guests/jannr/github/BAEMM/Meshes/Sphere_00081920T.txt";
        std::string file_name = path;
        Tensor2<Real, Int> coords;
        Tensor2<Int, Int> simplices;
        
        ReadFromFile<Real, Int>(file_name, coords, simplices);
        memcpy(B,coords.data(),3 * n * sizeof(Real));
    }

    Real * kappa = (Real*)malloc(wave_chunk_count * sizeof(Real));
    Real* inc = (Real*)malloc(wave_chunk_size * 3 * sizeof(Real));
    // Complex * coeff = (Complex*)malloc(4 * wave_chunk_count * sizeof(Complex));

    for (int i = 0 ; i < wave_chunk_count; i++)
    {
        kappa[i] = 2.0f*((float)i + 1.0f);
        // coeff[wave_chunk_count*i + 0] = Complex(0.0f,0.0f);
        // coeff[wave_chunk_count*i + 1] = Complex(0.0f,0.0f);
        // coeff[wave_chunk_count*i + 2] = Complex(1.0f,0.0f);
        // coeff[wave_chunk_count*i + 3] = Complex(0.0f,0.0f);
    }

    Complex* C = (Complex*)malloc(wave_count * H_GPU.GetMeasCount() * sizeof(Complex));
    for (int i = 0 ; i < 4; i++)
    {
        inc[12*i + 0] = 1.0f;
        inc[12*i + 1] = 0.0f;
        inc[12*i + 2] = 0.0f;
        inc[12*i + 3] = 0.0f;
        inc[12*i + 4] = 1.0f;
        inc[12*i + 5] = 0.0f;
        inc[12*i + 6] = 0.0f;
        inc[12*i + 7] = 0.0f;
        inc[12*i + 8] = 1.0f;
        inc[12*i + 9] = 1/std::sqrt(3.0f);
        inc[12*i + 10] = 1/std::sqrt(3.0f);
        inc[12*i + 11] = 1/std::sqrt(3.0f);
    }

    H_GPU.UseDiagonal(true);

    tic("derivative");
    H_GPU.Derivative_FF<Int,Real,Complex,wave_count>(kappa,wave_chunk_count,inc,wave_chunk_size,
                                                             B, C, 0.00001f, 0.0001f);
    toc("derivative");

    // GMRES<64,std::complex<float>,size_t,Side::Left> gmres(n,30,8);

    // BAEMM::Helmholtz_OpenCL::kernel_list list = H_GPU.LoadKernel(kappa,coeff,wave_count,wave_chunk_size);

    // auto A = [&H_GPU, &n, &wave_count]( const Complex * x, Complex *y )
    //             {
    //                 tic("inner");
    //                 H_GPU.ApplyBoundaryOperators_PL(wave_count,
    //                                 Complex(1.0f,0.0f),x,
    //                                 Complex(0.0f,0.0f),y
    //                                 );
    //                 toc("inner");
    //             };
    // auto P = [&n, &wave_count]( const Complex * x, Complex *y )
    //             {
    //                 memcpy(y,x,wave_count * n * sizeof(Complex));
    //             };
    // tic("outer");
    // bool succeeded = gmres(A,P,B,wave_count,C,wave_count,0.00001f,5);
    // toc("outer");

    // tic("GPU");
    // H_GPU.ApplyBoundaryOperators_PL(wave_count,
    //                 Complex(1.0f,0.0f),B,
    //                 Complex(0.0f,0.0f),C
    //                 );
    // toc("GPU");
    // H_GPU.DestroyKernel(&list);

    // tic("CPU");
    // H_CPU.ApplyBoundaryOperators_PL( Complex(1.0f,0.0f),B,wave_count,
    //                                 Complex(-1.0f,0.0f),C,wave_count,
    //                                 kappa,coeff,wave_count,wave_chunk_size
    //                                 );
    // toc("CPU");
    // float error = 0;
    // float abs = 0;
    // for (int i = 0; i < n; i++)
    // {   
    //     for (int j = 0; j < wave_count; j++)
    //     {
    //         std::complex<float> C_GPU = C[wave_count*i + j];
    //         // std::complex<float> C_CPU = H_CPU.C(i,j);
    //         abs = std::abs(C_GPU);
    //         if (abs > error)
    //         {
    //             error = abs;
    //         }
    //     }
    // }
    // std::cout << "error= " << error << std::endl;
    std::ofstream fout_r("data_real.txt");
    std::ofstream fout_i("data_imag.txt");
    if(fout_r.is_open() && fout_r.is_open())
	{
		for(int i = 0; i < H_GPU.GetMeasCount() ; i++)
		{
            for(int j = 0; j < wave_count ; j++)
            {
                fout_r << C[i * wave_count + j].real() << " "; 
                fout_i << C[i * wave_count + j].imag() << " "; 
            }
            fout_r << "\n";
            fout_i << "\n";
		}
	}            

    free(B);
    free(C);
    // free(coeff);
    free(inc);
    free(kappa);
    return 0;
}