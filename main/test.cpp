#include <iostream> 
#include <fstream> 
#include <cblas.h>
#include "read.hpp"
#include "../Tensors/GMRES.hpp"
#include "../Tensors/ConjugateGradient.hpp"

using Complex = std::complex<float>;

int main()
{
    BAEMM::Helmholtz_OpenCL H_GPU = read_OpenCL("/github/BAEMM/Meshes/TorusMesh_00038400T.txt");
    BAEMM::Helmholtz_CPU H_CPU = read_CPU("/github/BAEMM/Meshes/TorusMesh_00038400T.txt");
    
    Int n = H_GPU.VertexCount();
    const Int wave_count = 64;
    constexpr Int wave_chunk_size = 16;
    constexpr Int wave_chunk_count = wave_count/wave_chunk_size;

    Complex*B = (Complex*)malloc(wave_count * n * sizeof(Complex));

    for (int i = 0; i < n * wave_chunk_size; i++)
    {
        //B[i] = std::exp(Complex(0.0f,(float)i));
        B[i] = Complex(1.0f,0.0f);
    }

    Real * kappa = (Real*)malloc(wave_chunk_count * sizeof(Real));
    Complex * coeff = (Complex*)malloc(4 * wave_chunk_count * sizeof(Complex));

    for (int i = 0 ; i < wave_chunk_count; i++)
    {
        kappa[i] = 2.0f*((float)i + 1.0f);
        coeff[wave_chunk_count*i + 0] = Complex(0.5f,0.0f);
        coeff[wave_chunk_count*i + 1] = Complex(0.0f,-kappa[i]);
        coeff[wave_chunk_count*i + 2] = Complex(1.0f,0.0f);
        coeff[wave_chunk_count*i + 3] = Complex(0.0f,0.0f);
    }

    Complex* C = (Complex*)malloc(wave_count * n * sizeof(Complex));
    H_GPU.SetBlockSize(32);
    H_GPU.UseDiagonal(true);

    GMRES<64,std::complex<float>,size_t,Side::Left> gmres(n,30,8);

    BAEMM::Helmholtz_OpenCL::kernel_list list = H_GPU.LoadKernel(kappa,coeff,wave_count,wave_chunk_size);

    auto A = [&H_GPU, &n, &wave_count]( const Complex * x, Complex *y )
                {
                    tic("inner");
                    H_GPU.ApplyBoundaryOperators_PL(wave_count,
                                    Complex(1.0f,0.0f),x,
                                    Complex(0.0f,0.0f),y
                                    );
                    toc("inner");
                };
    auto P = [&n, &wave_count]( const Complex * x, Complex *y )
                {
                    memcpy(y,x,wave_count * n * sizeof(Complex));
                };
    tic("outer");
    bool succeeded = gmres(A,P,B,wave_count,C,wave_count,0.00001f,5);
    toc("outer");

    H_GPU.DestroyKernel(&list);

    tic("CPU");
    H_CPU.ApplyBoundaryOperators_PL( Complex(1.0f,0.0f),C,wave_count,
                                    Complex(-1.0f,0.0f),B,wave_count,
                                    kappa,coeff,wave_count,wave_chunk_size
                                    );
    toc("CPU");
    float error = 0;
    float abs = 0;
    for (int i = 0; i < n; i++)
    {   
        for (int j = 0; j < wave_count; j++)
        {
            std::complex<float> C_GPU = B[wave_count*i + j];
            // std::complex<float> C_CPU = H_CPU.C(i,j);
            abs = std::abs(C_GPU);
            if (abs > error)
            {
                error = abs;
            }
        }
    }
    std::cout << "error= " << error << std::endl;


    free(B);
    free(C);
    free(coeff);
    free(kappa);
    return 0;
}