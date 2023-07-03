#include <iostream> 
#include <fstream> 
#include <cblas.h>
#include "read.hpp"

#include "../../../Repulsor/Repulsor.hpp"


using Complex = std::complex<Real>;
using LInt = long long;
using SReal = Real;
using ExtReal = Real;

using namespace Tools;
using namespace Tensors;
using namespace Repulsor;

int main()
{
    BAEMM::Helmholtz_OpenCL H = read_OpenCL("/github/BAEMM/Meshes/TorusMesh_00153600T.txt");
    // BAEMM::Helmholtz_CPU H_CPU = read_CPU("/github/BAEMM/Meshes/TorusMesh_00153600T.txt");
    
    Int n = H.VertexCount();
    const Int wave_count = 16;
    constexpr Int wave_chunk_size = 16;
    constexpr Int wave_chunk_count = wave_count/wave_chunk_size;

    Complex* B = (Complex*)malloc(16 * H.GetMeasCount() * sizeof(Complex));

    Int thread_count = 4;

    Real regpar = 1.0f;

    // for (int i = 0; i < n; i++)
    // {
    //     B[i] = std::exp(Complex(0.0f,(float)i));
    //     B[i] = 1.0f;
    //     B[i] = 0.0f;
    //     B[i] = 0.0f;
    // }
    using namespace Tensors;
    using namespace Tools;

    const std::string path = "/HOME1/users/guests/jannr/github/BAEMM/Meshes/TorusMesh_00038400T.txt";
    std::string file_name = path;
    Tensor2<Real, Int> coords;
    Tensor2<Int, Int> simplices;

    ReadFromFile<Real, Int>(file_name, coords, simplices);

    Real * kappa = (Real*)malloc(wave_chunk_count * sizeof(Real));
    Real* inc = (Real*)malloc(wave_chunk_size * 3 * sizeof(Real));
    // Complex * coeff = (Complex*)malloc(4 * wave_chunk_count * sizeof(Complex));

    for (int i = 0 ; i < wave_chunk_count; i++)
    {
        kappa[i] = 2.0f*((float)i + 1.0f);
    }

    Real* C = (Real*)malloc(3 * n * sizeof(Real));

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

    H.UseDiagonal(true);

    Real cg_tol = static_cast<Real>(0.00001);
    Real gmres_tol = static_cast<Real>(0.0005);

    Tensor2<Complex,Int> neumann_data_scat;
    Complex* neumann_data_scat_ptr = NULL;

    using Mesh_T     = SimplicialMesh<2,3,Real,Int,LInt,SReal,ExtReal>;
    using Mesh_Ptr_T = std::shared_ptr<Mesh_T>;

    constexpr Int NRHS = 3;

    Mesh_Ptr_T M = std::make_shared<Mesh_T>(
        coords.data(),  n,
        simplices.data(),  H.SimplexCount(),
        thread_count
        );

    M->cluster_tree_settings.split_threshold                        =  2;
    M->cluster_tree_settings.thread_count                           =  thread_count; // take as many threads as there are used by SimplicialMesh M
    M->block_cluster_tree_settings.far_field_separation_parameter   =  0.125f;
    M->adaptivity_settings.theta                                    = 10.0f;

    const Real q  = 6;
    const Real p  = 12;
    const Real s = (p - 2) / q;

    TangentPointEnergy0<Mesh_T>       tpe        (q,p);
    TangentPointMetric0<Mesh_T>       tpm        (q,p);
    PseudoLaplacian    <Mesh_T,false> pseudo_lap (2-s);

    // The operator for the metric.
    auto A = [&]( ptr<Real> X, mut<Real> Y )
    {
        tpm.MultiplyMetric( *M, regpar, X, Scalar::One<Real>, Y, NRHS );
    };

    Real one_over_regpar = 1/regpar;

    Tensor2<Real,Int> Z_buffer  ( M->VertexCount(), NRHS );

    mut<Real> Z  = Z_buffer.data();

    // The operator for the preconditioner.
    auto P = [&]( ptr<Real> X, mut<Real> Y )
    {
        M->H1Solve( X, Y, NRHS );
        pseudo_lap.MultiplyMetric( *M, one_over_regpar, Y, Scalar::Zero<Real>, Z, NRHS );
        M->H1Solve( Z, Y, NRHS );
    };
    //-----------------------------------------------------------------------------------------------------------------------------------------------------------------

    Real gmres_tol_outer = 0.005;

    Tensor2<Real,Int> DE (n,3);
    Tensor2<Real,Int> grad (n,3);

    ptr<Real> DE_ptr = DE.data();
    mut<Real> grad_ptr = grad.data();

    tpe.Differential( *M ).Write( DE_ptr );

    H.AdjointDerivative_FF<Int,Real,Complex,16>( kappa, wave_chunk_count, inc, wave_chunk_size,
                        B, grad_ptr, &neumann_data_scat_ptr, cg_tol, gmres_tol);

    combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Generic>(regpar,DE_ptr,1.0f,grad_ptr,n * 3, thread_count);

    H.GaussNewtonStep<Int,16>( kappa, wave_chunk_count, inc, wave_chunk_size, 
                A, P, grad_ptr, C, &neumann_data_scat_ptr, cg_tol, gmres_tol, gmres_tol_outer);

    
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
    // free(coeff);
    free(inc);
    free(kappa);
    return 0;
}