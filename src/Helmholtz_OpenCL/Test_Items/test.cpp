#include <iostream> 
#include <fstream> 
#include <cblas.h>
#include "read.hpp"

#include "../../../Helmholtz_OpenCL.hpp"

using Complex = std::complex<Real>;
using LInt = long long;
using SReal = Real;
using ExtReal = Real;

using namespace Tools;
using namespace Tensors;
using namespace Repulsor;

int main()
{
    BAEMM::Helmholtz_OpenCL H = read_OpenCL("/github/BAEMM/Meshes/Blub_00056832T.txt");
    // BAEMM::Helmholtz_CPU H_CPU = read_CPU("/github/BAEMM/Meshes/TorusMesh_00153600T.txt");
    
    Int n = H.VertexCount();
    Int grid_coarse = 10;
    Int grid_fine = 2000;

    Int grid_coarse_2 = grid_coarse * grid_coarse;
    Int grid_coarse_3 = grid_coarse * grid_coarse * grid_coarse;

    Int grid_fine_2 = grid_fine * grid_fine;
    const Int wave_count = 1;
    constexpr Int wave_chunk_size = 1;
    constexpr Int wave_chunk_count = wave_count/wave_chunk_size;
    Complex* B = (Complex*)malloc( wave_count * n * sizeof(Complex));
    Complex* phi = (Complex*)malloc( wave_count * n * sizeof(Complex));
    Real* evaluation_points_1 = (Real*)malloc(3 * grid_coarse_3 * sizeof(Real));
    Real* evaluation_points_2 = (Real*)malloc(3 * grid_fine_2 * sizeof(Real));
    Complex* C_1 = (Complex*)malloc(grid_coarse_3  * sizeof(Complex));
    Complex* C_2 = (Complex*)malloc(grid_fine_2  * sizeof(Complex));

    Int thread_count = 4;

    // using namespace Tensors;
    // using namespace Tools;

    // // const std::string path = "/HOME1/users/guests/jannr/github/BAEMM/Meshes/TorusMesh_00038400T.txt";
    // // std::string file_name = path;
    // // Tensor2<Real, Int> coords;
    // // Tensor2<Int, Int> simplices;

    // // ReadFromFile<Real, Int>(file_name, coords, simplices);

    Real * kappa = (Real*)malloc(wave_chunk_count * sizeof(Real));
    Real* inc = (Real*)malloc(wave_chunk_size * 3 * sizeof(Real));
    Complex * coeff = (Complex*)malloc(4 * wave_chunk_count * sizeof(Complex));
    Complex * wave_coeff = (Complex*)malloc(4 * wave_chunk_count * sizeof(Complex));

    for (int i = 0 ; i < wave_chunk_count; i++)
    {
        kappa[i] = 7 * Scalar::Pi<Real>;
    }

    for(Int i = 0 ; i < wave_chunk_count ; i++)
    {
        coeff[4 * i + 0] = 0.0;
        coeff[4 * i + 1] = Complex(0.0,-kappa[i]);
        coeff[4 * i + 2] = 1.0;
        coeff[4 * i + 3] = 0.0;
    }

    for(Int i = 0 ; i < wave_chunk_count ; i++)
    {
        wave_coeff[4 * i + 0] = 0.0f;
        wave_coeff[4 * i + 1] = 1.0f;
        wave_coeff[4 * i + 2] = 0.0f;
        wave_coeff[4 * i + 3] = 0.0f;
    }

    // Real* C = (Real*)malloc(3 * n * sizeof(Real));

    for (int i = 0 ; i < wave_chunk_count; i++)
    {
        inc[3*i + 0] = 1.0f;
        inc[3*i + 1] = 0.0f;
        inc[3*i + 2] = 0.0f;
    }

    H.UseDiagonal(true);

    Real cg_tol = static_cast<Real>(0.00001);
    Real gmres_tol = static_cast<Real>(0.0001);

    // Complex* neumann_data_scat_ptr = NULL;

    // const Real* B = H.VertexCoordinates();
    // for (int i = 0; i < 16 * n; i++)
    // {
    //     B[i] = Complex(1.0f,2.0f);
    // }
    Real size_x = 2;
    Real size_y = 2;
    Real size_z = 2;

    for (int i = 0 ; i < grid_coarse; i++)
    {
        for (int j = 0 ; j < grid_coarse; j++)
        {
            for (int k = 0 ; k < grid_coarse; k++)
            {
                evaluation_points_1[3 * grid_coarse_2 * i + 3 * grid_coarse * j + 3 * k + 0]  = -size_x + k*( 2 * size_x )/( grid_coarse - 1 );
                evaluation_points_1[3 * grid_coarse_2 * i + 3 * grid_coarse * j + 3 * k + 1]  = -size_y + j*( 2 * size_y )/( grid_coarse - 1 );
                evaluation_points_1[3 * grid_coarse_2 * i + 3 * grid_coarse * j + 3 * k + 2]  = -size_z + i*( 2 * size_z )/( grid_coarse - 1 );
            }
        }
    }

    //plane for near field measurements specified by base point, span direction 1 and span direction 2
    Real* plane = (Real*)malloc(3 * 3 * sizeof(Real));
    
    // {
    //     plane[0] = 0.0f;
    //     plane[1] = -0.2f;
    //     plane[2] = 0.0f;

    //     plane[3] = 1.0f;
    //     plane[4] = 0.0f;
    //     plane[5] = 0.0f;

    //     plane[6] = 0.0f;
    //     plane[7] = 0.0f;
    //     plane[8] = 1.0f;
    // }

    {
        plane[0] = -0.1432f;
        plane[1] = -0.1065f;
        plane[2] = 0.1494f;

        plane[3] = 1.0f/sqrt(6);
        plane[4] = 1.0f/sqrt(6);
        plane[5] = 2.0f/sqrt(6);

        plane[6] = 1.0f/sqrt(2);
        plane[7] = -1.0f/sqrt(2);
        plane[8] = 0.0f;
    }

    Real size_dir_1 = 2;
    Real size_dir_2 = 2;

    for (int i = 0 ; i < grid_fine; i++)
    {
        for (int j = 0 ; j < grid_fine; j++)
        {
            Real s_1 = -size_dir_1 + (i* 2 * size_dir_1 )/( grid_fine - 1 ) ;
            Real s_2 = -size_dir_2 + (j* 2 * size_dir_2 )/( grid_fine - 1 ) ;

            evaluation_points_2[3 * grid_fine * i + 3 * j + 0]  = plane[0] + plane[3] * s_1 + plane[6] * s_2;
            
            evaluation_points_2[3 * grid_fine * i + 3 * j + 1]  = plane[1] + plane[4] * s_1 + plane[7] * s_2;

            evaluation_points_2[3 * grid_fine * i + 3 * j + 2]  = plane[2] + plane[5] * s_1 + plane[8] * s_2;
        }
    }

    // H.type_cast(B,H.VertexCoordinates(),3*n,4);
    // H.CreateIncidentWave_PL(Complex(1.0f,0.0f), inc, wave_chunk_size,
    //                         Complex(0.0f,0.0f), B, wave_count,
    //                         kappa, wave_coeff, wave_count, wave_chunk_size,
    //                         BAEMM::Helmholtz_OpenCL::WaveType::Plane
    //                         );

    // H.BoundaryPotential<wave_count>( kappa, coeff, B, phi, 
    //                                         wave_chunk_count, wave_chunk_size, cg_tol, gmres_tol );


    // H.ApplyNearFieldOperators_PL(
    //                     Complex(1.0f,0.0f), phi, wave_count, 
    //                     Complex(0.0f,0.0f), C_1, wave_count, 
    //                     kappa, coeff, wave_count, wave_chunk_size,
    //                     evaluation_points_1, grid_coarse_3);

    // H.ApplyNearFieldOperators_PL(
    //                     Complex(1.0f,0.0f), phi, wave_count, 
    //                     Complex(0.0f,0.0f), C_2, wave_count, 
    //                     kappa, coeff, wave_count, wave_chunk_size,
    //                     evaluation_points_2, grid_fine_2);

    // H.DestroyKernel(&list);

    std::ofstream fout_points_3D("blub_eval_points_3D.txt");
    std::ofstream fout_points_plane("bunny_eval_points_plane_3.txt");
    if(fout_points_3D.is_open() && fout_points_plane.is_open())
	{
		for(int i = 0; i < grid_coarse_3 ; i++)
		{
            for(int j = 0; j < 3 ; j++)
            {
            }
            fout_points_3D << "\n";
		}
        fout_points_3D.close();
        for(int i = 0; i < grid_fine_2 ; i++)
		{
            for(int j = 0; j < 3 ; j++)
            {
                fout_points_plane << evaluation_points_2[i * 3 + j] << " "; 
            }
            fout_points_plane << "\n";
		}
        fout_points_plane.close();
	}

    std::ofstream fout_eval_3D("blub_eval_3D.txt");
    std::ofstream fout_eval_plane("blub_eval_plane.txt");

    if(fout_eval_3D.is_open() && fout_eval_plane.is_open() )
	{
		for(int i = 0; i < grid_coarse_3 ; i++)
		{
            fout_eval_3D << C_1[i].real() << " "; 
            fout_eval_3D << C_1[i].imag() << " "; 

            fout_eval_3D<< "\n";
		}
        fout_eval_3D.close();

        for(int i = 0; i < grid_fine_2 ; i++)
		{
            fout_eval_plane << C_2[i].real() << " "; 
            fout_eval_plane << C_2[i].imag() << " ";

            fout_eval_plane << "\n";
		}
        fout_eval_plane.close();
	}

    free(B);
    free(C_1);
    free(C_2);
    free(evaluation_points_1);
    free(evaluation_points_2);
    free(inc);
    free(kappa);
    free(coeff);
    free(wave_coeff);
    return 0;
}