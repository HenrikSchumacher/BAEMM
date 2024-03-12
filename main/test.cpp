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
    // BAEMM::Helmholtz_OpenCL H = read_OpenCL("/github/BAEMM/Meshes/Bunny_00086632T.txt");
    BAEMM::Helmholtz_OpenCL H = read_OpenCL("/github/BAEMM/Meshes/Blub_00227328T.txt");
    
    Int n = H.VertexCount();
    Int grid_coarse = 200;
    Int grid_fine = 2000;

    Int grid_coarse_2 = grid_coarse * grid_coarse;
    Int grid_coarse_3 = grid_coarse * grid_coarse * grid_coarse;

    Int grid_fine_2 = grid_fine * grid_fine;
    const Int wave_count = 4;
    constexpr Int wave_chunk_size = 1;
    constexpr Int wave_chunk_count = wave_count/wave_chunk_size;
    Complex* B = (Complex*)malloc( wave_count * n * sizeof(Complex));
    Complex* phi = (Complex*)malloc( wave_count * n * sizeof(Complex));
    Real* evaluation_points_1 = (Real*)malloc(3 * grid_coarse_3 * sizeof(Real));
    Real* evaluation_points_2 = (Real*)malloc(3 * grid_fine_2 * sizeof(Real));
    Complex* C_1 = (Complex*)malloc(wave_count * grid_coarse_3  * sizeof(Complex));
    Complex* C_2 = (Complex*)malloc(wave_count * grid_fine_2  * sizeof(Complex));

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

    kappa[0] = 2 * Scalar::Pi<Real>;
    kappa[1] = 4 * Scalar::Pi<Real>;
    kappa[2] = 5 * Scalar::Pi<Real>;
    kappa[3] = 7 * Scalar::Pi<Real>;

    // kappa[0] = 4 * Scalar::Pi<Real>;

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
    // for (int i = 0 ; i < wave_chunk_count; i++)
    // {
    //     inc[3*i + 0] = 1.0f/sqrt(3);
    //     inc[3*i + 1] = 1.0f/sqrt(3);
    //     inc[3*i + 2] = -1.0f/sqrt(3);
    // }

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
    // Real size_x = 1.5;
    // Real size_y = 1.5;
    // Real size_z = 1.5;


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
    
    {
        plane[0] = 0.0f;
        plane[1] = -0.2f;
        plane[2] = 0.0f;

        plane[3] = 1.0f;
        plane[4] = 0.0f;
        plane[5] = 0.0f;

        plane[6] = 0.0f;
        plane[7] = 0.0f;
        plane[8] = 1.0f;
    }

    // {
    //     plane[0] = -0.1432;
    //     plane[1] = -0.1065;
    //     plane[2] = 0.1494;

    //     plane[3] = 1.0/sqrt(3);
    //     plane[4] = 1.0/sqrt(3);
    //     plane[5] = -1.0/sqrt(3);

    //     plane[6] = 1.0/sqrt(6);
    //     plane[7] = 1.0/sqrt(6);
    //     plane[8] = 2.0/sqrt(6);
    // }

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
    H.CreateIncidentWave_PL(Complex(1.0f,0.0f), inc, wave_chunk_size,
                            Complex(0.0f,0.0f), B, wave_count,
                            kappa, wave_coeff, wave_count, wave_chunk_size,
                            BAEMM::Helmholtz_OpenCL::WaveType::Plane
                            );

    H.BoundaryPotential<wave_count>( kappa, coeff, B, phi, 
                                            wave_chunk_count, wave_chunk_size, cg_tol, gmres_tol );


    // H.ApplyNearFieldOperators_PL(
    //                     Complex(1.0f,0.0f), phi, wave_count, 
    //                     Complex(0.0f,0.0f), C_1, wave_count, 
    //                     kappa, coeff, wave_count, wave_chunk_size,
    //                     evaluation_points_1, grid_coarse_3);

    H.ApplyNearFieldOperators_PL(
                        Complex(1.0f,0.0f), phi, wave_count, 
                        Complex(0.0f,0.0f), C_2, wave_count, 
                        kappa, coeff, wave_count, wave_chunk_size,
                        evaluation_points_2, grid_fine_2);

    std::ofstream fout_points_3D("blub_eval_points_3D.txt");
    std::ofstream fout_points_plane("blub_eval_points_plane.txt");
    // std::ofstream fout_points_3D("bunny_eval_points_3D.txt");
    // std::ofstream fout_points_plane("bunny_eval_points_plane_1.txt");
    if(fout_points_3D.is_open() && fout_points_plane.is_open())
	{
		// for(int i = 0; i < grid_coarse_3 ; i++)
		// {
        //     for(int j = 0; j < 3 ; j++)
        //     {
        //         fout_points_3D << evaluation_points_1[i * 3 + j] << " "; 
        //     }
        //     fout_points_3D << "\n";
		// }
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

    std::ofstream fout_eval_3D_real("blub_eval_3D_2pi_4pi_5pi_7pi_real.txt");
    std::ofstream fout_eval_3D_imag("blub_eval_3D_2pi_4pi_5pi_7pi_imag.txt");
    std::ofstream fout_eval_plane_real("blub_eval_plane_2pi_4pi_5pi_7pi_real.txt");
    std::ofstream fout_eval_plane_imag("blub_eval_plane_2pi_4pi_5pi_7pi_imag.txt");

    // std::ofstream fout_eval_3D_real("bunny_eval_3D_4pi_real.txt");
    // std::ofstream fout_eval_3D_imag("bunny_eval_3D_4pi_imag.txt");
    // std::ofstream fout_eval_plane_real("bunny_eval_plane_1_4pi_real.txt");
    // std::ofstream fout_eval_plane_imag("bunny_eval_plane_1_4pi_imag.txt");

    if(fout_eval_3D_real.is_open() && fout_eval_plane_real.is_open()
    && fout_eval_3D_imag.is_open() && fout_eval_plane_imag.is_open() )
	{
		// for(int i = 0; i < grid_coarse_3 ; i++)
		// {
        //     for(int j = 0; j < wave_count ; j++)
        //     {
        //         fout_eval_3D_real << C_1[wave_count*i + j].real() << " "; 
        //         fout_eval_3D_imag << C_1[wave_count*i + j].imag() << " "; 
        //     }
        //     fout_eval_3D_real<< "\n";
        //     fout_eval_3D_imag<< "\n";
		// }
        fout_eval_3D_real.close();
        fout_eval_3D_imag.close();

        for(int i = 0; i < grid_fine_2 ; i++)
		{
            for(int j = 0; j < wave_count ; j++)
            {
                fout_eval_plane_real << C_2[wave_count*i + j].real() << " "; 
                fout_eval_plane_imag << C_2[wave_count*i + j].imag() << " ";
            }
            fout_eval_plane_real << "\n";
            fout_eval_plane_imag << "\n";
		}
        fout_eval_plane_real.close();
        fout_eval_plane_imag.close();
	}

    // {
    //     plane[0] = -0.05f;
    //     plane[1] = -0.02f;
    //     plane[2] = 0.8f;

    //     plane[3] = 1.0/sqrt(6);
    //     plane[4] = 1.0/sqrt(6);
    //     plane[5] = 2.0/sqrt(6);

    //     plane[6] = 1.0/sqrt(2);
    //     plane[7] = -1.0/sqrt(2);
    //     plane[8] = 0.0;
    // }
    // // H.DestroyKernel(&list);

    // for (int i = 0 ; i < grid_fine; i++)
    // {
    //     for (int j = 0 ; j < grid_fine; j++)
    //     {
    //         Real s_1 = -size_dir_1 + (i* 2 * size_dir_1 )/( grid_fine - 1 ) ;
    //         Real s_2 = -size_dir_2 + (j* 2 * size_dir_2 )/( grid_fine - 1 ) ;

    //         evaluation_points_2[3 * grid_fine * i + 3 * j + 0]  = plane[0] + plane[3] * s_1 + plane[6] * s_2;
            
    //         evaluation_points_2[3 * grid_fine * i + 3 * j + 1]  = plane[1] + plane[4] * s_1 + plane[7] * s_2;

    //         evaluation_points_2[3 * grid_fine * i + 3 * j + 2]  = plane[2] + plane[5] * s_1 + plane[8] * s_2;
    //     }
    // }

    // H.ApplyNearFieldOperators_PL(
    //                     Complex(1.0f,0.0f), phi, wave_count, 
    //                     Complex(0.0f,0.0f), C_2, wave_count, 
    //                     kappa, coeff, wave_count, wave_chunk_size,
    //                     evaluation_points_2, grid_fine_2);

    // std::ofstream fout_points_plane_2("bunny_eval_points_plane_2.txt");
    // if(fout_points_plane_2.is_open())
	// {
    //     for(int i = 0; i < grid_fine_2 ; i++)
	// 	{
    //         for(int j = 0; j < 3 ; j++)
    //         {
    //             fout_points_plane_2 << evaluation_points_2[i * 3 + j] << " "; 
    //         }
    //         fout_points_plane_2 << "\n";
	// 	}
    //     fout_points_plane_2.close();
	// }

    // std::ofstream fout_eval_plane_2_real("bunny_eval_plane_2_4pi_real.txt");
    // std::ofstream fout_eval_plane_2_imag("bunny_eval_plane_2_4pi_imag.txt");

    // if(fout_eval_plane_2_real.is_open() && fout_eval_plane_2_imag.is_open() )
	// {
    //     for(int i = 0; i < grid_fine_2 ; i++)
	// 	{
    //         for(int j = 0; j < wave_count ; j++)
    //         {
    //             fout_eval_plane_2_real << C_2[wave_count*i + j].real() << " "; 
    //             fout_eval_plane_2_imag << C_2[wave_count*i + j].imag() << " ";
    //         }
    //         fout_eval_plane_2_real << "\n";
    //         fout_eval_plane_2_imag << "\n";
	// 	}
    //     fout_eval_plane_2_real.close();
    //     fout_eval_plane_2_imag.close();
	// }

    // {
    //     plane[0] = -0.1432f;
    //     plane[1] = -0.1065f;
    //     plane[2] = 0.1494f;

    //     plane[3] = 1.0f/sqrt(6);
    //     plane[4] = 1.0f/sqrt(6);
    //     plane[5] = 2.0f/sqrt(6);

    //     plane[6] = 1.0f/sqrt(2);
    //     plane[7] = -1.0f/sqrt(2);
    //     plane[8] = 0.0f;
    // }
    // // H.DestroyKernel(&list);

    // for (int i = 0 ; i < grid_fine; i++)
    // {
    //     for (int j = 0 ; j < grid_fine; j++)
    //     {
    //         Real s_1 = -size_dir_1 + (i* 2 * size_dir_1 )/( grid_fine - 1 ) ;
    //         Real s_2 = -size_dir_2 + (j* 2 * size_dir_2 )/( grid_fine - 1 ) ;

    //         evaluation_points_2[3 * grid_fine * i + 3 * j + 0]  = plane[0] + plane[3] * s_1 + plane[6] * s_2;
            
    //         evaluation_points_2[3 * grid_fine * i + 3 * j + 1]  = plane[1] + plane[4] * s_1 + plane[7] * s_2;

    //         evaluation_points_2[3 * grid_fine * i + 3 * j + 2]  = plane[2] + plane[5] * s_1 + plane[8] * s_2;
    //     }
    // }

    // H.ApplyNearFieldOperators_PL(
    //                     Complex(1.0f,0.0f), phi, wave_count, 
    //                     Complex(0.0f,0.0f), C_2, wave_count, 
    //                     kappa, coeff, wave_count, wave_chunk_size,
    //                     evaluation_points_2, grid_fine_2);

    // std::ofstream fout_points_plane_3("bunny_eval_points_plane_3.txt");
    // if(fout_points_plane_2.is_open())
	// {
    //     for(int i = 0; i < grid_fine_2 ; i++)
	// 	{
    //         for(int j = 0; j < 3 ; j++)
    //         {
    //             fout_points_plane_2 << evaluation_points_2[i * 3 + j] << " "; 
    //         }
    //         fout_points_plane_2 << "\n";
	// 	}
    //     fout_points_plane_2.close();
	// }

    // std::ofstream fout_eval_plane_3_real("bunny_eval_plane_3_4pi_real.txt");
    // std::ofstream fout_eval_plane_3_imag("bunny_eval_plane_3_4pi_imag.txt");

    // if(fout_eval_plane_3_real.is_open() && fout_eval_plane_3_imag.is_open() )
	// {
    //     for(int i = 0; i < grid_fine_2 ; i++)
	// 	{
    //         for(int j = 0; j < wave_count ; j++)
    //         {
    //             fout_eval_plane_3_real << C_2[wave_count*i + j].real() << " "; 
    //             fout_eval_plane_3_imag << C_2[wave_count*i + j].imag() << " ";
    //         }
    //         fout_eval_plane_3_real << "\n";
    //         fout_eval_plane_3_imag << "\n";
	// 	}
    //     fout_eval_plane_3_real.close();
    //     fout_eval_plane_3_imag.close();
	// }


    free(B);
    free(C_1);
    free(C_2);
    free(evaluation_points_1);
    free(evaluation_points_2);
    free(inc);
    free(kappa);
    free(coeff);
    free(wave_coeff);
    free(plane);
    return 0;
}