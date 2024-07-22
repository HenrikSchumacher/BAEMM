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
    // BAEMM::Helmholtz_OpenCL H = read_OpenCL("/github/BAEMM/Meshes/Dolphin_00075046T.txt");
    // BAEMM::Helmholtz_OpenCL H = read_OpenCL("/github/BAEMM/Meshes/SnakeTorus_00149340T.txt");
    // BAEMM::Helmholtz_OpenCL H = read_OpenCL("/github/BAEMM/Meshes/Bob_00171008T.txt");
    BAEMM::Helmholtz_OpenCL H = read_OpenCL("/github/BAEMM/Meshes/Spot_00093696T.txt");
    // BAEMM::Helmholtz_OpenCL H = read_OpenCL("/github/BAEMM/Meshes/Blub_00227328T.txt");
    // BAEMM::Helmholtz_OpenCL H = read_OpenCL("/github/BAEMM/Meshes/Bunny_00086632T.txt");
    // BAEMM::Helmholtz_OpenCL H = read_OpenCL("/github/BAEMM/Meshes/Triceratops_00090560T.txt");
    // BAEMM::Helmholtz_CPU H_CPU = read_CPU("/github/BAEMM/Meshes/TorusMesh_00153600T.txt");
    
    Int n = H.VertexCount();
    // Int grid_coarse = 10;
    // Int grid_fine = 2000;

    // Int grid_coarse_2 = grid_coarse * grid_coarse;
    // Int grid_coarse_3 = grid_coarse * grid_coarse * grid_coarse;

    // Int grid_fine_2 = grid_fine * grid_fine;
    const Int wave_count = 32;
    constexpr Int wave_chunk_size = 16;
    constexpr Int wave_chunk_count = wave_count/wave_chunk_size;
    // Complex* B = (Complex*)malloc( wave_count * n * sizeof(Complex));
    // Complex* phi = (Complex*)malloc( wave_count * n * sizeof(Complex));
    // Real* evaluation_points_1 = (Real*)malloc(3 * grid_coarse_3 * sizeof(Real));
    // Real* evaluation_points_2 = (Real*)malloc(3 * grid_fine_2 * sizeof(Real));
    // Complex* C_1 = (Complex*)malloc(grid_coarse_3  * sizeof(Complex));
    Complex* C = (Complex*)malloc(2562 *  wave_count * sizeof(Complex));

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
    // Complex * coeff = (Complex*)malloc(4 * wave_chunk_count * sizeof(Complex));
    // Complex * wave_coeff = (Complex*)malloc(4 * wave_chunk_count * sizeof(Complex));

    for (int i = 0 ; i < wave_chunk_count; i++)
    {
        kappa[i] = ( 1 + 2 * ( i + 2 ) ) * Scalar::Pi<Real>;
    }

    // for(Int i = 0 ; i < wave_chunk_count ; i++)
    // {
    //     coeff[4 * i + 0] = 0.0;
    //     coeff[4 * i + 1] = Complex(0.0,-kappa[i]);
    //     coeff[4 * i + 2] = 1.0;
    //     coeff[4 * i + 3] = 0.0;
    // }

    // for(Int i = 0 ; i < wave_chunk_count ; i++)
    // {
    //     wave_coeff[4 * i + 0] = 0.0f;
    //     wave_coeff[4 * i + 1] = 1.0f;
    //     wave_coeff[4 * i + 2] = 0.0f;
    //     wave_coeff[4 * i + 3] = 0.0f;
    // }

    // Real* C = (Real*)malloc(3 * n * sizeof(Real));
    //     Real s2 = 1 / std::sqrt(2.0);
    //     Real s3 = 1 / std::sqrt(3.0);
    //     Real s6 = s2 * s3;

    // {
    //     inc[0] = -0.6139603386989176;
    //     inc[1] = 0.5752246209433741;
    //     inc[2] = 0.5405269077162229;

    //     inc[3] = -0.6427081534742487;
    //     inc[4] = 0.444316073423647;
    //     inc[5] = -0.6241069270206141;

    //     inc[6] = 0.5236317008808439;
    //     inc[7] = 0.5291418103709742;
    //     inc[8] = -0.6676966274813454;

    //     inc[9] = 0.02327260011138572;
    //     inc[10] = -0.6534915372980049;
    //     inc[11] = -0.7565759689310427;

    //     inc[12] = 0.06786995521063467;
    //     inc[13] = -0.4608259755015667;
    //     inc[14] = 0.8848915693364562;

    //     inc[15] = 0.8719213179679728;
    //     inc[16] = -0.4883600089084584;
    //     inc[17] = 0.03546430560329308;

    //     inc[18] = -0.7815027083474867;
    //     inc[19] = -0.6171064624289668;
    //     inc[20] = 0.09183207976491509;

    //     inc[21] = 0.5514889942457563;
    //     inc[22] = 0.6709379834249211;
    //     inc[23] = 0.49568347927231204;
    // }

    // {
    //     inc[0] = s3;
    //     inc[1] = s3;
    //     inc[2] = -s3;

    //     inc[3] = 0.5f;
    //     inc[4] = 0.5f;
    //     inc[5] = -s2;

    //     inc[6] = s2;
    //     inc[7] = 0.5f;
    //     inc[8] = -0.5f;

    //     inc[9] = 0.5f;
    //     inc[10] = s2;
    //     inc[11] = -0.5f;

    //     inc[12] = s2;
    //     inc[13] = s3;
    //     inc[14] = -s6;

    //     inc[15] = s2;
    //     inc[16] = s6;
    //     inc[17] = -s3;

    //     inc[18] = s3;
    //     inc[19] = s2;
    //     inc[20] = -s6;

    //     inc[21] = s3;
    //     inc[22] = s6;
    //     inc[23] = -s2;
    // }
    // for (int i = 0 ; i < wave_chunk_count; i++)
    {
        inc[0] = 0.22663516023574246;
        inc[1] = 0.4654289185373517;
        inc[2] = 0.8555772472045235;
        inc[3] = -0.3474510994575331;
        inc[4] = -0.7864451523456505;
        inc[5] = -0.5106679506663584;
        inc[6] = -0.9541836471144971;
        inc[7] = -0.27451273343735705;
        inc[8] = -0.11906438073591856;
        inc[9] = 0.43775029208027166;
        inc[10] = -0.8876694056416434;
        inc[11] = 0.1428905457734996;
        inc[12] = -0.44907802104219113;
        inc[13] = -0.8139693295211834;
        inc[14] = 0.36848726113078095;
        inc[15] = 0.47666884531973414;
        inc[16] = 0.3192753380458682;
        inc[17] = -0.8190543757390276;
        inc[18] = -0.02194389227182042;
        inc[19] = 0.8818743114784301;
        inc[20] = -0.47097363444932655;
        inc[21] = 0.786615210967289;
        inc[22] = -0.1684579992437276;
        inc[23] = 0.5940188653281036;
        inc[24] = 0.42244725439759406;
        inc[25] = -0.5580931680624308;
        inc[26] = -0.7141920841160128;
        inc[27] = 0.6792359217126083;
        inc[28] = 0.7269051927489542;
        inc[29] = 0.10122945919953094;
        inc[30] = -0.3904321861768485;
        inc[31] = 0.03454064212408624;
        inc[32] = -0.919983506394991;
        inc[33] = -0.3009524704632311;
        inc[34] = 0.8792870632871999;
        inc[35] = 0.36916374531886265;
        inc[36] = -0.006080463353741372;
        inc[37] = -0.38604914872328944;
        inc[38] = 0.9224581739761577;
        inc[39] = 0.9742753958166932;
        inc[40] = -0.0853848807263007;
        inc[41] = -0.20855904499585115;
        inc[42] = -0.8068074072061289;
        inc[43] = 0.5272347230598002;
        inc[44] = -0.2666183686046023;
        inc[45] = -0.7266826985737691;
        inc[46] = 0.12599246204091846;
        inc[47] = 0.675320779409617;
    }

    // {
    //     inc[0] = 0.22663516023574246;
    //     inc[1] = 0.8555772472045235;
    //     inc[2] = 0.4654289185373517;

    //     inc[3] = -0.3474510994575331;
    //     inc[4] = -0.5106679506663584;
    //     inc[5] = -0.7864451523456505;

    //     inc[6] = -0.9541836471144971;
    //     inc[7] = -0.11906438073591856;
    //     inc[8] = -0.27451273343735705;

    //     inc[9] = 0.43775029208027166;
    //     inc[10] = 0.1428905457734996;
    //     inc[11] = -0.8876694056416434;

    //     inc[12] = -0.44907802104219113;
    //     inc[14] = -0.8139693295211834;
    //     inc[13] = 0.36848726113078095;

    //     inc[15] = 0.47666884531973414;
    //     inc[17] = 0.3192753380458682;
    //     inc[16] = -0.8190543757390276;

    //     inc[18] = -0.02194389227182042;
    //     inc[20] = 0.8818743114784301;
    //     inc[19] = -0.47097363444932655;

    //     inc[21] = 0.786615210967289;
    //     inc[23] = -0.1684579992437276;
    //     inc[22] = 0.5940188653281036;

    //     inc[24] = 0.42244725439759406;
    //     inc[26] = -0.5580931680624308;
    //     inc[25] = -0.7141920841160128;

    //     inc[27] = 0.6792359217126083;
    //     inc[29] = 0.7269051927489542;
    //     inc[28] = 0.10122945919953094;
        
    //     inc[30] = -0.3904321861768485;
    //     inc[32] = 0.03454064212408624;
    //     inc[31] = -0.919983506394991;

    //     inc[33] = -0.3009524704632311;
    //     inc[35] = 0.8792870632871999;
    //     inc[34] = 0.36916374531886265;

    //     inc[36] = -0.006080463353741372;
    //     inc[38] = -0.38604914872328944;
    //     inc[37] = 0.9224581739761577;

    //     inc[39] = 0.9742753958166932;
    //     inc[41] = -0.0853848807263007;
    //     inc[40] = -0.20855904499585115;

    //     inc[42] = -0.8068074072061289;
    //     inc[44] = 0.5272347230598002;
    //     inc[43] = -0.2666183686046023;

    //     inc[45] = -0.7266826985737691;
    //     inc[47] = 0.12599246204091846;
    //     inc[46] = 0.675320779409617;
    // }

    // {
    //     inc[0] = -0.30885429214287735;
    //     inc[1] = 0.5110289960384409;
    //     inc[2] = -0.8021585824716119;
    //     inc[3] = -0.7122172661855586;
    //     inc[4] = -0.3835378525059779;
    //     inc[5] = 0.587916049655282;
    //     inc[6] = 0.9921122654979078;
    //     inc[7] = 0.07428216572294265;
    //     inc[8] = -0.10097233534052051;
    //     inc[9] = 0.309323102455332;
    //     inc[10] = -0.5110726949593175;
    //     inc[11] = 0.801950072482339;
    //     inc[12] = 0.8306115622422868;
    //     inc[13] = -0.5458345991888321;
    //     inc[14] = -0.11022260656414734;
    //     inc[15] = -0.33943387939894687;
    //     inc[16] = -0.7431110130629967;
    //     inc[17] = -0.5766893997470979;
    //     inc[18] = -0.9920868267338052;
    //     inc[19] = -0.07416947325749827;
    //     inc[20] = 0.10130457767521617;
    //     inc[21] = -0.42855244102089546;
    //     inc[22] = 0.8597017215835386;
    //     inc[23] = -0.27794919536010876;
    //     inc[24] = 0.7023161250736549;
    //     inc[25] = -0.25283003018941874;
    //     inc[26] = -0.6654540076488713;
    //     inc[27] = -0.8223323638481848;
    //     inc[28] = -0.3391671348454271;
    //     inc[29] = -0.4568754075331702;
    //     inc[30] = -0.8305599943345937;
    //     inc[31] = 0.5459060486677456;
    //     inc[32] = 0.11025734369596002;
    //     inc[33] = 0.8220029049707602;
    //     inc[34] = 0.3396963769066609;
    //     inc[35] = 0.4570750438780478;
    //     inc[36] = 0.13126177104481765;
    //     inc[37] = -0.43991543701645097;
    //     inc[38] = -0.8883944820499521;
    //     inc[39] = -0.6931978532018274;
    //     inc[40] = -0.7194892386255246;
    //     inc[41] = 0.042567262285011655;
    //     inc[42] = 0.8230148545822212;
    //     inc[43] = -0.3007426185760727;
    //     inc[44] = 0.4818717946809214;
    //     inc[45] = 0.21881464529251152;
    //     inc[46] = 0.791442710736852;
    //     inc[47] = -0.5707351282574221;
    //     inc[48] = -0.3541756450565573;
    //     inc[49] = 0.8298131182705122;
    //     inc[50] = 0.431242160734477;
    //     inc[51] = -0.702379633300557;
    //     inc[52] = 0.25252582973091;
    //     inc[53] = 0.6655024838746213;
    //     inc[54] = -0.2241122969981281;
    //     inc[55] = -0.19086825794033443;
    //     inc[56] = 0.9556897961394403;
    //     inc[57] = -0.2184184726319404;
    //     inc[58] = -0.7916474458849384;
    //     inc[59] = 0.5706029199338049;
    //     inc[60] = -0.8228452584509618;
    //     inc[61] = 0.3011388794743019;
    //     inc[62] = -0.4819139507357426;
    //     inc[63] = 0.35404074104951877;
    //     inc[64] = -0.8298537554989681;
    //     inc[65] = -0.4312747362892548;
    //     inc[66] = -0.11794224929402719;
    //     inc[67] = -0.992580675912642;
    //     inc[68] = -0.02955042531451197;
    //     inc[69] = 0.3389090325663378;
    //     inc[70] = 0.7433256868940885;
    //     inc[71] = 0.5767214152850411;
    //     inc[72] = 0.11778379216067858;
    //     inc[73] = 0.9926061049827104;
    //     inc[74] = 0.029327438607942645;
    //     inc[75] = 0.7125199631198716;
    //     inc[76] = 0.3831781083549273;
    //     inc[77] = -0.5877838373357985;
    //     inc[78] = 0.4286583863809451;
    //     inc[79] = -0.8597293673420009;
    //     inc[80] = 0.27770020294376324;
    //     inc[81] = -0.13157747259006372;
    //     inc[82] = 0.43970600976350127;
    //     inc[83] = 0.8884514582601973;
    //     inc[84] = 0.6930829416939326;
    //     inc[85] = 0.7195988052369916;
    //     inc[86] = -0.04258632919587133;
    //     inc[87] = 0.2244168145419607;
    //     inc[88] = 0.1905206668424083;
    //     inc[89] = -0.9556876942059908;
    //     inc[90] = 0.41934025505508316;
    //     inc[91] = 0.11254183896121271;
    //     inc[92] = 0.9008263345248995;
    //     inc[93] = -0.41947141576481733;
    //     inc[94] = -0.11242811729205947;
    //     inc[95] = -0.9007794679045603;
    // }

    H.UseDiagonal(true);

    Real cg_tol = static_cast<Real>(0.00001);
    Real gmres_tol = static_cast<Real>(0.0001);

    // Complex* neumann_data_scat_ptr = nullptr;

    // const Real* B = H.VertexCoordinates();
    // for (int i = 0; i < 16 * n; i++)
    // {
    //     B[i] = Complex(1.0f,2.0f);
    // }
    // Real size_x = 2;
    // Real size_y = 2;
    // Real size_z = 2;

    // for (int i = 0 ; i < grid_coarse; i++)
    // {
    //     for (int j = 0 ; j < grid_coarse; j++)
    //     {
    //         for (int k = 0 ; k < grid_coarse; k++)
    //         {
    //             evaluation_points_1[3 * grid_coarse_2 * i + 3 * grid_coarse * j + 3 * k + 0]  = -size_x + k*( 2 * size_x )/( grid_coarse - 1 );
    //             evaluation_points_1[3 * grid_coarse_2 * i + 3 * grid_coarse * j + 3 * k + 1]  = -size_y + j*( 2 * size_y )/( grid_coarse - 1 );
    //             evaluation_points_1[3 * grid_coarse_2 * i + 3 * grid_coarse * j + 3 * k + 2]  = -size_z + i*( 2 * size_z )/( grid_coarse - 1 );
    //         }
    //     }
    // }

    //plane for near field measurements specified by base point, span direction 1 and span direction 2
    // Real* plane = (Real*)malloc(3 * 3 * sizeof(Real));
    
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

    // Real size_dir_1 = 2;
    // Real size_dir_2 = 2;

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

    // H.type_cast(B,H.VertexCoordinates(),3*n,4);
    // H.CreateIncidentWave_PL(Complex(1.0f,0.0f), inc, wave_chunk_size,
    //                         Complex(0.0f,0.0f), B, wave_count,
    //                         kappa, wave_coeff, wave_count, wave_chunk_size,
    //                         BAEMM::WaveType::Plane
    //                         );

    // H.BoundaryPotential<wave_count>( kappa, coeff, B, phi, 
    //                                         wave_chunk_count, wave_chunk_size, cg_tol, gmres_tol );



    H.FarField<32>( kappa, wave_chunk_count, inc, wave_chunk_size,
                            C, BAEMM::WaveType::Plane, cg_tol, gmres_tol);

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
    Real abs = 0;
    Real factor = ( 2 * Scalar::Pi<Real> ) / (wave_count * 2562);
    // std::cout << factor << std::endl;
    // std::cout << C[0] << std::endl;
    for(int i = 0 ; i < wave_count * 2562 ; i++)
    {
        abs += ( C[i].real() )* ( C[i].real() ) + ( C[i].imag() )* ( C[i].imag() ) ;
    }
    abs *= factor;
    abs = std::sqrt(abs);

    std::cout << abs << std::endl;
    // std::ofstream fout_points_3D("blub_eval_points_3D.txt");
    // std::ofstream fout_points_plane("bunny_eval_points_plane_3.txt");
    // if(fout_points_3D.is_open() && fout_points_plane.is_open())
	// {
	// 	for(int i = 0; i < grid_coarse_3 ; i++)
	// 	{
    //         for(int j = 0; j < 3 ; j++)
    //         {
    //         }
    //         fout_points_3D << "\n";
	// 	}
    //     fout_points_3D.close();
    //     for(int i = 0; i < grid_fine_2 ; i++)
	// 	{
    //         for(int j = 0; j < 3 ; j++)
    //         {
    //             fout_points_plane << evaluation_points_2[i * 3 + j] << " "; 
    //         }
    //         fout_points_plane << "\n";
	// 	}
    //     fout_points_plane.close();
	// }

    // std::ofstream fout_eval_3D("blub_eval_3D.txt");
    // std::ofstream fout_eval_plane("blub_eval_plane.txt");

    // if(fout_eval_3D.is_open() && fout_eval_plane.is_open() )
	// {
	// 	for(int i = 0; i < grid_coarse_3 ; i++)
	// 	{
    //         fout_eval_3D << C_1[i].real() << " "; 
    //         fout_eval_3D << C_1[i].imag() << " "; 

    //         fout_eval_3D<< "\n";
	// 	}
    //     fout_eval_3D.close();

    //     for(int i = 0; i < grid_fine_2 ; i++)
	// 	{
    //         fout_eval_plane << C_2[i].real() << " "; 
    //         fout_eval_plane << C_2[i].imag() << " ";

    //         fout_eval_plane << "\n";
	// 	}
    //     fout_eval_plane.close();
	// }

    // free(B);
    free(C);
    // free(C_2);
    // free(evaluation_points_1);
    // free(evaluation_points_2);
    free(inc);
    free(kappa);
    // free(coeff);
    // free(wave_coeff);
    return 0;
}
