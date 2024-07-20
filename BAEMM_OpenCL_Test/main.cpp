#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <pwd.h>

#define TOOLS_ENABLE_PROFILER

#include "../Helmholtz_OpenCL.hpp"
#include "../ReadMeshFromFile.hpp"

using namespace Tools;
using namespace Tensors;
using namespace Repulsor;
using namespace BAEMM;

using Int     = int;
using Real    = double;
using Complex = std::complex<Real>;
using LInt    = long long;
using SReal   = Real;
using ExtReal = Real;


int main()
{
    const char * homedir = getenv("HOME");
    
    if( homedir == nullptr)
    {
        homedir = getpwuid(getuid())->pw_dir;
    }
    std::string path ( homedir );
    
    Profiler::Clear(path);
    
    Int thread_count = 8;
    
    
//    std::vector<Device> devices;
//    
//    size_t find_devices()
//    {
//        std::vector<Platform> platforms; // get all platforms
//        
//        std::vector<Device> devices_available;
//        
//        size_t n = 0; // number of available devices
//        
//        Platform::get(&platforms);
//        
//        for( size_t i = 0; i < platforms.size(); i++ )
//        {
//            devices_available.clear();
//            platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &devices_available);
//            
//            if(devices_available.size()==0)
//            {
//                continue; // no device found in plattform i
//            }
//            
//            for( size_t j = 0; j < devices_available.size(); j++ )
//            {
//                n++;
//                devices.push_back(devices_available[j]);
//            }
//        }
//        
//        if( (platforms.size()==0) || (devices.size()==0) )
//        {
//            std::cout << "Error: There are no OpenCL devices available!" << std::endl;
//            return -1;
//        }
//        
//        for( size_t i = 0; i < n; i++)
//        {
//            std::cout << "ID: " << i << ", Device: " << devices[i].getInfo<CL_DEVICE_NAME>() << std::endl;
//        }
//        
//        return n; // return number of available devices
//    }
    
    std::string file_name = "/Users/Henrik/github/BAEMM/Meshes/Spot_00023424T.txt";

    Tensor2<Real,Int> coords;
    Tensor2<Int, Int> simplices;

    ReadMeshFromFile<Real,Int>(file_name, coords, simplices);

    Tensor2<Real,Int> meas_directions;
    Tensor2<Int, Int> simplices_meas;

    std::string meas_file_name = "/Users/Henrik/github/BAEMM/Meshes/Sphere_00005120T.txt";
    
    ReadMeshFromFile<Real,Int>(meas_file_name, meas_directions, simplices_meas);

    print("A");
    
    Int device = 1;
    
    BAEMM::Helmholtz_OpenCL H (
        coords.data(),          coords.Dimension(0),
        simplices.data(),       simplices.Dimension(0),
        meas_directions.data(), meas_directions.Dimension(0), device, thread_count
    );
    
    Int n = H.VertexCount();
    
    const     Int meas_count = meas_directions.Dimension(0);
    
    constexpr Int wave_count = 32;
    constexpr Int wave_chunk_size = 16;
    constexpr Int wave_chunk_count = wave_count/wave_chunk_size;
    
    Tensor2<Complex,Int> C ( meas_count, wave_count );
    
    Tensor1<Real,Int> kappa ( wave_chunk_count    );
    Tensor1<Real,Int> inc   ( wave_chunk_size * 3 );
    
    // Complex * coeff = (Complex*)malloc(4 * wave_chunk_count * sizeof(Complex));
    // Complex * wave_coeff = (Complex*)malloc(4 * wave_chunk_count * sizeof(Complex));

    for (int i = 0 ; i < wave_chunk_count; i++)
    {
        kappa[i] = ( 1 + 2 * ( i + 2 ) ) * Scalar::Pi<Real>;
    }

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

    H.UseDiagonal(true);

    Real cg_tol    = static_cast<Real>(0.00001);
    Real gmres_tol = static_cast<Real>(0.0001);


    tic("FarField");
    H.FarField<wave_count>(
        kappa.data(), wave_chunk_count,
        inc.data(),   wave_chunk_size,
        C.data(), 
        BAEMM::Helmholtz_OpenCL::WaveType::Plane, cg_tol, gmres_tol
    );
    toc("FarField");

    // H.DestroyKernel(&list);
    Real abs = 0;
    Real factor = ( 2 * Scalar::Pi<Real> ) / (meas_count * wave_count );

    abs = std::sqrt(factor) * C.FrobeniusNorm();

    std::cout << abs << std::endl;
    
    return 0;
}
