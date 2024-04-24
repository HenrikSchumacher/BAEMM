#include <iostream>

#include <sys/types.h>
#include <complex>


using uint = unsigned int;


// #include "../Helmholtz_CPU.hpp"
#include "../Helmholtz_OpenCL.hpp"

using namespace Tools;
using namespace Tensors;

using Real = double;
using Int = int;

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


// BAEMM::Helmholtz_CPU read_CPU(const std::string & filename)
// {
//     using namespace Tensors;
//     using namespace Tools;

//     const std::string path = "/HOME1/users/guests/jannr";
//     std::string file_name = path + filename;
//     std::cout << file_name << std::endl;
//     Tensor2<Real, Int> coords;
//     Tensor2<Int, Int> simplices;
//     tic("read");
//     ReadFromFile<Real, Int>(file_name, coords, simplices);
//     toc("read");
//     BAEMM::Helmholtz_CPU H_CPU (
//         coords.data(),    coords.Dimension(0),
//         simplices.data(), simplices.Dimension(0), 8
//     );
//     return H_CPU;
// }

BAEMM::Helmholtz_OpenCL read_OpenCL(const std::string & filename)
{
    using namespace Tensors;
    using namespace Tools;

    const std::string path = "/HOME1/users/guests/jannr";
    std::string file_name = path + filename;

    Tensor2<Real, Int> coords;
    Tensor2<Int, Int> simplices;

    ReadFromFile<Real, Int>(file_name, coords, simplices);

    Tensor2<Real, Int> meas_directions;
    Tensor2<Int, Int> simplices_meas;

    ReadFromFile<Real, Int>(path + "/github/BAEMM/Meshes/Sphere_00005120T.txt", meas_directions, simplices_meas);


    BAEMM::Helmholtz_OpenCL H (
        coords.data(),    coords.Dimension(0),
        simplices.data(), simplices.Dimension(0), 
        meas_directions.data(),    meas_directions.Dimension(0), 3, 8
    );
    return H;
}