#include <iostream>

using namespace Tools;
using namespace Tensors;
using namespace std;

template<typename R_ext, typename I_ext>
void ReadFixes(
    I_ext                 & vertex_count,
    I_ext                 & simplex_count,
    I_ext                 & meas_count,
    I_ext                 & wave_chunk_count,
    I_ext                 & wave_chunk_size,
    Tensor2<I_ext,I_ext>  & simplices,
    Tensor2<R_ext,I_ext>  & meas_directions,
    Tensor2<R_ext,I_ext>  & incident_directions,
    Tensor1<R_ext,I_ext>  & kappa
)
{
    ifstream s ("data.txt");
    
    if( !s.good() )
    {
        eprint("ReadFromFile: File data.txt could not be opened.");
        
        return;
    }
    
    string str;
    
    s >> str;
    s >> vertex_count;

    s >> str;
    s >> simplex_count;
    
    s >> str;
    s >> meas_count;
    
    s >> str;
    s >> wave_chunk_size;
    
    s >> str;
    s >> wave_chunk_count;

    valprint("simplex_count",simplex_count);
    valprint("meas_count",meas_count);
    valprint("wave_chunk_size",wave_chunk_size);
    valprint("wave_chunk_count",wave_chunk_count);   
    
    simplices            = Tensor2<I_ext, I_ext>(  simplex_count,3   );
    meas_directions      = Tensor2<R_ext,I_ext>(  meas_count, 3     );
    incident_directions  = Tensor2<R_ext,I_ext>(  wave_chunk_size, 3     );
    kappa                = Tensor1<R_ext,I_ext>(  wave_chunk_count );
    
    mut<I_ext>       S = simplices.data();   
    mut<R_ext>       M = meas_directions.data();
    mut<R_ext>       I = incident_directions.data();
    mut<R_ext>       K = kappa.data();

    for( int i = 0; i < wave_chunk_count; ++i )
    {

        s >> K[i];
    }

    for( int i = 0; i < wave_chunk_size; ++i )
    {
        for( int k = 0; k < 3; ++k )
        {
            s >> I[i * 3 + k];
        }
    }
    
    fstream file;
    file.open("simplices.bin",ios::in | ios::binary );
    file.read( (char*)S , sizeof(I_ext) * 3 * simplex_count );
    if( !file )
    {
        eprint("ReadFromFile: File simplices.bin could not be opened.");
        
        return;
    }
    file.close();

    file.open("meas_directions.bin",ios::in | ios::binary );
    file.read( (char*)M , sizeof(R_ext) * 3 * meas_count );
    if( !file )
    {
        eprint("ReadFromFile: File meas_directions.bin could not be opened.");
        
        return;
    }
    file.close();
}


template<typename R_ext, typename I_ext>
void ReadCoordinates(
    I_ext                 & vertex_count,
    Tensor2<R_ext,I_ext>  & coords
)
{
    coords    = Tensor2<R_ext,I_ext>(   vertex_count, 3     );
    
    mut<R_ext> V = coords.data();
    
    fstream file;
    file.open("coords.bin",ios::in | ios::binary );
    file.read( (char*)V , sizeof(R_ext) * 3 * vertex_count );
    if( !file )
    {
        eprint("ReadFromFile: File coords.bin could not be opened.");
        
        return;
    }
    file.close();
}


template<typename T, typename I_ext>
void ReadInOut(
    I_ext                     & rows,
    I_ext                     & columns,
    Tensor2<T,I_ext>          & B_in
)
{   
    B_in    = Tensor2<T,I_ext>(     columns, rows    );
    
    mut<T> B = B_in.data();
    
    fstream file;
    file.open("B.bin",ios::in | ios::binary );
    file.read( (char*)B , sizeof(T) * rows * columns );
    if( !file )
    {
        eprint("ReadFromFile: File B.bin could not be opened.");
        
        return;
    }
    file.close();
}
