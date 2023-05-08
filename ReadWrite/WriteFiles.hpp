using namespace Tools;
using namespace Tensors;
using namespace std;

template<typename R_ext, typename I_ext>
void WriteFixes(
    const I_ext                 & vertex_count,
    const I_ext                 & simplex_count,
    const I_ext                 & meas_count,
    const I_ext                 & wave_chunk_count,
    const I_ext                 & wave_chunk_size,
    Tensor2<I_ext,I_ext>        & simplices,
    Tensor2<R_ext,I_ext>        & meas_directions,
    Tensor2<R_ext,I_ext>        & incident_directions,
    Tensor1<R_ext,I_ext>        & kappa
)
{   
    ofstream s("data.txt");

    if( !s.good() )
    {
        eprint("ReadFromFile: File data.txt could not be opened.");
        
        return;
    }

    s << "vertex_count " << vertex_count << "\n";

    s << "simplex_count " << simplex_count << "\n";

    s << "meas_count " << meas_count << "\n";

    s << "wave_chunk_size " << wave_chunk_size << "\n";

    s << "wave_chunk_count " << wave_chunk_count << "\n";

    ptr<I_ext>       S = simplices.data();   
    ptr<R_ext>       M = meas_directions.data();
    ptr<R_ext>       I = incident_directions.data();
    ptr<R_ext>       K = kappa.data();

    for(int i = 0; i < wave_chunk_count ; i++)
    {
        s << K[i] << " "; 
    }
    
    for(int i = 0; i < wave_chunk_size ; i++)
    {
        for(int j = 0; j < 3 ; j++)
        {
            s << I[i * 3 + j] << " "; 
        }
        s << "\n";
    }
    
    fstream file;
    file.open("simplices.bin",ios::out | ios::binary );
    file.write( (char*)S , sizeof(I_ext) * 3 * simplex_count );
    if( !file )
    {
        eprint("WriteToFile: File simplices.bin could not be opened.");
        
        return;
    }
    file.close();

    file.open("meas_directions.bin",ios::out | ios::binary );
    file.write( (char*)M , sizeof(R_ext) * 3 * meas_count );
    if( !file )
    {
        eprint("WriteToFile: File meas_directions.bin could not be opened.");
        
        return;
    }
    file.close();
}


template<typename R_ext, typename I_ext>
void WriteCoordinates(
    const I_ext                 & vertex_count,
    Tensor2<R_ext,I_ext>        & coords
)
{ 
    ptr<R_ext> V = coords.data();
    
    fstream file;
    file.open("coords.bin",ios::out | ios::binary );
    file.write( (char*)V , sizeof(R_ext) * 3 * vertex_count );
    if( !file )
    {
        eprint("WriteToFile: File coords.bin could not be opened.");
        
        return;
    }
    file.close();
}


template<typename T, typename I_ext>
void WriteInOut(
    const I_ext                     & rows,
    const I_ext                     & columns,
    Tensor2<T,I_ext>                & B_out
)
{   
    ptr<T> B = B_out.data();
    
    fstream file;
    file.open("B.bin",ios::out | ios::binary );
    file.write( (char*)B , sizeof(T) * rows * columns );
    if( !file )
    {
        eprint("WriteToFile: File B.bin could not be opened.");
        
        return;
    }
    file.close();
}
