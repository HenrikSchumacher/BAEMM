namespace BAEMM
{
    template<typename Real, typename Int>
    void ReadMeshFromFile(
                          const std::string & file_name,
                          Tensor2<Real,Int> & coords,
                          Tensor2<Int,Int>  & simplices
                          )
    {
        ptic("ReadMeshFromFile");
        
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
        dump(dom_dim);
        s >> str;
        s >> amb_dim;
        dump(amb_dim);
        s >> str;
        s >> vertex_count;
        dump(vertex_count);
        s >> str;
        s >> simplex_count;
        dump(simplex_count);
        
        const Int simplex_size = dom_dim+1;
        
        dump(simplex_size);
        
        coords    = Tensor2<Real,Int>(vertex_count, amb_dim     );
        simplices = Tensor2<Int, Int>(simplex_count,simplex_size);
        
        mptr<Real> V = coords.data();
        mptr<Int>     S = simplices.data();
        
        
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
        
        ptoc("ReadMeshFromFile");
    }
    
} // namespace BAEMM
