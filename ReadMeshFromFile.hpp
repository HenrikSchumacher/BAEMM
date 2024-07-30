namespace BAEMM
{
    template<typename Real, typename Int>
    bool ReadMeshFromFile(
        const std::filesystem::path & file,
        mref<Tensor2<Real,Int>> coords,
        mref<Tensor2<Int, Int>> simplices
    )
    {
        std::string tag = "ReadMeshFromFile";
        
        ptic(tag);
        
        print("Reading mesh from file " + file.string() + ".");
        
        std::ifstream s ( file.string() );
        
        if( !s.good() )
        {
            eprint( tag + ": File " + file.string() + " could not be opened.");
            ptoc(tag);
            return false;
        }
        
        std::string str;
        Int amb_dim;
        Int dom_dim;
        Int vertex_count;
        Int simplex_count;
        s >> str;
        if( str != "dom_dim" )
        {
            dump(str);
            eprint( tag + ": Invalid file. Aborting.");
            ptoc(tag);
            return false;
        }
        s >> dom_dim;
        dump(dom_dim);
        
        s >> str;
        if( str != "amb_dim" )
        {
            dump(str);
            eprint( tag + ": Invalid file. Aborting.");
            ptoc(tag);
            return false;
        }
        s >> amb_dim;
        dump(amb_dim);
        
        s >> str;
        if( str != "vertex_count" )
        {
            dump(str);
            eprint( tag + ": Invalid file. Aborting.");
            ptoc(tag);
            return false;
        }
        s >> vertex_count;
        dump(vertex_count);
        
        s >> str;
        if( str != "simplex_count" )
        {
            dump(str);
            eprint( tag + ": Invalid file. Aborting.");
            ptoc(tag);
            return false;
        }
        s >> simplex_count;
        dump(simplex_count);
        
        const Int simplex_size = dom_dim+1;


        coords    = Tensor2<Real,Int>(vertex_count, amb_dim     );
        simplices = Tensor2<Int, Int>(simplex_count,simplex_size);
        
        mptr<Real> V = coords.data();
        mptr<Int>  S = simplices.data();
        
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
        
        ptoc(tag);
        
        return true;
    }
    
} // namespace BAEMM
