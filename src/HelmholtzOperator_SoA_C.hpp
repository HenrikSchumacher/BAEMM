namespace BAEMM
{
    
    template<class Mesh_T>
    class HelmholtzOperator_SoA_C
    {
    public:
        
        using Real       = typename Mesh_T::Real;
//        using Complex    = std::complex<Real>;
        using Int        = typename Mesh_T::Int;
        using SReal      = typename Mesh_T::SReal;
        using ExtReal    = typename Mesh_T::ExtReal;
//        using ExtComplex = std::complex<ExtReal>;
        
//        static constexpr Complex I {0,1};
        
        static constexpr Real one   = 1;
        static constexpr Real two   = 2;
        static constexpr Real three = 3;
        
        static constexpr Real half  = one / two;
        static constexpr Real third = one / three;
        
        
//        using Mesh_T = SimplicialMesh<2,3,Real,Int,Real,ExtReal>;
        
        HelmholtzOperator_SoA_C() = delete;
        
        explicit HelmholtzOperator_SoA_C( Mesh_T & M_ )
        :   M             ( M_ )
        ,   n             ( M.SimplexCount() )
        ,   vertex_coords ( M.VertexCoordinates().data(), M.VertexCount()   )
        ,   triangles     ( M.Simplices().data(),         M.SimplexCount()  )
        ,   mid_points    ( n )
        ,   areas         ( n )
        ,   normals       ( n )
        {
//            const Tensor2<Real,Int> & X = M.VertexCoordinates();
//            const Tensor2<Int, Int> & S = M.Simplices();
            
            Tiny::Vector<3,Real,Int> x;
            Tiny::Vector<3,Real,Int> y;
            Tiny::Vector<3,Real,Int> z;
            
            Tiny::Vector<3,Real,Int> nu;
            
            for( Int i = 0; i < n; ++i )
            {
                x.Read( vertex_coords, triangles(0,i) );
                y.Read( vertex_coords, triangles(1,i) );
                z.Read( vertex_coords, triangles(2,i) );

                mid_points(0,i) = third * ( x[0] + y[0] + z[0] );
                mid_points(1,i) = third * ( x[1] + y[1] + z[1] );
                mid_points(2,i) = third * ( x[2] + y[2] + z[2] );
                
                y -= x;
                z -= x;

                nu[0] = y[1] * z[2] - y[2] * z[1];
                nu[1] = y[2] * z[0] - y[0] * z[2];
                nu[2] = y[0] * z[1] - y[1] * z[0];

                const Real a = nu.Norm();
                areas[i] = a;

                nu /= a;

                nu.Write( normals, i );
            }
        }
        
        ~HelmholtzOperator_SoA_C() = default;
        
    protected:
        
        Mesh_T & M;
        
        const Int n;

        Tiny::VectorList<3,Real,Int> vertex_coords;
        Tiny::VectorList<3,Int ,Int> triangles;
        
        Tensor1<Real,Int> areas;
        Tiny::VectorList<3,Real,Int> mid_points;
        Tiny::VectorList<3,Real,Int> normals;
        
        
    public:
        
        
        
        
    public:
        
        std::string ClassName() const
        {
            return "HelmholtzOperator_SoA_C<"+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
        
    }; // HelmholtzOperator_SoA_C
    
} // namespace BAEMM

