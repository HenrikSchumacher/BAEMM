public:
    
    void Initialize_CPU()
    {
        ptic(ClassName()+"::Initialize_CPU");
        
        areas      = Tensor1<Real,Int>( simplex_count    );
        mid_points = Tensor2<Real,Int>( simplex_count, 4 );
        normals    = Tensor2<Real,Int>( simplex_count, 4 );
        
        areas_ptr      = areas.data();
        mid_points_ptr = mid_points.data();
        normals_ptr    = normals.data();
        
        
        ptoc(ClassName()+"::Initialize_CPU");
    }
