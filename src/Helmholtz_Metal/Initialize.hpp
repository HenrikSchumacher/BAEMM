public:
    
    void Initialize()
    {
        tic(ClassName()+"::Initialize");
        
        const uint size  =     simplex_count * sizeof(Real);
        const uint size4 = 4 * simplex_count * sizeof(Real);
        
        areas      = device->newBuffer(size,  MTL::ResourceStorageModeManaged);
        mid_points = device->newBuffer(size4, MTL::ResourceStorageModeManaged);
        normals    = device->newBuffer(size4, MTL::ResourceStorageModeManaged);
        
        mut<Real> areas_      = static_cast<Real *>(     areas->contents());
        mut<Real> mid_points_ = static_cast<Real *>(mid_points->contents());
        mut<Real> normals_    = static_cast<Real *>(   normals->contents());

        // For assembling AvOp.
        Tensor1<LInt,    Int> outer ( simplex_count + 1 );
        Tensor1<Int,    LInt> inner ( 3 * simplex_count );
        Tensor1<Complex,LInt> vals  ( 3 * simplex_count );
        outer[0] = 0;
        
        // For assembling MassMatrix.
        Tensor3<Int,    LInt> i_list ( simplex_count, 3, 3 );
        Tensor3<Int,    LInt> j_list ( simplex_count, 3, 3 );
        Tensor3<Complex,LInt> a_list ( simplex_count, 3, 3 );

        
        // We pad 3-vector with an additional float so that we can use float3 in the metal kernels. (float3 has size 4 * 4 Byte to preserve alignement.)
        #pragma omp parallel for num_threads( OMP_thread_count ) schedule( static )
        for( Int i = 0; i < simplex_count; ++i )
        {
            Tiny::Vector<4,Real,Int> x;
            Tiny::Vector<4,Real,Int> y;
            Tiny::Vector<4,Real,Int> z;
            Tiny::Vector<4,Real,Int> nu;
            
            Int i_0 = triangles(i,0);
            Int i_1 = triangles(i,1);
            Int i_2 = triangles(i,2);
            
            x[0] = vertex_coords(i_0,0);
            x[1] = vertex_coords(i_0,1);
            x[2] = vertex_coords(i_0,2);
            
            y[0] = vertex_coords(i_1,0);
            y[1] = vertex_coords(i_1,1);
            y[2] = vertex_coords(i_1,2);
            
            z[0] = vertex_coords(i_2,0);
            z[1] = vertex_coords(i_2,1);
            z[2] = vertex_coords(i_2,2);
            
            mid_points_[4*i+0] = third * ( x[0] + y[0] + z[0] );
            mid_points_[4*i+1] = third * ( x[1] + y[1] + z[1] );
            mid_points_[4*i+2] = third * ( x[2] + y[2] + z[2] );
            mid_points_[4*i+3] = zero;
            
            y[0] -= x[0]; y[1] -= x[1]; y[2] -= x[2];
            z[0] -= x[0]; z[1] -= x[1]; z[2] -= x[2];
            
            nu[0] = y[1] * z[2] - y[2] * z[1];
            nu[1] = y[2] * z[0] - y[0] * z[2];
            nu[2] = y[0] * z[1] - y[1] * z[0];

            const Real a = half * std::sqrt( nu[0] * nu[0] + nu[1] * nu[1] + nu[2] * nu[2] );
            areas_[i] = a;
            nu /= a;

            normals_[4*i+0] = nu[0];
            normals_[4*i+1] = nu[1];
            normals_[4*i+2] = nu[2];
            normals_[4*i+3] = zero;
            
            const Complex a_over_3  = a * third;
            
            outer[i+1]   = 3 * i;
            inner[3*i+0] = i_0;
            inner[3*i+1] = i_1;
            inner[3*i+2] = i_2;
            vals [3*i+0] = a_over_3;
            vals [3*i+1] = a_over_3;
            vals [3*i+2] = a_over_3;
            
            
            i_list(i,0,0) = i_0;
            i_list(i,0,1) = i_0;
            i_list(i,0,2) = i_0;
            i_list(i,1,0) = i_1;
            i_list(i,1,1) = i_1;
            i_list(i,1,2) = i_1;
            i_list(i,2,0) = i_2;
            i_list(i,2,1) = i_2;
            i_list(i,2,2) = i_2;
            
            j_list(i,0,0) = i_0;
            j_list(i,0,1) = i_1;
            j_list(i,0,2) = i_2;
            j_list(i,1,0) = i_0;
            j_list(i,1,1) = i_1;
            j_list(i,1,2) = i_2;
            j_list(i,2,0) = i_0;
            j_list(i,2,1) = i_1;
            j_list(i,2,2) = i_2;
            
            const Complex a_over_6  = a * sixth;
            const Complex a_over_12 = a * twelveth;
            
            a_list(i,0,0) = a_over_6;
            a_list(i,0,1) = a_over_12;
            a_list(i,0,2) = a_over_12;
            a_list(i,1,0) = a_over_12;
            a_list(i,1,1) = a_over_6;
            a_list(i,1,2) = a_over_12;
            a_list(i,2,0) = a_over_12;
            a_list(i,2,1) = a_over_12;
            a_list(i,2,2) = a_over_6;
        }
        
             areas->didModifyRange({0,size });
        mid_points->didModifyRange({0,size4});
           normals->didModifyRange({0,size4});
        
        AvOp = Sparse_T(
            std::move(outer), std::move(inner), std::move(vals),
            simplex_count, vertex_count,
            OMP_thread_count
        );
        
        AvOp.SortInner();

        AvOpTransp = AvOp.Transpose();
        
        Mass = Sparse_T(
            i_list.data(), j_list.data(), a_list.data(),
            vertex_count, vertex_count,
            static_cast<LInt>(9) * static_cast<LInt>(simplex_count),
            OMP_thread_count, true, false
        );
        
        command_queue = device->newCommandQueue();
        
        if( command_queue == nullptr )
        {
            std::cout << "Failed to find the command queue." << std::endl;
            return;
        }
        
        
        toc(ClassName()+"::Initialize");
    }
