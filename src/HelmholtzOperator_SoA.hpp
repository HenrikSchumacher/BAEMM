namespace BAEMM
{
    
    template<class Mesh_T>
    class HelmholtzOperator_SoA
    {
    public:
        
        using Real       = typename Mesh_T::Real;
        using Complex    = std::complex<Real>;
        using Int        = typename Mesh_T::Int;
        using SReal      = typename Mesh_T::SReal;
        using ExtReal    = typename Mesh_T::ExtReal;
        using ExtComplex = std::complex<ExtReal>;
        
        static constexpr Complex I {0,1};
        
        static constexpr Real one   = 1;
        static constexpr Real two   = 2;
        static constexpr Real three = 3;
        
        static constexpr Real half  = one / two;
        static constexpr Real third = one / three;
        
        HelmholtzOperator_SoA() = delete;
        
        explicit HelmholtzOperator_SoA( Mesh_T & M_ )
        :   M             ( M_ )
        ,   n             ( M.SimplexCount() )
        ,   vertex_coords ( M.VertexCoordinates().data(), M.VertexCount()   )
        ,   triangles     ( M.Simplices().data(),         M.SimplexCount()  )
        ,   mid_points    ( n )
        ,   areas         ( n )
        ,   normals       ( n )
        {
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
        
        ~HelmholtzOperator_SoA() = default;
        
    protected:
        
        Mesh_T & M;
        
        const Int n;

        Tiny::VectorList<3,Real,Int> vertex_coords;
        Tiny::VectorList<3,Int ,Int> triangles;
        
        Tensor1<Real,Int> areas;
        Tiny::VectorList<3,Real,Int> mid_points;
        Tiny::VectorList<3,Real,Int> normals;
        
        
    public:
        
        template< Int n_waves>
        void Multiply(
            ptr<Complex> Y,
            mut<Complex> X,
            const Real kappa,
            const Real kappa_step,
            const Int  thread_count
        ) const
        {
            tic(ClassName()+"::Multiply<"+ToString(n_waves)+">");
            
            JobPointers<Int> job_ptr(n, thread_count);
            
            #pragma omp parallel for num_threads( thread_count)
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                Tiny::Vector<3,Real,Int> x;
                Tiny::Vector<3,Real,Int> y;
                Tiny::Vector<3,Real,Int> z;
//                Tiny::Vector<3,Real,Int> nu;
                
                Tiny::Vector<n_waves,Complex,Int> X_i;
                
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    x.Read( mid_points, i );
                 
                    X_i.SetZero();
                    
                    for( Int j = 0; j < n; ++j )
                    {
                        const Real delta = static_cast<Real>(i==j);
                        
                        y.Read( mid_points, j );
                        
                        z[0] = y[0] - x[0];
                        z[1] = y[1] - x[1];
                        z[2] = y[2] - x[2];
                        
                        const Real r = z.Norm() + delta;
                        
                        const Real r_inv = (one-delta) / r;
                        
                        Complex factor (
                            std::cos(kappa * r) * r_inv,
                            std::sin(kappa * r) * r_inv
                        );

                        const Complex multiplier (
                            std::cos(kappa_step * r),
                            std::sin(kappa_step * r)
                        );
                        
                        X_i[0] += factor * Y[n_waves * j + 0];
                        
                        LOOP_UNROLL_FULL
                        for( Int k = 1; k < n_waves; ++k )
                        {
                            factor *= multiplier;

                            X_i[k] += factor * Y[n_waves * j + k];
                        }
                    }
                    
                    X_i.Write( &X[n_waves*i] );
                }
            }
            
            toc(ClassName()+"::Multiply<"+ToString(n_waves)+">");
        }
        
        
        template< Int i_blk_size, Int j_blk_size, Int n_waves>
        void Multiply_Blocked(
            ptr<Complex> Y,
            mut<Complex> X,
            const Real kappa,
            const Real kappa_step,
            const Int  thread_count
        ) const
        {
            tic(ClassName()+"::Multiply_Blocked<"+ToString(i_blk_size)+","+ToString(j_blk_size)+","+ToString(n_waves)+">");
            
            constexpr Int i_chunk = i_blk_size * n_waves;
            constexpr Int j_chunk = j_blk_size * n_waves;
            
            if( (n/i_blk_size) * i_blk_size != n )
            {
                wprint(ClassName()+"::Multiply_Blocked: Loop peeling not applied.");
            }

            JobPointers<Int> job_ptr(n/i_blk_size, thread_count);
            
            #pragma omp parallel for num_threads( thread_count)
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                Tiny::Vector<3,Real,Int> x;
                Tiny::Vector<3,Real,Int> y;
                Tiny::Vector<3,Real,Int> z;
//                Tiny::Vector<3,Real,Int> nu;
                
                Tiny::Matrix<i_blk_size,n_waves,Complex,Int> X_blk;
                Tiny::Matrix<j_blk_size,n_waves,Complex,Int> Y_blk;
                
                const Int i_blk_begin = job_ptr[thread  ];
                const Int i_blk_end   = job_ptr[thread+1];
                
                const Int j_blk_begin = 0;
                const Int j_blk_end   = n/j_blk_size;
                
                for( Int i_blk = i_blk_begin; i_blk < i_blk_end; ++i_blk )
                {
                    const Int i_base = i_blk_size * i_blk;
                    
                    X_blk.SetZero();

                    for( Int j_blk = j_blk_begin; j_blk < j_blk_end; ++j_blk )
                    {
                        const Int j_base = j_blk_size * j_blk;
                        
                        Y_blk.Read( &Y[j_chunk * j_blk] );
                        
                        for( Int i_loc = 0; i_loc < i_blk_size; ++i_loc )
                        {
                            const Int i = i_base + i_loc;

                            x.Read( mid_points, i );

                            for( Int j_loc = 0; j_loc < j_blk_size; ++j_loc )
                            {
                                const Int j = j_base + j_loc;

                                const Real delta = static_cast<Real>(i==j);

                                y.Read( mid_points, j );

                                z[0] = y[0] - x[0];
                                z[1] = y[1] - x[1];
                                z[2] = y[2] - x[2];

                                const Real r = z.Norm() + delta;

                                const Real r_inv = (one-delta) / r;
                                
                                Complex factor (
                                    std::cos(kappa * r) * r_inv,
                                    std::sin(kappa * r) * r_inv
                                );
                                
                                const Complex multiplier (
                                    std::cos(kappa_step * r),
                                    std::sin(kappa_step * r)
                                );

                                X_blk[i_loc][0] += factor * Y_blk[j_loc][0];

                                LOOP_UNROLL_FULL
                                for( Int k = 1; k < n_waves; ++k )
                                {
                                    factor *= multiplier;
                                    X_blk[i_loc][k] += factor * Y_blk[j_loc][k];
                                }
                            }
                        }
                    
                    }

                    X_blk.Write( &X[i_chunk * i_blk] );
                }
            }
            
            toc(ClassName()+"::Multiply_Blocked<"+ToString(i_blk_size)+","+ToString(j_blk_size)+","+ToString(n_waves)+">");
        }
        
        template< Int n_waves>
        void Multiply_C(
            ptr<Real> Re_Y, ptr<Real> Im_Y,
            mut<Real> Re_X, mut<Real> Im_X,
            const Real kappa,
            const Real kappa_step,
            const Int  thread_count
        ) const
        {
            tic(ClassName()+"::Multiply_C<"+ToString(n_waves)+">");
            
            JobPointers<Int> job_ptr(n, thread_count);
            
            #pragma omp parallel for num_threads( thread_count)
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                Tiny::Vector<3,Real,Int> x;
                Tiny::Vector<3,Real,Int> y;
                Tiny::Vector<3,Real,Int> z;
//                Tiny::Vector<3,Real,Int> nu;
                
                Tiny::Vector<n_waves,Real,Int> Re_X_i;
                Tiny::Vector<n_waves,Real,Int> Im_X_i;
                
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    x.Read( mid_points, i );
                 
                    Re_X_i.SetZero();
                    Im_X_i.SetZero();
                    
                    for( Int j = 0; j < n; ++j )
                    {
                        const Real delta = static_cast<Real>(i==j);
                        
                        y.Read( mid_points, j );
                        
                        z[0] = y[0] - x[0];
                        z[1] = y[1] - x[1];
                        z[2] = y[2] - x[2];
                        
                        const Real r = z.Norm() + delta;
                        
                        const Real r_inv = (one-delta) / r;
                        
                        Real Re_factor = std::cos(kappa * r) * r_inv;
                        Real Im_factor = std::sin(kappa * r) * r_inv;

                        Real Re_multiplier = std::cos(kappa_step * r);
                        Real Im_multiplier = std::sin(kappa_step * r);
                        
                        ptr<Real> Re_Y_j = &Re_Y[n_waves * j];
                        ptr<Real> Im_Y_j = &Im_Y[n_waves * j];
                        
                        cfma(
                            Re_factor, Im_factor,
                            Re_Y_j[0], Im_Y_j[0],
                            Re_X_i[0], Im_X_i[0]
                        );
                        
                        LOOP_UNROLL_FULL
                        for( Int k = 1; k < n_waves; ++k )
                        {
                            cmulby(
                                Re_factor,     Im_factor,
                                Re_multiplier, Im_multiplier
                            );
                            
                            cfma(
                                Re_factor, Im_factor,
                                Re_Y_j[k], Im_Y_j[k],
                                Re_X_i[k], Im_X_i[k]
                            );
                        }
                    }
                                        
                    Re_X_i.Write( &Re_X[n_waves*i] );
                    Im_X_i.Write( &Im_X[n_waves*i] );
                }
            }
            
            toc(ClassName()+"::Multiply_C<"+ToString(n_waves)+">");
        }
        
        
        template< Int i_blk_size, Int j_blk_size, Int n_waves>
        void Multiply_Blocked_C(
            ptr<Real> Re_Y, ptr<Real> Im_Y,
            mut<Real> Re_X, mut<Real> Im_X,
            const Real kappa,
            const Real kappa_step,
            const Int  thread_count
        ) const
        {
            tic(ClassName()+"::Multiply_Blocked_C<"+ToString(i_blk_size)+","+ToString(j_blk_size)+","+ToString(n_waves)+">");
            
            constexpr Int i_chunk = i_blk_size * n_waves;
            constexpr Int j_chunk = j_blk_size * n_waves;
            
            if( (n/i_blk_size) * i_blk_size != n )
            {
                wprint(ClassName()+"::Multiply_Blocked_C: Loop peeling not applied.");
            }

            JobPointers<Int> job_ptr(n/i_blk_size, thread_count);
            
            #pragma omp parallel for num_threads( thread_count)
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                Tiny::Vector<3,Real,Int> x;
                Tiny::Vector<3,Real,Int> y;
                Tiny::Vector<3,Real,Int> z;
//                Tiny::Vector<3,Real,Int> nu;
                
                Tiny::Matrix<i_blk_size,n_waves,Real,Int> Re_X_blk;
                Tiny::Matrix<i_blk_size,n_waves,Real,Int> Im_X_blk;
                
                Tiny::Matrix<j_blk_size,n_waves,Real,Int> Re_Y_blk;
                Tiny::Matrix<j_blk_size,n_waves,Real,Int> Im_Y_blk;
                
                const Int i_blk_begin = job_ptr[thread  ];
                const Int i_blk_end   = job_ptr[thread+1];
                
                const Int j_blk_begin = 0;
                const Int j_blk_end   = n/j_blk_size;
                
                for( Int i_blk = i_blk_begin; i_blk < i_blk_end; ++i_blk )
                {
                    const Int i_base = i_blk_size * i_blk;
                    
                    Re_X_blk.SetZero();
                    Im_X_blk.SetZero();

                    for( Int j_blk = j_blk_begin; j_blk < j_blk_end; ++j_blk )
                    {
                        const Int j_base = j_blk_size * j_blk;
                        
                        Re_Y_blk.Read( &Re_Y[j_chunk * j_blk] );
                        Im_Y_blk.Read( &Im_Y[j_chunk * j_blk] );
                        
                        for( Int i_loc = 0; i_loc < i_blk_size; ++i_loc )
                        {
                            const Int i = i_base + i_loc;

                            x.Read( mid_points, i );

                            for( Int j_loc = 0; j_loc < j_blk_size; ++j_loc )
                            {
                                const Int j = j_base + j_loc;

                                const Real delta = static_cast<Real>(i==j);

                                y.Read( mid_points, j );

                                z[0] = y[0] - x[0];
                                z[1] = y[1] - x[1];
                                z[2] = y[2] - x[2];

                                const Real r = z.Norm() + delta;

                                const Real r_inv = (one-delta) / r;
                                
                                const Real kappa_r      = kappa * r;
                                const Real kappa_step_r = kappa_step * r;
                                
                                Real Re_factor = std::cos(kappa_r) * r_inv;
                                Real Im_factor = std::sin(kappa_r) * r_inv;

                                const Real Re_multiplier = std::cos(kappa_step_r);
                                const Real Im_multiplier = std::sin(kappa_step_r);
                                
                                cfma(
                                    Re_factor,          Im_factor,
                                    Re_Y_blk[j_loc][0], Im_Y_blk[j_loc][0],
                                    Re_X_blk[i_loc][0], Im_X_blk[i_loc][0]
                                );
                                
                                LOOP_UNROLL_FULL
                                for( Int k = 1; k < n_waves; ++k )
                                {
                                    cmulby(
                                        Re_factor,     Im_factor,
                                        Re_multiplier, Im_multiplier
                                    );
                                    
                                    cfma(
                                        Re_factor,          Im_factor,
                                        Re_Y_blk[j_loc][k], Im_Y_blk[j_loc][k],
                                        Re_X_blk[i_loc][k], Im_X_blk[i_loc][k]
                                    );
                                }
                            }
                        }
                    
                    }

                    Re_X_blk.Write( &Re_X[i_chunk * i_blk] );
                    Im_X_blk.Write( &Im_X[i_chunk * i_blk] );
                }
            }
            
            toc(ClassName()+"::Multiply_Blocked_C<"+ToString(i_blk_size)+","+ToString(j_blk_size)+","+ToString(n_waves)+">");
        }
        
    public:
        
        std::string ClassName() const
        {
            return "HelmholtzOperator_SoA<"+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
        
    }; // HelmholtzOperator_SoA
    
} // namespace BAEMM
