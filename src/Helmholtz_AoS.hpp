namespace BAEMM
{
    
    template<typename Real_, typename Int_>
    class Helmholtz_AoS
    {
    public:
        
        using Real    = Real_;
        using Int     = Int_;
        using Complex = std::complex<Real>;
        
        static constexpr Real one   = 1;
        static constexpr Real two   = 2;
        static constexpr Real three = 3;
        
        static constexpr Real half  = one / two;
        static constexpr Real third = one / three;
        
        static constexpr Real pi      = M_PI;
        static constexpr Real two_pi  = two * pi;
        static constexpr Real four_pi = two * two_pi;
        
//        static constexpr Real one_over_two_pi  = one / two_pi;
        static constexpr Real one_over_four_pi = one / four_pi;
        
        
//        using Mesh_T = SimplicialMesh<2,3,Real,Int,Real,ExtReal>;
        
        Helmholtz_AoS() = delete;
        
        template<typename ExtReal,typename ExtInt>
        Helmholtz_AoS(
            ptr<ExtReal> vertex_coords_, ExtInt vertex_count_,
            ptr<ExtInt>  triangles_    , ExtInt simplex_count_
        )
        :   m             ( vertex_count_                     )
        ,   n             ( simplex_count_                    )
        ,   vertex_coords ( vertex_coords_, vertex_count_,  3 )
        ,   triangles     ( triangles_,     simplex_count_, 3 )
        ,   areas         (                 simplex_count_    )
        ,   mid_points    (                 simplex_count_, 3 )
        ,   normals       (                 simplex_count_, 3 )
        {
            Tiny::Vector<3,Real,Int> x;
            Tiny::Vector<3,Real,Int> y;
            Tiny::Vector<3,Real,Int> z;
            
            Tiny::Vector<3,Real,Int> nu;
            
            for( Int i = 0; i < n; ++i )
            {
                x.Read( vertex_coords.data(triangles(i,0)) );
                y.Read( vertex_coords.data(triangles(i,1)) );
                z.Read( vertex_coords.data(triangles(i,2)) );

                mid_points(i,0) = third * ( x[0] + y[0] + z[0] );
                mid_points(i,1) = third * ( x[1] + y[1] + z[1] );
                mid_points(i,2) = third * ( x[2] + y[2] + z[2] );
                
                y -= x;
                z -= x;

                nu[0] = y[1] * z[2] - y[2] * z[1];
                nu[1] = y[2] * z[0] - y[0] * z[2];
                nu[2] = y[0] * z[1] - y[1] * z[0];

                const Real a = nu.Norm();
                areas[i] = a;

                nu /= a;

                nu.Write( normals.data(i) );
            }
        }
        
        ~Helmholtz_AoS() = default;
        
    protected:
        
        const Int m;    // vertex_count
        const Int n;    // triangle_count

        Tensor2<Real,Int> vertex_coords;
        Tensor2<Int ,Int> triangles;
        
        Tensor1<Real,Int> areas;
        Tensor2<Real,Int> mid_points;
        Tensor2<Real,Int> normals;
        
        
    public:
        
        template<Int n_waves, bool ascending>
        void Neumann_to_Dirichlet(
            ptr<Complex> Y,
            mut<Complex> X,
            const Real kappa,
            const Real kappa_step,
            const Int  thread_count
        ) const
        {
            tic(ClassName()+"::Neumann_to_Dirichlet<"+ToString(n_waves)+","+ToString(ascending)+">");
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
                    x.Read( mid_points.data(i) );
                 
                    X_i.SetZero();
                    
                    for( Int j = 0; j < n; ++j )
                    {
                        const Real delta = static_cast<Real>(i==j);
                        
                        y.Read( mid_points.data(j) );
                        
                        z[0] = y[0] - x[0];
                        z[1] = y[1] - x[1];
                        z[2] = y[2] - x[2];
                        
                        const Real r = z.Norm() + delta;
                        
                        const Real r_inv = one_over_four_pi * (one-delta) / r;

                        Complex factor ( std::exp( Complex( 0, kappa * r ) ) * r_inv );
                        
                        const Complex multiplier (
                            COND(
                                ascending,
                                std::exp( Complex( 0, kappa_step * r ) ),
                                Complex(0)
                            )
                        );
                        
                        X_i[0] += factor * Y[n_waves * j + 0];
                        
                        LOOP_UNROLL_FULL
                        for( Int k = 1; k < n_waves; ++k )
                        {
                            if constexpr ( ascending )
                            {
                                factor *= multiplier;
                            }

                            X_i[k] += factor * Y[n_waves * j + k];
                        }
                    }
                    
                    X_i.Write( &X[n_waves*i] );
                }
            }
            
            toc(ClassName()+"::Neumann_to_Dirichlet<"+ToString(n_waves)+","+ToString(ascending)+">");
        }
        
        
        template<
            Int i_blk_size, Int j_blk_size, Int n_waves, bool ascending,
            bool copy_x = true, bool copy_y = true, bool copy_Y = false
        >
        void Neumann_to_Dirichlet_Blocked(
            ptr<std::complex<Real>> Y,
            mut<std::complex<Real>> X,
            const Real kappa,
            const Real kappa_step,
            const Int  thread_count
        ) const
        {
            tic(ClassName()+"::Neumann_to_Dirichlet_Blocked<"+ToString(i_blk_size)+","+ToString(j_blk_size)+","+ToString(n_waves)+","+ToString(ascending)+","+ToString(copy_x)+","+ToString(copy_y)+","+ToString(copy_Y)+">");
                    
            constexpr Int i_chunk = i_blk_size * n_waves;
            constexpr Int j_chunk = j_blk_size * n_waves;

            if( (n/i_blk_size) * i_blk_size != n ) 
            {
                wprint(ClassName()+"::Neumann_to_Dirichlet_Blocked: Loop peeling not applied.");
            }
        
            
            JobPointers<Int> job_ptr(n/i_blk_size, thread_count);
            
            #pragma omp parallel for num_threads( thread_count)
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                Tiny::Matrix<i_blk_size,3,Real,Int> x;
                Tiny::Matrix<j_blk_size,3,Real,Int> y;
                
                Tiny::Vector<3,Real,Int> z;
//                Tiny::Vector<3,Real,Int> nu;
                
                Tiny::Matrix<i_blk_size,n_waves,   Complex,Int> X_blk;
                Tiny::Matrix<j_blk_size,n_waves,   Complex,Int> Y_blk;
                
                Tiny::Matrix<i_blk_size,j_blk_size,Complex,Int> A;
                Tiny::Matrix<i_blk_size,j_blk_size,Complex,Int> B;
                
                const Int i_blk_begin = job_ptr[thread  ];
                const Int i_blk_end   = job_ptr[thread+1];
                
                const Int j_blk_begin = 0;
                const Int j_blk_end   = n/j_blk_size;
                
                for( Int i_blk = i_blk_begin; i_blk < i_blk_end; ++i_blk )
                {
                    const Int i_base = i_blk_size * i_blk;
                    
                    X_blk.SetZero();
                    
                    if constexpr ( copy_x )
                    {
                        x.Read( mid_points.data(i_base) );
                    }
                    
                    for( Int j_blk = j_blk_begin; j_blk < j_blk_end; ++j_blk )
                    {
                        const Int j_base = j_blk_size * j_blk;
                        
                        if constexpr ( copy_y )
                        {
                            y.Read( mid_points.data(j_base) );
                        }
                        
                        LOOP_UNROLL_FULL
                        for( Int i = 0; i < i_blk_size; ++i )
                        {
                            const Int i_global = i_base + i;
                            
                            LOOP_UNROLL_FULL
                            for( Int j = 0; j < j_blk_size; ++j )
                            {
                                const Int j_global = j_base + j;
                                
                                const Real delta_ij = static_cast<Real>(i_global==j_global);
                                
                                if constexpr ( copy_y )
                                {
                                    if constexpr ( copy_x )
                                    {
                                        z[0] = y[j][0] - x[i][0];
                                        z[1] = y[j][1] - x[i][1];
                                        z[2] = y[j][2] - x[i][2];
                                    }
                                    else
                                    {
                                        z[0] = y[j][0] - mid_points[i_global][0];
                                        z[1] = y[j][1] - mid_points[i_global][1];
                                        z[2] = y[j][2] - mid_points[i_global][2];
                                    }
                                }
                                else
                                {
                                    if constexpr ( copy_x )
                                    {
                                        z[0] = mid_points[j_global][0] - x[i][0];
                                        z[1] = mid_points[j_global][1] - x[i][1];
                                        z[2] = mid_points[j_global][2] - x[i][2];
                                    }
                                    else
                                    {
                                        z[0] = mid_points[j_global][0] - mid_points[i_global][0];
                                        z[1] = mid_points[j_global][1] - mid_points[i_global][1];
                                        z[2] = mid_points[j_global][2] - mid_points[i_global][2];
                                    }
                                }
                                
                                const Real r = z.Norm();
                                
                                const Real r_inv = one_over_four_pi * (one-delta_ij)/(r + delta_ij);
                                
                                A[i][j] = std::exp( Complex(0,kappa * r) ) * r_inv;
                                
                                
                                if constexpr ( ascending )
                                {
                                    B[i][j] = std::exp( Complex(0,kappa_step * r) );
                                }

                            } // for( Int j = 0; j < j_blk_size; ++j )
                            
                        } // for( Int i = 0; i < i_blk_size; ++i )
                        
                        
                        if constexpr ( copy_Y )
                        {
                            Y_blk.Read( &Y[j_chunk * j_blk] );
                        }
                        
                        LOOP_UNROLL_FULL
                        for( Int i = 0; i < i_blk_size; ++i )
                        {
                            LOOP_UNROLL_FULL
                            for( Int j = 0; j < j_blk_size; ++j )
                            {
                                if constexpr ( copy_Y )
                                {
                                    X_blk[i][0] += A[i][j] * Y_blk[j][0];
                                }
                                else
                                {
                                    X_blk[i][0] += A[i][j] * Y[n_waves*(j_base+j)+0];
                                }
                            }
                        }
                        
                        LOOP_UNROLL_FULL
                        for( Int k = 1; k < n_waves; ++k )
                        {
                            LOOP_UNROLL_FULL
                            for( Int i = 0; i < i_blk_size; ++i )
                            {
                                LOOP_UNROLL_FULL
                                for( Int j = 0; j < j_blk_size; ++j )
                                {
                                    if constexpr ( ascending )
                                    {
                                        A[i][j] *= B[i][j];
                                    }
                                    
                                    if constexpr ( copy_Y )
                                    {
                                        X_blk[i][k] += A[i][j] * Y_blk[j][k];
                                    }
                                    else
                                    {
                                        X_blk[i][k] += A[i][j] * Y[n_waves*(j_base+j)+k];
                                    }
                                }
                            }
                        }
                        
                    } // for( Int j_blk = j_blk_begin; j_blk < j_blk_end; ++j_blk )
                    
                    X_blk.Write( &X[i_chunk * i_blk] );
                    
                } // for( Int i_blk = i_blk_begin; i_blk < i_blk_end; ++i_blk )
                
            } // for( Int thread = 0; thread < thread_count; ++thread )
            
            toc(ClassName()+"::Neumann_to_Dirichlet_Blocked<"+ToString(i_blk_size)+","+ToString(j_blk_size)+","+ToString(n_waves)+","+ToString(ascending)+","+ToString(copy_x)+","+ToString(copy_y)+","+ToString(copy_Y)+">");
        }
        
        
        template<
            Int i_blk_size, Int j_blk_size, Int n_waves, bool ascending,
            bool copy_x = true, bool copy_y = true, bool copy_Y = false
        >
        void Neumann_to_Dirichlet_Assembled(
            mut<std::complex<Real>> g_A,
            const Real kappa,
            const Real kappa_step,
            const Int  thread_count
        ) const
        {
            tic(ClassName()+"::Neumann_to_Dirichlet_Assembled<"+ToString(i_blk_size)+","+ToString(j_blk_size)+","+ToString(n_waves)+","+ToString(ascending)+","+ToString(copy_x)+","+ToString(copy_y)+","+ToString(copy_Y)+">");
                    
            if( (n/i_blk_size) * i_blk_size != n )
            {
                wprint(ClassName()+"::Neumann_to_Dirichlet_Assembled: Loop peeling not applied.");
            }
        
            
            JobPointers<Int> job_ptr(n/i_blk_size, thread_count);
            
            #pragma omp parallel for num_threads( thread_count)
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                Tiny::Matrix<i_blk_size,3,Real,Int> x;
                Tiny::Matrix<j_blk_size,3,Real,Int> y;
                
                Tiny::Vector<3,Real,Int> z;
//                Tiny::Vector<3,Real,Int> nu;
                
//                Tiny::Matrix<i_blk_size,j_blk_size,Complex,Int> A;
//                Tiny::Matrix<i_blk_size,j_blk_size,Complex,Int> B;
                
                const Int i_blk_begin = job_ptr[thread  ];
                const Int i_blk_end   = job_ptr[thread+1];
                
                const Int j_blk_begin = 0;
                const Int j_blk_end   = n/j_blk_size;
                
                for( Int i_blk = i_blk_begin; i_blk < i_blk_end; ++i_blk )
                {
                    const Int i_base = i_blk_size * i_blk;
                    
                    if constexpr ( copy_x )
                    {
                        x.Read( mid_points.data(i_base) );
                    }
                    
                    for( Int j_blk = j_blk_begin; j_blk < j_blk_end; ++j_blk )
                    {
                        const Int j_base = j_blk_size * j_blk;
                        
                        if constexpr ( copy_y )
                        {
                            y.Read( mid_points.data(j_base) );
                        }
                        
                        LOOP_UNROLL_FULL
                        for( Int i = 0; i < i_blk_size; ++i )
                        {
                            const Int i_global = i_base + i;
                            
                            LOOP_UNROLL_FULL
                            for( Int j = 0; j < j_blk_size; ++j )
                            {
                                const Int j_global = j_base + j;
                                
                                const Real delta_ij = static_cast<Real>(i_global==j_global);
                                
                                if constexpr ( copy_y )
                                {
                                    if constexpr ( copy_x )
                                    {
                                        z[0] = y[j][0] - x[i][0];
                                        z[1] = y[j][1] - x[i][1];
                                        z[2] = y[j][2] - x[i][2];
                                    }
                                    else
                                    {
                                        z[0] = y[j][0] - mid_points[i_global][0];
                                        z[1] = y[j][1] - mid_points[i_global][1];
                                        z[2] = y[j][2] - mid_points[i_global][2];
                                    }
                                }
                                else
                                {
                                    if constexpr ( copy_x )
                                    {
                                        z[0] = mid_points[j_global][0] - x[i][0];
                                        z[1] = mid_points[j_global][1] - x[i][1];
                                        z[2] = mid_points[j_global][2] - x[i][2];
                                    }
                                    else
                                    {
                                        z[0] = mid_points[j_global][0] - mid_points[i_global][0];
                                        z[1] = mid_points[j_global][1] - mid_points[i_global][1];
                                        z[2] = mid_points[j_global][2] - mid_points[i_global][2];
                                    }
                                }
                                
                                const Real r = z.Norm();
                                
                                const Real r_inv = one_over_four_pi * (one-delta_ij)/(r + delta_ij);
                                
                                g_A[n * i_global + j_global] = std::exp( Complex(0,kappa * r) ) * r_inv;
                                
                                
//                                if constexpr ( ascending )
//                                {
//                                    B[i][j] = std::exp( Complex(0,kappa_step * r) );
//                                }

                            } // for( Int j = 0; j < j_blk_size; ++j )
                            
                        } // for( Int i = 0; i < i_blk_size; ++i )
        
                        
                    } // for( Int j_blk = j_blk_begin; j_blk < j_blk_end; ++j_blk )
                                
                } // for( Int i_blk = i_blk_begin; i_blk < i_blk_end; ++i_blk )
                
            } // for( Int thread = 0; thread < thread_count; ++thread )
            
            toc(ClassName()+"::Neumann_to_Dirichlet_Assembled<"+ToString(i_blk_size)+","+ToString(j_blk_size)+","+ToString(n_waves)+","+ToString(ascending)+","+ToString(copy_x)+","+ToString(copy_y)+","+ToString(copy_Y)+">");
        }
        
        template<Int n_waves, bool ascending>
        void Neumann_to_Dirichlet_C(
            ptr<Real> Re_Y, ptr<Real> Im_Y,
            mut<Real> Re_X, mut<Real> Im_X,
            const Real kappa,
            const Real kappa_step,
            const Int  thread_count
        ) const
        {
            tic(ClassName()+"::Neumann_to_Dirichlet_C<"+ToString(n_waves)+","+ToString(ascending)+">");
            
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
                    x.Read( mid_points.data(i) );
                 
                    Re_X_i.SetZero();
                    Im_X_i.SetZero();
                    
                    for( Int j = 0; j < n; ++j )
                    {
                        const Real delta = static_cast<Real>(i==j);
                        
                        y.Read( mid_points.data(j) );
                        
                        z[0] = y[0] - x[0];
                        z[1] = y[1] - x[1];
                        z[2] = y[2] - x[2];
                        
                        const Real r = z.Norm() + delta;
                        
                        const Real r_inv = one_over_four_pi * (one-delta) / r;
                        
                        Real Re_factor = std::cos(kappa * r) * r_inv;
                        Real Im_factor = std::sin(kappa * r) * r_inv;

                        Real Re_multiplier = COND( ascending, std::cos(kappa_step * r), 0 );
                        Real Im_multiplier = COND( ascending, std::sin(kappa_step * r), 0 );
                        
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
                            if constexpr ( ascending )
                            {
                                cmulby(
                                    Re_factor,     Im_factor,
                                    Re_multiplier, Im_multiplier
                                );
                            }
                            
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
            
            toc(ClassName()+"::Neumann_to_Dirichlet_C<"+ToString(n_waves)+","+ToString(ascending)+">");
        }
        
        
        template<
            Int i_blk_size, Int j_blk_size, Int n_waves, bool ascending,
            bool copy_x = true, bool copy_y = true, bool copy_Y = false
        >
        void Neumann_to_Dirichlet_Blocked_C(
            ptr<Real> Re_Y, ptr<Real> Im_Y,
            mut<Real> Re_X, mut<Real> Im_X,
            const Real kappa,
            const Real kappa_step,
            const Int  thread_count
        ) const
        {
            tic(ClassName()+"::Neumann_to_Dirichlet_Blocked_C<"+ToString(i_blk_size)+","+ToString(j_blk_size)+","+ToString(n_waves)+","+ToString(ascending)+","+ToString(copy_x)+","+ToString(copy_y)+","+ToString(copy_Y)+">");
                    
            constexpr Int i_chunk = i_blk_size * n_waves;
            constexpr Int j_chunk = j_blk_size * n_waves;

            if( (n/i_blk_size) * i_blk_size != n )
            {
                wprint(ClassName()+"::Neumann_to_Dirichlet_Blocked_C: Loop peeling not applied.");
            }
            
            JobPointers<Int> job_ptr(n/i_blk_size, thread_count);
            
            #pragma omp parallel for num_threads( thread_count)
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                Tiny::Matrix<i_blk_size,3,Real,Int> x;
                Tiny::Matrix<j_blk_size,3,Real,Int> y;
                
                Tiny::Vector<3,Real,Int> z;
                Tiny::Vector<3,Real,Int> nu;
                
                Tiny::Matrix<i_blk_size,n_waves,   Real,Int> Re_X_blk;
                Tiny::Matrix<i_blk_size,n_waves,   Real,Int> Im_X_blk;
                Tiny::Matrix<j_blk_size,n_waves,   Real,Int> Re_Y_blk;
                Tiny::Matrix<j_blk_size,n_waves,   Real,Int> Im_Y_blk;
                
                Tiny::Matrix<i_blk_size,j_blk_size,Real,Int> Re_A;
                Tiny::Matrix<i_blk_size,j_blk_size,Real,Int> Im_A;
                Tiny::Matrix<i_blk_size,j_blk_size,Real,Int> Re_B;
                Tiny::Matrix<i_blk_size,j_blk_size,Real,Int> Im_B;
                
                const Int i_blk_begin = job_ptr[thread  ];
                const Int i_blk_end   = job_ptr[thread+1];
                
                const Int j_blk_begin = 0;
                const Int j_blk_end   = n/j_blk_size;
                
                for( Int i_blk = i_blk_begin; i_blk < i_blk_end; ++i_blk )
                {
                    const Int i_base = i_blk_size * i_blk;
                    
                    Re_X_blk.SetZero();
                    Im_X_blk.SetZero();
                    
                    if constexpr ( copy_x )
                    {
                        x.Read( mid_points.data(i_base) );
                        
                        nu.Read( normals.data(i_base) );
                    }
                    
                    for( Int j_blk = j_blk_begin; j_blk < j_blk_end; ++j_blk )
                    {
                        const Int j_base = j_blk_size * j_blk;
                        
                        if constexpr ( copy_y )
                        {
                            y.Read( mid_points.data(j_base) );
                        }
                        
                        LOOP_UNROLL_FULL
                        for( Int i = 0; i < i_blk_size; ++i )
                        {
                            const Int i_global = i_base + i;
                            
                            LOOP_UNROLL_FULL
                            for( Int j = 0; j < j_blk_size; ++j )
                            {
                                const Int j_global = j_base + j;
                                
                                const Real delta = static_cast<Real>(i_global==j_global);
                                
                                if constexpr ( copy_y )
                                {
                                    if constexpr ( copy_x )
                                    {
                                        z[0] = y[j][0] - x[i][0];
                                        z[1] = y[j][1] - x[i][1];
                                        z[2] = y[j][2] - x[i][2];
                                    }
                                    else
                                    {
                                        z[0] = y[j][0] - mid_points[i_global][0];
                                        z[1] = y[j][1] - mid_points[i_global][1];
                                        z[2] = y[j][2] - mid_points[i_global][2];
                                    }
                                }
                                else
                                {
                                    if constexpr ( copy_x )
                                    {
                                        z[0] = mid_points[j_global][0] - x[i][0];
                                        z[1] = mid_points[j_global][1] - x[i][1];
                                        z[2] = mid_points[j_global][2] - x[i][2];
                                    }
                                    else
                                    {
                                        z[0] = mid_points[j_global][0] - mid_points[i_global][0];
                                        z[1] = mid_points[j_global][1] - mid_points[i_global][1];
                                        z[2] = mid_points[j_global][2] - mid_points[i_global][2];
                                    }
                                }
                                
                                const Real r2   = z[0] * z [0] + z[1] * z [1] + z[2] * z [2];
                                
                                const Real z_nu = COND(
                                    copy_x,
                                    z[0] * nu[0] + z[1] * nu[1] + z[2] * nu[2],
                                    z[0]*normals[3*i+0] + z[1]*normals[3*i+1] + z[2]*normals[3*i+2]
                                );
                                
//                                const Real z_nu = z[0] * nu[0] + z[1] * nu[1] + z[2] * nu[2];
                                
                                const Real r   = std::sqrt(r2);
                                
                                const Real r_inv = one_over_four_pi * (one-delta) / (r + delta);
                                
//                                const Real r3  = r * r2;
                                
                                // A[i][j] = one_over_two_pi * std::exp(I * kappa * r) / r3 * ( r2 + eta * (I - kappa * r) * z_nu);
                                
                                Re_A[i][j] = std::cos(kappa * r) * r_inv;
                                Im_A[i][j] = std::sin(kappa * r) * r_inv;
                                
                                Re_B[i][j] = std::cos(kappa_step * r);
                                Im_B[i][j] = std::sin(kappa_step * r);

                            } // for( Int j = 0; j < j_blk_size; ++j )
                            
                        } // for( Int u = 0; u < i_blk_size; ++u )
                        
                        
                        if constexpr ( copy_Y )
                        {
                            Re_Y_blk.Read( &Re_Y[j_chunk * j_blk] );
                            Im_Y_blk.Read( &Im_Y[j_chunk * j_blk] );
                        }
                        
                        LOOP_UNROLL_FULL
                        for( Int i = 0; i < i_blk_size; ++i )
                        {
                            LOOP_UNROLL_FULL
                            for( Int j = 0; j < j_blk_size; ++j )
                            {
                                if constexpr ( copy_Y )
                                {
                                    cfma(
                                        Re_A    [i][j], Im_A    [i][j],
                                        Re_Y_blk[j][0], Im_Y_blk[j][0],
                                        Re_X_blk[i][0], Im_X_blk[i][0]
                                    );
                                }
                                else
                                {
                                    cfma(
                                        Re_A    [i][j],               Im_A    [i][j],
                                        Re_Y [n_waves*(j_base+j)+0],  Im_Y[n_waves*(j_base+j)+0],
                                        Re_X_blk[i][0],               Im_X_blk[i][0]
                                    );
                                }
                            }
                        }
                        
                        LOOP_UNROLL_FULL
                        for( Int k = 1; k < n_waves; ++k )
                        {
                            LOOP_UNROLL_FULL
                            for( Int i = 0; i < i_blk_size; ++i )
                            {
                                LOOP_UNROLL_FULL
                                for( Int j = 0; j < j_blk_size; ++j )
                                {
                                    if constexpr ( ascending )
                                    {
                                        cmulby( Re_A[i][j], Im_A[i][j], Re_B[i][j], Im_B[i][j] );
                                    }
                                    
                                    if constexpr ( copy_Y )
                                    {
                                        cfma(
                                            Re_A    [i][j], Im_A    [i][j],
                                            Re_Y_blk[j][k], Im_Y_blk[j][k],
                                            Re_X_blk[i][k], Im_X_blk[i][k]
                                        );
                                    }
                                    else
                                    {
                                        cfma(
                                            Re_A    [i][j],               Im_A    [i][j],
                                            Re_Y [n_waves*(j_base+j)+k],  Im_Y[n_waves*(j_base+j)+k],
                                            Re_X_blk[i][k],               Im_X_blk[i][k]
                                        );
                                    }
                                }
                            }
                        }
                        
                    } // for( Int j_blk = j_blk_begin; j_blk < j_blk_end; ++j_blk )
                    
                    Re_X_blk.Write( &Re_X[i_chunk * i_blk] );
                    Im_X_blk.Write( &Im_X[i_chunk * i_blk] );
                    
                } // for( Int i_blk = i_blk_begin; i_blk < i_blk_end; ++i_blk )
                
            } // for( Int thread = 0; thread < thread_count; ++thread )
            
            toc(ClassName()+"::Neumann_to_Dirichlet_Blocked_C<"+ToString(i_blk_size)+","+ToString(j_blk_size)+","+ToString(n_waves)+","+ToString(ascending)+","+ToString(copy_x)+","+ToString(copy_y)+","+ToString(copy_Y)+">");
        }
        
    public:
        
        std::string ClassName() const
        {
            return "Helmholtz_AoS<"+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
        
    }; // Helmholtz_AoS
    
} // namespace BAEMM

