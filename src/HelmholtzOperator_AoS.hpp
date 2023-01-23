namespace BAEMM
{
    
    template<class Mesh_T>
    class HelmholtzOperator_AoS
    {
    public:
        
        using Real       = typename Mesh_T::Real;
        using Int        = typename Mesh_T::Int;
        using SReal      = typename Mesh_T::SReal;
        using Complex    = std::complex<SReal>;
        
        using ExtReal    = typename Mesh_T::ExtReal;
//        using ExtComplex = std::complex<ExtReal>;
        
        static constexpr SReal one   = 1;
        static constexpr SReal two   = 2;
        static constexpr SReal three = 3;
        
        static constexpr SReal half  = one / two;
        static constexpr SReal third = one / three;
        
        
//        using Mesh_T = SimplicialMesh<2,3,Real,Int,Real,ExtReal>;
        
        HelmholtzOperator_AoS() = delete;
        
        explicit HelmholtzOperator_AoS( Mesh_T & M_ )
        :   M             ( M_ )
        ,   n             ( M.SimplexCount() )
        ,   vertex_coords ( M.VertexCoordinates().data(), M.VertexCount(),  3   )
        ,   triangles     ( M.Simplices().data(),         M.SimplexCount(), 3   )
        ,   mid_points    ( n, 3 )
        ,   areas         ( n    )
        ,   normals       ( n, 3 )
        {
            Tiny::Vector<3,SReal,Int> x;
            Tiny::Vector<3,SReal,Int> y;
            Tiny::Vector<3,SReal,Int> z;
            
            Tiny::Vector<3,SReal,Int> nu;
            
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

                const SReal a = nu.Norm();
                areas[i] = a;

                nu /= a;

                nu.Write( normals.data(i) );
            }
        }
        
        ~HelmholtzOperator_AoS() = default;
        
    protected:
        
        Mesh_T & M;
        
        const Int n;

        Tensor2<SReal,Int> vertex_coords;
        Tensor2<Int  ,Int> triangles;
        
        Tensor1<SReal,Int> areas;
        Tensor2<SReal,Int> mid_points;
        Tensor2<SReal,Int> normals;
        
        
    public:
        
        template< Int n_waves>
        void Multiply(
            ptr<Complex> Y,
            mut<Complex> X,
            const SReal kappa,
            const SReal kappa_step,
            const Int  thread_count
        ) const
        {
            tic(ClassName()+"::Multiply<"+ToString(n_waves)+">");
            JobPointers<Int> job_ptr(n, thread_count);
            
            #pragma omp parallel for num_threads( thread_count)
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                Tiny::Vector<3,SReal,Int> x;
                Tiny::Vector<3,SReal,Int> y;
                Tiny::Vector<3,SReal,Int> z;
//                Tiny::Vector<3,SReal,Int> nu;
                
                Tiny::Vector<n_waves,Complex,Int> X_i;
                
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    x.Read( mid_points.data(i) );
                 
                    X_i.SetZero();
                    
                    for( Int j = 0; j < n; ++j )
                    {
                        const SReal delta = static_cast<SReal>(i==j);
                        
                        y.Read( mid_points.data(j) );
                        
                        z[0] = y[0] - x[0];
                        z[1] = y[1] - x[1];
                        z[2] = y[2] - x[2];
                        
                        const SReal r = z.Norm() + delta;
                        
                        const SReal r_inv = (one-delta) / r;

                              Complex factor     ( std::exp( Complex(0,kappa      * r) ) * r_inv );
                        const Complex multiplier ( std::exp( Complex(0,kappa_step * r) )         );
                        
                        
//                        Complex factor (
//                            std::cos(kappa * r) * r_inv,
//                            std::sin(kappa * r) * r_inv
//                        );
//
//                        const Complex multiplier (
//                            std::cos(kappa_step * r),
//                            std::sin(kappa_step * r)
//                        );
                        
                        
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
        
        
        template<
            Int i_blk_size, Int j_blk_size, Int n_waves,
            bool copy_x = true, bool copy_y = true, bool copy_Y = false
        >
        void Multiply_Blocked(
            ptr<std::complex<SReal>> Y,
            mut<std::complex<SReal>> X,
            const SReal kappa,
            const SReal kappa_step,
            const Int  thread_count
        ) const
        {
            tic(ClassName()+"::Multiply_Blocked<"+ToString(i_blk_size)+","+ToString(j_blk_size)+","+ToString(n_waves)+","+ToString(copy_x)+","+ToString(copy_y)+","+ToString(copy_Y)+">");
                    
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
                Tiny::Matrix<i_blk_size,3,SReal,Int> x;
                Tiny::Matrix<j_blk_size,3,SReal,Int> y;
                
                Tiny::Vector<3,SReal,Int> z;
//                Tiny::Vector<3,SReal,Int> nu;
                
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
                                
                                const SReal delta = static_cast<SReal>(i_global==j_global);
                                
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
                                
                                const SReal r = z.Norm() + delta;
                                
                                const SReal r_inv = (one-delta) / r;
                                
                                A[i][j] = std::exp( Complex(0,kappa      * r) ) * r_inv;
                                B[i][j] = std::exp( Complex(0,kappa_step * r) );
                                
//                                A[i][j].real( std::cos(kappa * r) * r_inv );
//                                A[i][j].imag( std::sin(kappa * r) * r_inv );
//
//                                B[i][j].real( std::cos(kappa_step * r)    );
//                                B[i][j].imag( std::sin(kappa_step * r)    );

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
                                    A[i][j] *= B[i][j];
                                    
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
            
            toc(ClassName()+"::Multiply_Blocked<"+ToString(i_blk_size)+","+ToString(j_blk_size)+","+ToString(n_waves)+","+ToString(copy_x)+","+ToString(copy_y)+","+ToString(copy_Y)+">");
        }
        
        template< Int n_waves>
        void Multiply_C(
            ptr<SReal> Re_Y, ptr<SReal> Im_Y,
            mut<SReal> Re_X, mut<SReal> Im_X,
            const SReal kappa,
            const SReal kappa_step,
            const Int  thread_count
        ) const
        {
            tic(ClassName()+"::Multiply_C<"+ToString(n_waves)+">");
            
            JobPointers<Int> job_ptr(n, thread_count);
            
            #pragma omp parallel for num_threads( thread_count)
            for( Int thread = 0; thread < thread_count; ++thread )
            {
                Tiny::Vector<3,SReal,Int> x;
                Tiny::Vector<3,SReal,Int> y;
                Tiny::Vector<3,SReal,Int> z;
//                Tiny::Vector<3,SReal,Int> nu;
                
                Tiny::Vector<n_waves,SReal,Int> Re_X_i;
                Tiny::Vector<n_waves,SReal,Int> Im_X_i;
                
                const Int i_begin = job_ptr[thread  ];
                const Int i_end   = job_ptr[thread+1];
                
                for( Int i = i_begin; i < i_end; ++i )
                {
                    x.Read( mid_points.data(i) );
                 
                    Re_X_i.SetZero();
                    Im_X_i.SetZero();
                    
                    for( Int j = 0; j < n; ++j )
                    {
                        const SReal delta = static_cast<SReal>(i==j);
                        
                        y.Read( mid_points.data(j) );
                        
                        z[0] = y[0] - x[0];
                        z[1] = y[1] - x[1];
                        z[2] = y[2] - x[2];
                        
                        const SReal r = z.Norm() + delta;
                        
                        const SReal r_inv = (one-delta) / r;
                        
                        SReal Re_factor = std::cos(kappa * r) * r_inv;
                        SReal Im_factor = std::sin(kappa * r) * r_inv;

                        SReal Re_multiplier = std::cos(kappa_step * r);
                        SReal Im_multiplier = std::sin(kappa_step * r);
                        
                        ptr<SReal> Re_Y_j = &Re_Y[n_waves * j];
                        ptr<SReal> Im_Y_j = &Im_Y[n_waves * j];
                        
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
        
        
        template<
            Int i_blk_size, Int j_blk_size, Int n_waves,
            bool copy_x = true, bool copy_y = true, bool copy_Y = false
        >
        void Multiply_Blocked_C(
            ptr<SReal> Re_Y, ptr<SReal> Im_Y,
            mut<SReal> Re_X, mut<SReal> Im_X,
            const SReal kappa,
            const SReal kappa_step,
            const Int  thread_count
        ) const
        {
            tic(ClassName()+"::Multiply_Blocked_C<"+ToString(i_blk_size)+","+ToString(j_blk_size)+","+ToString(n_waves)+","+ToString(copy_x)+","+ToString(copy_y)+","+ToString(copy_Y)+">");
                    
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
                Tiny::Matrix<i_blk_size,3,SReal,Int> x;
                Tiny::Matrix<j_blk_size,3,SReal,Int> y;
                
                Tiny::Vector<3,SReal,Int> z;
//                Tiny::Vector<3,SReal,Int> nu;
                
                Tiny::Matrix<i_blk_size,n_waves,   SReal,Int> Re_X_blk;
                Tiny::Matrix<i_blk_size,n_waves,   SReal,Int> Im_X_blk;
                Tiny::Matrix<j_blk_size,n_waves,   SReal,Int> Re_Y_blk;
                Tiny::Matrix<j_blk_size,n_waves,   SReal,Int> Im_Y_blk;
                
                Tiny::Matrix<i_blk_size,j_blk_size,SReal,Int> Re_A;
                Tiny::Matrix<i_blk_size,j_blk_size,SReal,Int> Im_A;
                Tiny::Matrix<i_blk_size,j_blk_size,SReal,Int> Re_B;
                Tiny::Matrix<i_blk_size,j_blk_size,SReal,Int> Im_B;
                
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
                                
                                const SReal delta = static_cast<SReal>(i_global==j_global);
                                
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
                                
                                const SReal r = z.Norm() + delta;
                                
                                const SReal r_inv = (one-delta) / r;
                                
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
                        for( Int k = 0; k < n_waves; ++k )
                        {
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
                                    
                                    cmulby( Re_A[i][j], Im_A[i][j], Re_B[i][j], Im_B[i][j] );
                                }
                            }
                        }
                        
                    } // for( Int j_blk = j_blk_begin; j_blk < j_blk_end; ++j_blk )
                    
                    Re_X_blk.Write( &Re_X[i_chunk * i_blk] );
                    Im_X_blk.Write( &Im_X[i_chunk * i_blk] );
                    
                } // for( Int i_blk = i_blk_begin; i_blk < i_blk_end; ++i_blk )
                
            } // for( Int thread = 0; thread < thread_count; ++thread )
            
            toc(ClassName()+"::Multiply_Blocked_C<"+ToString(i_blk_size)+","+ToString(j_blk_size)+","+ToString(n_waves)+","+ToString(copy_x)+","+ToString(copy_y)+","+ToString(copy_Y)+">");
        }
        
    public:
        
        std::string ClassName() const
        {
            return "HelmholtzOperator_AoS<"+TypeName<Real>::Get()+","+TypeName<Int>::Get()+","+TypeName<SReal>::Get()+","+TypeName<ExtReal>::Get()+">";
        }
        
    }; // HelmholtzOperator_AoS
    
} // namespace BAEMM

