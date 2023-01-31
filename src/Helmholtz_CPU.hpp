namespace BAEMM
{
    template<typename Real_, typename Int_>
    class Helmholtz_CPU
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
        
        static constexpr Real one_over_four_pi = one / four_pi;
        
        Helmholtz_CPU() = delete;
        
        
            
            template<typename ExtReal,typename ExtInt>
            Helmholtz_CPU(
                ptr<ExtReal> vertex_coords_, ExtInt vertex_count_,
                ptr<ExtInt>  triangles_    , ExtInt simplex_count_,
                Int OMP_thread_count_
            )
            :   OMP_thread_count ( OMP_thread_count_                 )
            ,   vertex_count     ( vertex_count_                     )
            ,   simplex_count    ( simplex_count_                    )
            ,   vertex_coords    ( vertex_coords_, vertex_count_,  3 )
            ,   triangles        ( triangles_,     simplex_count_, 3 )
            ,   areas            (                 simplex_count_    )
            ,   mid_points       (                 simplex_count_, 3 )
            ,   normals          (                 simplex_count_, 3 )
            {
                tic(ClassName());

                
                #pragma omp parallel for num_threads( OMP_thread_count ) schedule( static )
                for( Int i = 0; i < simplex_count; ++i )
                {
                    Tiny::Vector<3,Real,Int> x;
                    Tiny::Vector<3,Real,Int> y;
                    Tiny::Vector<3,Real,Int> z;
                    
                    Tiny::Vector<3,Real,Int> nu;
                    
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
                
                toc(ClassName());
            }
            
            ~Helmholtz_CPU() = default;
        
    protected:
        
        const Int OMP_thread_count = 1;
        
        const Int vertex_count;
        const Int simplex_count;

        Tensor2<Real,Int> vertex_coords;
        Tensor2<Int ,Int> triangles;
        
        Tensor1<Real,Int> areas;
        Tensor2<Real,Int> mid_points;
        Tensor2<Real,Int> normals;
        
        Real    coeff_over_four_pi   [4][2] = {{}};
        
        Complex coeff_C_over_four_pi [4]    = {{}};
        
    public:
        
        void LoadCoefficients( const std::array<Complex,3> & coeff )
        {
            // We have to process the coefficients anyways.
            // Hence we can already multiply by one_over_four_pi so that the kernels don't have to do that each time. (The performance gain is not measureable, though.)
            
            SetSingleLayerCoefficient(coeff[0]);
            SetDoubleLayerCoefficient(coeff[1]);
            SetAdjDblLayerCoefficient(coeff[2]);
        }
        
        Complex GetSingleLayerCoefficient() const
        {
            return coeff_C_over_four_pi[0] * four_pi;
        }
        
        
        void SetSingleLayerCoefficient( const Complex & z )
        {
            coeff_C_over_four_pi[0] = z * one_over_four_pi;
            
            coeff_over_four_pi[0][0] = real(coeff_C_over_four_pi[0]);
            coeff_over_four_pi[0][1] = imag(coeff_C_over_four_pi[0]);
        }
        
        Complex GetDoubleLayerCoefficient() const
        {
            return coeff_C_over_four_pi[1] * four_pi;
        }
        
        void SetDoubleLayerCoefficient( const Complex & z )
        {
            coeff_C_over_four_pi[1] = z * one_over_four_pi;
            
            coeff_over_four_pi[1][0] = real(coeff_C_over_four_pi[1]);
            coeff_over_four_pi[1][1] = imag(coeff_C_over_four_pi[1]);
        }
        
        Complex GetAdjDblLayerCoefficient() const
        {
            return coeff_C_over_four_pi[2] * four_pi;
        }
        
        void SetAdjDblLayerCoefficient( const Complex & z )
        {
            coeff_C_over_four_pi[2] = z * one_over_four_pi;
            
            coeff_over_four_pi[2][0] = real(coeff_C_over_four_pi[2]);
            coeff_over_four_pi[2][1] = imag(coeff_C_over_four_pi[2]);
        }
        
    public:
        
        template<
            Int i_blk_size, Int j_blk_size, Int wave_count
        >
        void BoundaryOperatorKernel_C(
            ptr<Complex> B,
            mut<Complex> C,
            const Real kappa,
            const std::array <Complex,3> & coeff
        )
        {
            tic(ClassName()+"::BoundaryOperatorKernel_C<"+ToString(i_blk_size)+","+ToString(j_blk_size)+","+ToString(wave_count)+">");
            
            constexpr Int i_chunk = i_blk_size * wave_count;

            if( (simplex_count/i_blk_size) * i_blk_size != simplex_count )
            {
                wprint(ClassName()+"::BoundaryOperatorKernel_C: Loop peeling not applied.");
            }
            
            LoadCoefficients(coeff);
            JobPointers<Int> job_ptr(simplex_count/i_blk_size, OMP_thread_count);
            
            #pragma omp parallel for num_threads( OMP_thread_count)
            for( Int thread = 0; thread < OMP_thread_count; ++thread )
            {
                Tiny::Matrix<i_blk_size,3,Real,Int> x;
                Tiny::Matrix<i_blk_size,3,Real,Int> nu;
                Tiny::Matrix<j_blk_size,3,Real,Int> y;
                Tiny::Matrix<j_blk_size,3,Real,Int> mu;
                
                Tiny::Vector<3,Real,Int> z;
                
                Tiny::Matrix<i_blk_size,wave_count, Complex,Int> C_blk;
                
                Tiny::Matrix<i_blk_size,j_blk_size,Complex,Int> A;
                
                const Int i_blk_begin = job_ptr[thread  ];
                const Int i_blk_end   = job_ptr[thread+1];
                
                const Int j_blk_begin = 0;
                const Int j_blk_end   = simplex_count/j_blk_size;
                
                for( Int i_blk = i_blk_begin; i_blk < i_blk_end; ++i_blk )
                {
                    const Int i_base = i_blk_size * i_blk;
                    
                    C_blk.SetZero();
                    
                    x.Read( mid_points.data(i_base) );
                    nu.Read( normals.data(i_base) );
                    
                    for( Int j_blk = j_blk_begin; j_blk < j_blk_end; ++j_blk )
                    {
                        const Int j_base = j_blk_size * j_blk;
                        
                        y.Read( mid_points.data(j_base) );
                        mu.Read( normals.data(j_base) );
                        
                        LOOP_UNROLL_FULL
                        for( Int i = 0; i < i_blk_size; ++i )
                        {
                            const Int i_global = i_base + i;
                            
                            LOOP_UNROLL_FULL
                            for( Int j = 0; j < j_blk_size; ++j )
                            {
                                const Int j_global = j_base + j;
                                
                                const Real delta_ij = static_cast<Real>(i_global==j_global);
                                
                                z[0] = y[j][0] - x[i][0];
                                z[1] = y[j][1] - x[i][1];
                                z[2] = y[j][2] - x[i][2];
                                
                                const Real r = z.Norm();
                                
                                const Real r_inv = (one-delta_ij)/(r + delta_ij);
                                
                                const Real dot_z_nu = z[0]*nu[i][0] + z[1]*nu[i][1] + z[2]*nu[i][2];
                                const Real dot_z_mu = z[0]*mu[j][0] + z[1]*mu[j][1] + z[2]*mu[j][2];
                                
                                
                                const Real r3_inv = r_inv * r_inv * r_inv;
                                
                                const Complex I_kappa_r ( 0, kappa * r );
                                
                                const Complex exp_I_kappa_r ( std::exp(I_kappa_r) );
                                
                                const Complex factor ( (I_kappa_r - one) * r3_inv );
                                
                                A[i][j] = exp_I_kappa_r * (
                                    coeff_C_over_four_pi[0] * r_inv
                                    +
                                    factor * (
                                        coeff_C_over_four_pi[1] * dot_z_mu
                                        -
                                        coeff_C_over_four_pi[2] * dot_z_nu
                                    )
                                );

                            } // for( Int j = 0; j < j_blk_size; ++j )
                            
                        } // for( Int i = 0; i < i_blk_size; ++i )
                    
                        
                        LOOP_UNROLL_FULL
                        for( Int k = 0; k < wave_count; ++k )
                        {
                            LOOP_UNROLL_FULL
                            for( Int i = 0; i < i_blk_size; ++i )
                            {
                                LOOP_UNROLL_FULL
                                for( Int j = 0; j < j_blk_size; ++j )
                                {
                                    C_blk[i][k] += A[i][j] * B[wave_count*(j_base+j)+k];
                                }
                            }
                        }
                        
                    } // for( Int j_blk = j_blk_begin; j_blk < j_blk_end; ++j_blk )
                    
                    C_blk.Write( &C[i_chunk * i_blk] );
                    
                } // for( Int i_blk = i_blk_begin; i_blk < i_blk_end; ++i_blk )
    
            } // for( Int thread = 0; thread < OMP_thread_count; ++thread )
            
            toc(ClassName()+"::BoundaryOperatorKernel_C<"+ToString(i_blk_size)+","+ToString(j_blk_size)+","+ToString(wave_count)+">");
        }
        
    public:
        
        std::string ClassName() const
        {
            return "Helmholtz_CPU<"+TypeName<Real>::Get()+","+TypeName<Int>::Get()+">";
        }
        
    }; // Helmholtz_CPU
    
} // namespace BAEMM

