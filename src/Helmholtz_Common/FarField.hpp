public:

    template<typename I_ext, typename R_ext, typename C_ext,I_ext solver_count>
    void FarField(const R_ext* kappa, const I_ext& wave_chunk_count_, 
                    R_ext* inc_directions,  const I_ext& wave_chunk_size_, C_ext* C_out, 
                    R_ext cg_tol, R_ext gmres_tol)
    {
        // Implement the bdry to Farfield map. wave ist the std incident wave defined pointwise by exp(i*kappa*<x,d>). A = (1/2) * I - i * kappa * SL + DL
        // phi = A\wave is the bdry potential which will be mapped onto the far field
        const C_ext One  = static_cast<C_ext>(Complex(1.0f,0.0f));
        const C_ext Zero = static_cast<C_ext>(Complex(0.0f,0.0f));

        const Int     n          = VertexCount();
        const I_ext  wave_count_ = wave_chunk_count_ * wave_chunk_size_;       

        C_ext*  inc_coeff       = (C_ext*)malloc(wave_chunk_count_ * 4 * sizeof(Complex));
        C_ext*  coeff           = (C_ext*)malloc(wave_chunk_count_ * 4 * sizeof(Complex));
        C_ext*  wave            = (C_ext*)malloc(wave_count_ * n * sizeof(Complex));     //weak representation of the incident wave
        C_ext*  phi             = (C_ext*)malloc(wave_count_ * n * sizeof(Complex));
        
        ConjugateGradient<solver_count,C_ext,size_t> cg(n,100,8);
        GMRES<solver_count,C_ext,size_t,Side::Left> gmres(n,30,8);

        // create weak representation of the negative incident wave
        for(int i = 0 ; i < wave_chunk_count_ ; i++)
        {
            inc_coeff[4 * i + 0] = Zero;
            inc_coeff[4 * i + 1] = -One;
            inc_coeff[4 * i + 2] = Zero;
            inc_coeff[4 * i + 3] = Zero;
        }

        CreateIncidentWave_PL(One, inc_directions, wave_chunk_size_,
                            Zero, wave, wave_count_,
                            kappa, inc_coeff, wave_count_, wave_chunk_size_
                            );

        BoundaryPotential<I_ext,R_ext,C_ext,solver_count>( kappa, coeff, wave, phi, wave_chunk_count_, cg_tol, gmres_tol );      

        ApplyFarFieldOperators_PL( One, phi, wave_count_,
                            Zero, C_out, wave_count_,
                            kappa,coeff, wave_count_, wave_chunk_size_
                            );

        free(inc_coeff);
        free(coeff);
        free(phi);
        free(wave);
    }


    template<typename I_ext, typename R_ext, typename C_ext,I_ext solver_count>
    void Derivative_FF(const R_ext* kappa, const I_ext& wave_chunk_count_, 
                    const R_ext* inc_directions,  const I_ext& wave_chunk_size_, 
                    const R_ext* h, C_ext* C_out, 
                    R_ext cg_tol, R_ext gmres_tol)
    {
        // Implement the action of the derivative of the bdry to Farfield map. 
        // inc_wave is a linear combination of the std incident wave defined pointwise by exp(i*kappa*<x,d>) and its normal derivative
        // A := (1/2) * I - i * kappa * SL + DL' for the calculation of du/dn
        // B := (1/2) * I - i * kappa * SL + DL  for the calculation of the Farfield
        // dudn = A\incident_wave is the normal derivative of the solution with inc wave wave
        // phi is the bdry potential for the incident wave dudn *(<h , n>), the solution is the far field to this
        // Formulas follow from Thortens book

        const C_ext One  = static_cast<C_ext>(Complex(1.0f,0.0f));
        const C_ext I    = static_cast<C_ext>(Complex(0.0f,1.0f));
        const C_ext Zero = static_cast<C_ext>(Complex(0.0f,0.0f));

        const Int    n           = VertexCount();
        const I_ext  wave_count_ = wave_chunk_count_ * wave_chunk_size_;
        

        C_ext*  inc_coeff       = (C_ext*)malloc(wave_chunk_count_ * 4 * sizeof(Complex));
        C_ext*  coeff           = (C_ext*)malloc(wave_chunk_count_ * 4 * sizeof(Complex));
        C_ext*  incident_wave   = (C_ext*)malloc(wave_count_ * n * sizeof(Complex));
        C_ext*  wave            = (C_ext*)malloc(wave_count_ * n * sizeof(Complex));     //weak representation of the incident wave
        C_ext*  du_dn           = (C_ext*)malloc(wave_count_ * n * sizeof(Complex));
        C_ext*  du_dn_weak      = (C_ext*)malloc(wave_count_ * n * sizeof(Complex));
        R_ext*  h_n             = (R_ext*)malloc(n * sizeof(Real));
        C_ext*  phi             = (C_ext*)malloc(wave_count_ * n * sizeof(Complex));

        // create weak representation of the negative incident wave
        for(int i = 0 ; i < wave_chunk_count_ ; i++)
        {
            inc_coeff[4 * i + 0] = Zero;
            inc_coeff[4 * i + 1] = -I;
            inc_coeff[4 * i + 2] = One;
            inc_coeff[4 * i + 3] = Zero;
        }

        CreateIncidentWave_PL( One, inc_directions, wave_chunk_size_,
                            Zero, incident_wave, wave_count_,
                            kappa, inc_coeff, wave_count_, wave_chunk_size_
                            );

        DirichletToNeumann<I_ext,R_ext,C_ext,solver_count>( kappa, incident_wave, du_dn, wave_chunk_count_, cg_tol, gmres_tol ); 

        DotWithNormals_PL( h, h_n, cg_tol );

        Int i,j;
        #pragma omp parallel for num_threads( OMP_thread_count ) schedule( static ) private(i) private(j)
        for( i = 0; i < n; ++i )
        {
            LOOP_UNROLL_FULL
            for( j = 0; j < wave_count_; ++j )
            {
                du_dn[i * wave_count_ + j] *= -h_n[i];
            }
        }
        
        // apply mass to du_dn to get weak representation
        for( Int chunk = 0; chunk < wave_chunk_count - 1; ++chunk )
        {
            Mass.Dot(
                One, &du_dn [wave_chunk_size * chunk], wave_count,
                Zero,  &du_dn_weak[wave_chunk_size * chunk], wave_count,
                wave_chunk_size
            );
        }
        {
            const Int chunk = wave_chunk_count - 1;
            Mass.Dot(
                One, &du_dn [wave_chunk_size * chunk], wave_count,
                Zero,  &du_dn_weak[wave_chunk_size * chunk], wave_count,
                wave_count - wave_chunk_size*chunk
            );
        }

        BoundaryPotential<I_ext,R_ext,C_ext,solver_count>( kappa, coeff, du_dn_weak, phi, wave_chunk_count_, cg_tol, gmres_tol);

        ApplyFarFieldOperators_PL( One, phi, wave_count_,
                            Zero, C_out, wave_count_,
                            kappa,coeff, wave_count_, wave_chunk_size_
                            );

        free(inc_coeff);
        free(coeff);
        free(phi);
        free(wave);
        free(incident_wave);
        free(du_dn);
        free(h_n);
    }


    template<typename I_ext, typename R_ext, typename C_ext,I_ext solver_count>
    void AdjointDerivative_FF(const R_ext* kappa, const I_ext& wave_chunk_count_, 
                    const R_ext* inc_directions,  const I_ext& wave_chunk_size_, 
                    const C_ext* g, R_ext* C_out, 
                    R_ext cg_tol, R_ext gmres_tol)
    {
        // Implement the action of the adjoint to the derivative of the bdry to Farfield map. 
        // incident_wave is a linear combination of the std incident wave defined pointwise by exp(i*kappa*<x,d>) and its normal derivative
        // herglotz_wave is a linear combination the herglotz wave with kernel g and its normal derivative
        // A := (1/2) * I - i * kappa * SL + DL'
        // dudn = A\incident_wave is the normal derivative of the solution with inc wave wave
        // anh phi_h = A\herglotz_wave is the normal derivative of the solution with inc wave herglotz_wave

        const C_ext One  = static_cast<C_ext>(Complex(1.0f,0.0f));
        const C_ext I    = static_cast<C_ext>(Complex(0.0f,1.0f));
        const C_ext Zero = static_cast<C_ext>(Complex(0.0f,0.0f));

        const Int    n                      = VertexCount();
        const I_ext  wave_count_            = wave_chunk_count_ * wave_chunk_size_;

        C_ext*  inc_coeff       = (C_ext*)malloc(wave_chunk_count_ * 4 * sizeof(Complex));
        C_ext*  incident_wave   = (C_ext*)malloc(wave_count_ * n * sizeof(Complex));     //weak representation of the incident wave
        C_ext*  herglotz_wave   = (C_ext*)malloc(wave_count_ * n * sizeof(Complex));     //weak representation of the herglotz wave
        C_ext*  du_dn           = (C_ext*)malloc(wave_count_ * n * sizeof(Complex));
        C_ext*  dv_dn           = (C_ext*)malloc(wave_count_ * n * sizeof(Complex));
        C_ext*  wave_product    = (C_ext*)malloc(n * sizeof(Complex));

        // create weak representation of the negative incident wave
        for(int i = 0 ; i < wave_chunk_count_ ; i++)
        {
            inc_coeff[4 * i + 0] = Zero;
            inc_coeff[4 * i + 1] = -I;
            inc_coeff[4 * i + 2] = One;
            inc_coeff[4 * i + 3] = Zero;
        }

        CreateIncidentWave_PL(One, inc_directions, wave_chunk_size_,
                            Zero, incident_wave, wave_count_,
                            kappa, inc_coeff, wave_count_, wave_chunk_size_
                            );

        CreateHerglotzWave_PL(One, g, wave_count,
                            Zero, herglotz_wave, wave_count_,
                            kappa, inc_coeff, wave_count_, wave_chunk_size_
                            );


        // solve for the normal derivatives of the near field solutions
        DirichletToNeumann<I_ext,R_ext,C_ext,solver_count>( kappa, incident_wave, du_dn, wave_chunk_count_, cg_tol, gmres_tol );
        DirichletToNeumann<I_ext,R_ext,C_ext,solver_count>( kappa, herglotz_wave, dv_dn, wave_chunk_count_, cg_tol, gmres_tol );

        // calculate du_dn .* dv_dn and sum over the leading dimension
        HadamardProduct( du_dn, dv_dn, wave_product, n, wave_count_, true);

        // calculate (-1/wave_count)*Re(du_dn .* dv_dn).*normals
        MultiplyWithNormals_PL(wave_product,C_out,-( 1 /(R_ext)wave_count_ ), cg_tol);

        free(inc_coeff);
        free(du_dn);
        free(dv_dn);
        free(incident_wave);
        free(herglotz_wave);
        free(wave_product);
    }

    //----------------------------------------------------------------------------------------------------------------------------------------------------------------

    template<typename I_ext, typename R_ext, typename C_ext,I_ext solver_count>
    void BoundaryPotential(const R_ext* kappa, C_ext* coeff, C_ext * wave, C_ext* phi, I_ext wave_chunk_count_, R_ext cg_tol, R_ext gmres_tol)
    {
        const C_ext One  = static_cast<C_ext>(Complex(1.0f,0.0f));
        const C_ext Zero = static_cast<C_ext>(Complex(0.0f,0.0f));

        const Int      n = VertexCount();

        ConjugateGradient<solver_count,C_ext,size_t> cg(n,100,8);
        GMRES<solver_count,C_ext,size_t,Side::Left> gmres(n,30,8);

        // setup the mass matrix Preconditionier P:=M^-1. P is also used for transf. into strong form
        auto mass = [&]( const C_ext * x, C_ext *y )
        {
            for( Int chunk = 0; chunk < wave_chunk_count - 1; ++chunk )
            {
                Mass.Dot(
                    One, &x [wave_chunk_size * chunk], wave_count,
                    Zero,  &y[wave_chunk_size * chunk], wave_count,
                    wave_chunk_size
                );
            }
            {
                const Int chunk = wave_chunk_count - 1;
                Mass.Dot(
                    One, &x [wave_chunk_size * chunk], wave_count,
                    Zero,  &y[wave_chunk_size * chunk], wave_count,
                    wave_count - wave_chunk_size*chunk
                );
            }
        };

        auto id = [&]( const C_ext * x, C_ext *y )
        {
            memcpy(y,x,wave_count * n * sizeof(C_ext));
        };

        auto P = [&]( const C_ext * x, C_ext *y )
        {
            bool succeeded = cg(mass,id,x,wave_count,y,wave_count,cg_tol);
        };

        // set up the bdry operator and solve
        for(int i = 0 ; i < wave_chunk_count ; i++)
        {
            coeff[4 * i + 0] = static_cast<C_ext>(Complex(0.5f,0.0f));
            coeff[4 * i + 1] = static_cast<C_ext>(Complex(0.0f,-kappa[i]));
            coeff[4 * i + 2] = One;
            coeff[4 * i + 3] = Zero;
        }

        kernel_list list = LoadKernel(kappa,coeff,wave_count,wave_chunk_size);

        auto A = [&]( const C_ext * x, C_ext *y )
        {   
            ApplyBoundaryOperators_PL(
                            wave_count, One,x,Zero,y
                            );
        };

        bool succeeded = gmres(A,P,wave,wave_count,phi,wave_count,gmres_tol,10);

        DestroyKernel(&list);
    }


    template<typename I_ext, typename R_ext, typename C_ext,I_ext solver_count>
    void DirichletToNeumann(const R_ext* kappa, C_ext * wave, C_ext* du_dn, I_ext wave_chunk_count_, R_ext cg_tol, R_ext gmres_tol)
    {
        const C_ext One  = static_cast<C_ext>(Complex(1.0f,0.0f));
        const C_ext I    = static_cast<C_ext>(Complex(0.0f,1.0f));
        const C_ext Zero = static_cast<C_ext>(Complex(0.0f,0.0f));

        const Int   n    = VertexCount();

        C_ext*  coeff    = (C_ext*)malloc(wave_chunk_count_ * 4 * sizeof(Complex));

        ConjugateGradient<solver_count,C_ext,size_t> cg(n,100,8);
        GMRES<solver_count,C_ext,size_t,Side::Left> gmres(n,30,8);

        // setup the mass matrix Preconditionier P:=M^-1. P is also used for transf. into strong form
        auto mass = [&]( const C_ext * x, C_ext *y )
        {
            for( Int chunk = 0; chunk < wave_chunk_count - 1; ++chunk )
            {
                Mass.Dot(
                    One, &x [wave_chunk_size * chunk], wave_count,
                    Zero,  &y[wave_chunk_size * chunk], wave_count,
                    wave_chunk_size
                );
            }
            {
                const Int chunk = wave_chunk_count - 1;
                Mass.Dot(
                    One, &x [wave_chunk_size * chunk], wave_count,
                    Zero,  &y[wave_chunk_size * chunk], wave_count,
                    wave_count - wave_chunk_size*chunk
                );
            }
        };

        auto id = [&]( const C_ext * x, C_ext *y )
        {
            memcpy(y,x,wave_count * n * sizeof(C_ext));
        };

        auto P = [&]( const C_ext * x, C_ext *y )
        {
            bool succeeded = cg(mass,id,x,wave_count,y,wave_count,cg_tol);
        };

        // set up the bdry operator and solve
        for(int i = 0 ; i < wave_chunk_count ; i++)
        {
            coeff[4 * i + 0] = static_cast<C_ext>(Complex(0.5f,0.0f));
            coeff[4 * i + 1] = -I;
            coeff[4 * i + 2] = Zero;
            coeff[4 * i + 3] = One;
        }

        kernel_list list = LoadKernel(kappa,coeff,wave_count,wave_chunk_size);

        auto A = [&]( const C_ext * x, C_ext *y )
        {   
            ApplyBoundaryOperators_PL(
                            wave_count, One,x,Zero,y
                            );
        };

        // solve for the normal derivatives of the near field solutions
        bool succeeded = gmres(A,P,wave,wave_count,du_dn,wave_count,gmres_tol,10);

        DestroyKernel(&list);

        free(coeff);
    }

    //------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    // calculate factor * Re(B_in) .* normals
    template<typename R_ext, typename C_ext>
    void MultiplyWithNormals_PL( ptr<C_ext> B_in, mut<R_ext> C_out, R_ext factor, const R_ext cg_tol)
    {
        const R_ext One  = static_cast<R_ext>(1.0f);
        const R_ext Zero = static_cast<R_ext>(0.0f);

        const Int m = SimplexCount();
        const Int n = VertexCount();

        Complex*    B = (Complex*)malloc(m * sizeof(Complex));
        Real*       C = (Real*)malloc(3 * m * sizeof(Complex));        
        R_ext* C_weak = (R_ext*)malloc( 3 * n * sizeof(R_ext));

        ConjugateGradient<3,R_ext,size_t> cg(n,100,8);

        auto id = [&]( const R_ext * x, R_ext *y )
        {
            memcpy(y,x,3 * n * sizeof(R_ext));
        };

        auto mass = [&]( const R_ext * x, R_ext *y )
        {
            Mass.Dot(
                One, x, 3,
                Zero, y, 3,
                3
            );
        };
        
        // make the input from PL to a PC function
        AvOp.Dot(
                Complex(1.0f,0.0f),  B_in,  1,
                Complex(0.0f,0.0f), B, 1,
                1
            );

        Int i,j;
        // pointwise multiplication of the STRONG FORM with the normals
        #pragma omp parallel for num_threads( OMP_thread_count ) schedule( static ) private(i) private(j)
        for( i = 0; i < m; ++i )
        {
            Real a_i = B[i].real() / areas_ptr[i];
            LOOP_UNROLL_FULL
            for( j = 0; j < 3; ++j )
            {
                C[i * 3 + j] = normals_ptr[i * 4 + j] * a_i;
            }
        }

        // retransf. from PC to PL
        AvOpTransp.Dot(
                factor, C, 3,
                Zero,  C_weak, 3,
                3
            );

        // apply M^(-1) to get trong form
        bool succeeded = cg(mass,id,C_weak,3,C_out,3,cg_tol);    

        free(B);
        free(C);
        free(C_weak);
    }


    // calculate <B_in , normals>
    template<typename R_ext>
    void DotWithNormals_PL( ptr<R_ext> B_in, mut<R_ext> C_out, const R_ext cg_tol)
    {
        const R_ext One  = static_cast<R_ext>(1.0f);
        const R_ext Zero = static_cast<R_ext>(0.0f);

        const Int   m = SimplexCount();
        const Int   n = VertexCount();

        Real*       B = (Real*)malloc( 3 * m * sizeof(Real));
        Real*       C = (Real*)malloc( m * sizeof(Real));        
        R_ext* C_weak = (R_ext*)malloc( n * sizeof(R_ext));

        ConjugateGradient<1,R_ext,size_t> cg(n,100,8);

        auto id = [&]( const R_ext * x, R_ext *y )
        {
            memcpy(y,x, n * sizeof(R_ext));
        };

        auto mass = [&]( const R_ext * x, R_ext *y )
        {
            Mass.Dot(
                One, x, 1,
                Zero, y, 1,
                1
            );
        };
        
        // make the input from PL to a PC function
        AvOp.Dot( 
                1.0f,  B_in,  3,
                0.0f, B, 3,
                3
            );

        Int i,j;
        // pointwise multiplication of the STRONG FORM with the normals
        #pragma omp parallel for num_threads( OMP_thread_count ) schedule( static ) private(i) private(j)
        for( i = 0; i < m; ++i )
        {
            Real a_i = 1.0f / areas_ptr[i];
            LOOP_UNROLL_FULL
            for( j = 0; j < 3; ++j )
            {
                C[i] += normals_ptr[i * 4 + j] * B[i * 3 + j] * a_i;
            }
        }

        // retransf. from PC to PL
        AvOpTransp.Dot(
                One, C, 1,
                Zero,  C_weak, 1,
                1
            );
        
        // apply M^(-1) to get trong form
        bool succeeded = cg(mass,id,C_weak,1,C_out,1,cg_tol);    
        
        free(B);
        free(C);
        free(C_weak);
    }