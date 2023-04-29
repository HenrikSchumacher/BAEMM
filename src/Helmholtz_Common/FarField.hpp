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

        const I_ext  n          = VertexCount();
        const I_ext  wave_count_ = wave_chunk_count_ * wave_chunk_size_;
        

        C_ext*  inc_coeff       = (C_ext*)malloc(wave_chunk_count_ * 4 * sizeof(Complex));
        C_ext*  coeff           = (C_ext*)malloc(wave_chunk_count_ * 4 * sizeof(Complex));
        C_ext*  wave            = (C_ext*)malloc(wave_count_ * n * sizeof(Complex));     //weak representation of the incident wave
        C_ext*  phi             = (C_ext*)malloc(wave_count_ * n * sizeof(Complex));
        
        ConjugateGradient<solver_count,std::complex<float>,size_t> cg(n,100,8);
        GMRES<solver_count,std::complex<float>,size_t,Side::Left> gmres(n,30,8);

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
            bool succeeded = cg(mass,id,x,wave_count_,y,wave_count_,cg_tol);
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
    void AdjointDerivative_FF(const R_ext* kappa, const I_ext& wave_chunk_count_, 
                    C_ext* inc_directions,  const I_ext& wave_chunk_size_, 
                    C_ext* g, R_ext* C_out, 
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

        const I_ext  n                      = SimplexCount();
        const I_ext  wave_count_            = wave_chunk_count_ * wave_chunk_size_;
        const R_ext  one_over_wave_count    = 1 / ((R_ext)wave_count);

        C_ext*  inc_coeff       = (C_ext*)malloc(wave_chunk_count_ * 4 * sizeof(Complex));
        C_ext*  coeff           = (C_ext*)malloc(wave_chunk_count_ * 4 * sizeof(Complex));
        C_ext*  incident_wave   = (C_ext*)malloc(wave_count_ * n * sizeof(Complex));     //weak representation of the incident wave
        C_ext*  herglotz_wave   = (C_ext*)malloc(wave_count_ * n * sizeof(Complex));     //weak representation of the incident wave
        C_ext*  du_dn           = (C_ext*)malloc(wave_count_ * n * sizeof(Complex));
        C_ext*  dv_dn           = (C_ext*)malloc(wave_count_ * n * sizeof(Complex));
        C_ext*  wave_product    = (C_ext*)malloc(n * sizeof(Complex));
        
        ConjugateGradient<solver_count,std::complex<float>,size_t> cg(n,100,8);
        GMRES<solver_count,std::complex<float>,size_t,Side::Left> gmres(n,30,8);

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
            bool succeeded = cg(mass,id,x,wave_count_,y,wave_count_,cg_tol);
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
        bool succeeded = gmres(A,P,incident_wave,wave_count,du_dn,wave_count,gmres_tol,10);
        bool succeeded_h = gmres(A,P,herglotz_wave,wave_count,dv_dn,wave_count,gmres_tol,10);

        DestroyKernel(&list);

        // calculate du_dn .* dv_dn and sum over the leading dimension
        HadamardProduct(du_dn,dv_dn,wave_product,n,wave_count,true);

        MultiplyWithNormals(wave_product,C_out,one_over_wave_count, cg_tol)

        free(inc_coeff);
        free(coeff);
        free(du_dn);
        free(dv_dn);
        free(incident_wave);
        free(herglotz_wave);
        free(wave_product);
    }

    // calculate B_in .* normals
    void MultiplyWithNormals_PL( ptr<Complex> B_in, mut<Complex> C_out, C_ext factor, const R_ext cg_tol)
    {
        const Int m = SimplexCount();
        const Int n = VertexCount();
        Complex* B = (Complex*)malloc(m * sizeof(Complex));
        Complex* C = (Complex*)malloc(3 *m * sizeof(Complex));

        ConjugateGradient<3,std::complex<float>,size_t> cg(n,100,8);

        auto id = [&]( const Complex * x, Complex *y )
        {
            memcpy(y,x,wave_count * n * sizeof(C_ext));
        };

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

        // make the input from PL to a PC function
        AvOp.Dot(
                factor,  B_in,  1,
                Complex(0.0f,0.0f), B, 1,
                1
            );
        
        // pointwise multiplication of the STRONG FORM with the normals
        #pragma omp parallel for num_threads( OMP_thread_count ) schedule( static ) private(i) private(j)
        for( i = 0; i < m; ++i )
        {
            Complex a_i = B[i] * (1/areas_ptr[i]);
            LOOP_UNROLL_FULL
            for( j = 0; j < 3; ++j )
            {
                C[i * columns + j] = normals_ptr[i * columns + j] * a_i;
            }
        }

        realloc( B, 3 * n * sizeof(Complex));
        // retransf. from PC to PL
        AvOpTransp.Dot(
                Complex(1.0f,0.0f), C, 3,
                Complex(0.0f,0.0f),  B, 3,
                3
            );
        // apply M^(-1) to get trong form
        bool succeeded = cg(mass,id,B,3,C_out,3,cg_tol);    

        free(B);
        free(C);
    }