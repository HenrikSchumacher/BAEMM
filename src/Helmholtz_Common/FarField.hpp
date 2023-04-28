public:

    template<typename I_ext, typename R_ext, typename C_ext,I_ext solver_count>
    void FarField(const R_ext* kappa, const I_ext& wave_chunk_count_, R_ext* inc_directions,  const I_ext& wave_chunk_size_, C_ext* C_out)
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
            bool succeeded = cg(mass,id,x,wave_count_,y,wave_count_,0.00001f);
        };

        // set up the bdry operator and solve
        for(int i = 0 ; i < wave_chunk_count ; i++)
        {
            coeff[4 * i + 0] = static_cast<C_ext>(Complex(0.5f,0.0f));
            coeff[4 * i + 1] = static_cast<C_ext>(Complex(0.0f,-kappa[i]));
            coeff[4 * i + 2] = static_cast<C_ext>(Complex(1.0f,0.0f));
            coeff[4 * i + 3] = static_cast<C_ext>(Complex(0.0f,0.0f));
        }

        kernel_list list = LoadKernel(kappa,coeff,wave_count,wave_chunk_size);

        auto A = [&]( const C_ext * x, C_ext *y )
        {   
            ApplyBoundaryOperators_PL(
                            wave_count, One,x,Zero,y
                            );
        };

        bool succeeded = gmres(A,P,wave,wave_count,phi,wave_count,0.00001f,10);

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