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

        const I_ext  n           = static_cast<I_ext>(VertexCount());
        const I_ext  wave_count_ = wave_chunk_count_ * wave_chunk_size_;       

        C_ext*  inc_coeff       = (C_ext*)malloc(wave_chunk_count_ * 4 * sizeof(C_ext));
        C_ext*  coeff           = (C_ext*)malloc(wave_chunk_count_ * 4 * sizeof(C_ext));
        C_ext*  wave            = (C_ext*)malloc(wave_count_ * n * sizeof(C_ext));     //weak representation of the incident wave
        C_ext*  phi             = (C_ext*)calloc(wave_count_ * n, sizeof(C_ext));

        // create weak representation of the negative incident wave
        for(I_ext i = 0 ; i < wave_chunk_count_ ; i++)
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

        BoundaryPotential<R_ext,C_ext,solver_count>( kappa, coeff, wave, phi, cg_tol, gmres_tol );      

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
                    C_ext** pdu_dn,                 //pdu_dn is the pointer to the Neumann data of the scattered wave
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

        const I_ext  n           = static_cast<I_ext>(VertexCount());
        const I_ext  wave_count_ = wave_chunk_count_ * wave_chunk_size_;
        

        C_ext*  coeff                         = (C_ext*)malloc(wave_chunk_count_ * 4 * sizeof(C_ext));
        C_ext*  boundary_conditions           = (C_ext*)malloc(wave_count_ * n * sizeof(C_ext)); 
        C_ext*  boundary_conditions_weak      = (C_ext*)malloc(wave_count_ * n * sizeof(C_ext));
        R_ext*  h_n                           = (R_ext*)calloc(n, sizeof(R_ext));
        C_ext*  phi                           = (C_ext*)calloc(wave_count_ * n, sizeof(C_ext)); 
        
        if (*pdu_dn == NULL)
        {            
            C_ext*  inc_coeff                     = (C_ext*)malloc(wave_chunk_count_ * 4 * sizeof(C_ext));
            C_ext*  incident_wave                 = (C_ext*)malloc(wave_count_ * n * sizeof(C_ext));  //weak representation of the incident wave 

            *pdu_dn           = (C_ext*)calloc(wave_count_ * n, sizeof(C_ext)); 

            // create weak representation of the negative incident wave
            for(I_ext i = 0 ; i < wave_chunk_count_ ; i++)
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
            

            DirichletToNeumann<R_ext,C_ext,solver_count>( kappa, incident_wave, *pdu_dn, cg_tol, gmres_tol ); 

            free(inc_coeff);
            free(incident_wave);
        }


        DotWithNormals_PL( h, h_n, cg_tol );


        #pragma omp parallel for num_threads( OMP_thread_count ) schedule( static )
        for(I_ext i = 0; i < n; ++i )
        {
            LOOP_UNROLL(8)
            for(I_ext j = 0; j < wave_count_; ++j )
            {
                boundary_conditions[i * wave_count_ + j] = (*pdu_dn)[i * wave_count_ + j] * h_n[i];
            }
            std::cout << boundary_conditions[i * wave_count_ ] << std::endl;
        }
        
        // apply mass to the boundary conditions to get weak representation
        Mass.Dot(
            One, boundary_conditions, wave_count_,
            Zero,  boundary_conditions_weak, wave_count_,
            wave_count_
        );

        BoundaryPotential<R_ext,C_ext,solver_count>( kappa, coeff, boundary_conditions_weak, phi, cg_tol, gmres_tol);

        ApplyFarFieldOperators_PL( One, phi, wave_count_,
                            Zero, C_out, wave_count_,
                            kappa,coeff, wave_count_, wave_chunk_size_
                            );

        free(coeff);
        free(phi);
        free(boundary_conditions);
        free(boundary_conditions_weak);
        free(h_n);
    }

    template<typename I_ext, typename R_ext, typename C_ext,I_ext solver_count>
    void AdjointDerivative_FF(const R_ext* kappa, const I_ext& wave_chunk_count_, 
                    const R_ext* inc_directions,  const I_ext& wave_chunk_size_, 
                    const C_ext* g, R_ext* C_out, 
                    C_ext** pdu_dn,                         //pdu_dn is the pointer to the Neumann data of the scattered wave
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

        const I_ext  n                      = static_cast<I_ext>(VertexCount());
        const I_ext  wave_count_            = wave_chunk_count_ * wave_chunk_size_;


        C_ext*  inc_coeff       = (C_ext*)malloc(wave_chunk_count_ * 4 * sizeof(C_ext));
        C_ext*  herglotz_wave   = (C_ext*)malloc(wave_count_ * n * sizeof(C_ext));     //weak representation of the herglotz wave
        C_ext*  dv_dn           = (C_ext*)calloc(wave_count_ * n, sizeof(C_ext));
        C_ext*  wave_product    = (C_ext*)malloc(n * sizeof(C_ext));

        // create weak representation of the negative incident wave
        for(I_ext i = 0 ; i < wave_chunk_count_ ; i++)
        {
            inc_coeff[4 * i + 0] = Zero;
            inc_coeff[4 * i + 1] = -I;
            inc_coeff[4 * i + 2] = One;
            inc_coeff[4 * i + 3] = Zero;
        }

        if (*pdu_dn == NULL)
        {
            C_ext*  incident_wave   = (C_ext*)malloc(wave_count_ * n * sizeof(C_ext));     //weak representation of the incident wave
            
            *pdu_dn           = (C_ext*)calloc(wave_count_ * n, sizeof(C_ext)); 

            
            CreateIncidentWave_PL( One, inc_directions, wave_chunk_size_,
                                Zero, incident_wave, wave_count_,
                                kappa, inc_coeff, wave_count_, wave_chunk_size_
                                );
            

            DirichletToNeumann<R_ext,C_ext,solver_count>( kappa, incident_wave, *pdu_dn, cg_tol, gmres_tol ); 

            free(incident_wave);
        }

        CreateHerglotzWave_PL(One, g, wave_count_,
                            Zero, herglotz_wave, wave_count_,
                            kappa, inc_coeff, wave_count_, wave_chunk_size_
                            );
        
        // solve for the normal derivatives of the near field solutions
        DirichletToNeumann<R_ext,C_ext,solver_count>( kappa, herglotz_wave, dv_dn, cg_tol, gmres_tol );
        
        // calculate du_dn .* dv_dn and sum over the leading dimension
        HadamardProduct( *pdu_dn, dv_dn, wave_product, n, wave_count_, true);

        // calculate (-1/wave_count)*Re(du_dn .* dv_dn).*normals
        MultiplyWithNormals_PL(wave_product,C_out,-( 1 /static_cast<R_ext>(wave_count_) ), cg_tol);

        free(inc_coeff);
        free(dv_dn);
        free(herglotz_wave);
        free(wave_product);
    }

    // template<typename I_ext, typename R_ext, typename C_ext, I_ext solver_count>
    // void GaussNewtonStep(const R_ext* kappa, const I_ext& wave_chunk_count_, 
    //                 const R_ext* inc_directions,  const I_ext& wave_chunk_size_, 
    //                 // Operator M, Operator P,
    //                 const R_ext* h, R_ext* C_out, 
    //                 C_ext** pdu_dn,                         //pdu_dn is the pointer to the Neumann data of the scattered wave
    //                 R_ext cg_tol, R_ext gmres_tol_inner , R_ext gmres_tol_outer
    //                 )
    // {
    //     // Calculates a gauss newton step. Note that the metric M has to add the input to the result.
    //     const I_ext  n           = static_cast<I_ext>(VertexCount());
    //     const I_ext  wave_count_ = wave_chunk_count_ * wave_chunk_size_;

    //     C_ext*  DFa           = (C_ext*)malloc(wave_count_ * n * sizeof(C_ext));

    //     Derivative_FF<I_ext,R_ext,C_ext,solver_count>( kappa, wave_chunk_count_, inc_directions, wave_chunk_size_,
    //                     h, DFa, pdu_dn, cg_tol, gmres_tol_inner);
    //     AdjointDerivative_FF<I_ext,R_ext,C_ext,solver_count>( kappa, wave_chunk_count_, inc_directions, wave_chunk_size_,
    //                     DFa, C_out, pdu_dn, cg_tol, gmres_tol_inner);

    //     GMRES<3,R_ext,size_t,Side::Left> gmres(n,30,OMP_thread_count);

    //     auto A = [&]( const R_ext * x, R_ext *y )
    //     {   
    //         Derivative_FF<I_ext,R_ext,C_ext,solver_count>( kappa, wave_chunk_count_, inc_directions, wave_chunk_size_,
    //                     x, DFa, cg_tol, gmres_tol_inner);
    //         AdjointDerivative_FF<I_ext,R_ext,C_ext,solver_count>( kappa, wave_chunk_count_, inc_directions, wave_chunk_size_,
    //                     DFa, y, cg_tol, gmres_tol_inner);
    //         M(x,y); // The metric m has to return y + M*y
    //     };

    //     zerofy_buffer(C_out, (size_t)(3 * n), OMP_thread_count);

    //     bool succeeded = gmres(A,P,h,3,C_out,3,gmres_tol_outer,10);
    //     free(DFa);
    // }

    //----------------------------------------------------------------------------------------------------------------------------------------------------------------

    template<typename R_ext, typename C_ext,Int solver_count>
    void BoundaryPotential(const R_ext* kappa, C_ext* coeff, C_ext * wave, C_ext* phi, R_ext cg_tol, R_ext gmres_tol)
    {
        const C_ext One  = static_cast<C_ext>(Complex(1.0f,0.0f));
        const C_ext Zero = static_cast<C_ext>(Complex(0.0f,0.0f));

        const Int  n     = VertexCount();

        ConjugateGradient<solver_count,C_ext,size_t> cg(n,100,OMP_thread_count);
        GMRES<solver_count,C_ext,size_t,Side::Left> gmres(n,30,OMP_thread_count);

        // setup the mass matrix Preconditionier P:=M^-1. P is also used for transf. into strong form
        auto mass = [&]( const C_ext * x, C_ext *y )
        {
            Mass.Dot(
                One, x, wave_count,
                Zero,  y, wave_count,
                wave_count
            );
        };

        auto id = [&]( const C_ext * x, C_ext *y )
        {
            memcpy(y,x,wave_count * n * sizeof(C_ext));
        };

        auto P = [&]( const C_ext * x, C_ext *y )
        {
            zerofy_buffer(y, (size_t)(wave_count * n), OMP_thread_count);
            bool succeeded = cg(mass,id,x,wave_count,y,wave_count,cg_tol);
        };

        // set up the bdry operator and solve
        for(Int i = 0 ; i < wave_chunk_count ; i++)
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


    template<typename R_ext, typename C_ext,Int solver_count>
    void DirichletToNeumann(const R_ext* kappa, C_ext * wave, C_ext* neumann_trace, R_ext cg_tol, R_ext gmres_tol)
    {
        const C_ext One  = static_cast<C_ext>(Complex(1.0f,0.0f));
        const C_ext I    = static_cast<C_ext>(Complex(0.0f,1.0f));
        const C_ext Zero = static_cast<C_ext>(Complex(0.0f,0.0f));

        const Int    n   = VertexCount();

        C_ext*  coeff    = (C_ext*)malloc(wave_chunk_count * 4 * sizeof(C_ext));

        ConjugateGradient<solver_count,C_ext,size_t> cg(n,100,OMP_thread_count);
        GMRES<solver_count,C_ext,size_t,Side::Left> gmres(n,30,OMP_thread_count);

        // setup the mass matrix Preconditionier P:=M^-1. P is also used for transf. into strong form
        auto mass = [&]( const C_ext * x, C_ext *y )
        {
            Mass.Dot(
                One, x, wave_count,
                Zero,  y, wave_count,
                wave_count
            );
        };

        auto id = [&]( const C_ext * x, C_ext *y )
        {
            memcpy(y,x,wave_count * n * sizeof(C_ext));
        };

        auto P = [&]( const C_ext * x, C_ext *y )
        {
            zerofy_buffer(y, (size_t)(wave_count * n), OMP_thread_count);
            bool succeeded = cg(mass,id,x,wave_count,y,wave_count,cg_tol);
        };

        // set up the bdry operator and solve
        for(Int i = 0 ; i < wave_chunk_count ; i++)
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
                             wave_count,One,x,Zero,y
                            );
        };

        // solve for the normal derivatives of the near field solutions
        bool succeeded = gmres(A,P,wave,wave_count,neumann_trace,wave_count,gmres_tol,10);

        DestroyKernel(&list);

        free(coeff);
    }

    //------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    // calculate factor * Re(B_in) .* normals
    template<typename R_ext, typename C_ext>
    void MultiplyWithNormals_PL( const C_ext* B_in, R_ext* C_out, R_ext factor, const R_ext cg_tol)
    {
        const R_ext One  = static_cast<R_ext>(1.0f);
        const R_ext Zero = static_cast<R_ext>(0.0f);

        const Int m = SimplexCount();
        const Int n = VertexCount();

        Complex*    B = (Complex*)malloc(m * sizeof(Complex));
        Real*       C = (Real*)malloc(3 * m * sizeof(Real));        
        R_ext* C_weak = (R_ext*)malloc( 3 * n * sizeof(R_ext));

        ConjugateGradient<3,R_ext,size_t> cg(n,100,OMP_thread_count);

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


        // pointwise multiplication of the STRONG FORM with the normals
        #pragma omp parallel for num_threads( OMP_thread_count ) schedule( static )
        for(Int i = 0; i < m; ++i )
        {
            Real a_i = B[i].real() / areas_ptr[i];
            LOOP_UNROLL(3)
            for(Int j = 0; j < 3; ++j )
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
        zerofy_buffer(C_out, static_cast<size_t>(3 * n), int_cast<Int>(16));
        bool succeeded = cg(mass,id,C_weak,3,C_out,3,cg_tol);    

        free(B);
        free(C);
        free(C_weak);
    }


    // calculate <B_in , normals>
    template<typename R_ext>
    void DotWithNormals_PL( const R_ext* B_in, R_ext* C_out, const R_ext cg_tol)
    {
        const R_ext One  = static_cast<R_ext>(1.0f);
        const R_ext Zero = static_cast<R_ext>(0.0f);

        const Int   m = SimplexCount();
        const Int   n = VertexCount();

        Real*       B = (Real*)malloc( 3 * m * sizeof(Real));
        Real*       C = (Real*)calloc( m, sizeof(Real));        
        R_ext* C_weak = (R_ext*)malloc( n * sizeof(R_ext));

        ConjugateGradient<1,R_ext,size_t> cg(n,100,OMP_thread_count);

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


        // pointwise multiplication of the STRONG FORM with the normals
        #pragma omp parallel for num_threads( OMP_thread_count ) schedule( static )
        for(Int i = 0; i < m; ++i )
        {
            Real a_i = 1.0f / areas_ptr[i];
            LOOP_UNROLL_FULL
            for(Int j = 0; j < 3; ++j )
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