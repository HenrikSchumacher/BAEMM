public:

    template<size_t solver_count, typename I_ext, typename R_ext, typename C_ext>
    void FarField(const R_ext* kappa, const I_ext& wave_chunk_count_, 
                    R_ext* inc_directions,  const I_ext& wave_chunk_size_, C_ext* C_out, 
                    R_ext cg_tol, R_ext gmres_tol)
    {
        ptic(ClassName()+"::FarField");
        // Implement the bdry to Farfield map. wave ist the std incident wave defined pointwise by exp(i*kappa*<x,d>). A = (1/2) * I - i * kappa * SL + DL
        // phi = A\wave is the bdry potential which will be mapped onto the far field
        const C_ext One  = static_cast<C_ext>(Complex(1.0f,0.0f));
        const C_ext Zero = static_cast<C_ext>(Complex(0.0f,0.0f));

        const I_ext  n           = static_cast<I_ext>(VertexCount());
        const I_ext  wave_count_ = wave_chunk_count_ * wave_chunk_size_;       

        Tensor2<C_ext,I_ext>  inc_coeff ( wave_chunk_count_, 4 );
        Tensor2<C_ext,I_ext>  coeff (  wave_chunk_count_, 4  );
        Tensor2<C_ext,I_ext>  wave  (  wave_count_, n  );     //weak representation of the incident wave
        Tensor2<C_ext,I_ext>  phi   (   wave_count_, n);

        C_ext* inc_coeff_ptr = inc_coeff.data();

        phi.SetZero();

        // create weak representation of the negative incident wave
        for(I_ext i = 0 ; i < wave_chunk_count_ ; i++)
        {
            inc_coeff_ptr[4 * i + 0] = Zero;
            inc_coeff_ptr[4 * i + 1] = -One;
            inc_coeff_ptr[4 * i + 2] = Zero;
            inc_coeff_ptr[4 * i + 3] = Zero;
        }


        CreateIncidentWave_PL(One, inc_directions, wave_chunk_size_,
                            Zero, wave.data(), wave_count_,
                            kappa, inc_coeff.data(), wave_count_, wave_chunk_size_
                            );


        BoundaryPotential<solver_count>( kappa, coeff.data(), wave.data(), phi.data(), wave_chunk_count_, wave_chunk_size_, cg_tol, gmres_tol );      


        ApplyFarFieldOperators_PL( One, phi.data(), wave_count_,
                            Zero, C_out, wave_count_,
                            kappa,coeff.data(), wave_count_, wave_chunk_size_
                            );

        ptoc(ClassName()+"::FarField");
    }


    template<size_t solver_count, typename I_ext, typename R_ext, typename C_ext>
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
        ptic(ClassName()+"::Derivative_FF");

        const C_ext One  = static_cast<C_ext>(Complex(1.0f,0.0f));
        const C_ext I    = static_cast<C_ext>(Complex(0.0f,1.0f));
        const C_ext Zero = static_cast<C_ext>(Complex(0.0f,0.0f));

        const I_ext  n           = static_cast<I_ext>(VertexCount());
        const I_ext  wave_count_ = wave_chunk_count_ * wave_chunk_size_;
        

        Tensor2<C_ext,I_ext>  coeff (  wave_chunk_count_, 4  );
        Tensor2<C_ext,I_ext>  boundary_conditions (    wave_count_, n  ); 
        Tensor2<C_ext,I_ext>  boundary_conditions_weak (   wave_count_, n );
        Tensor1<R_ext,I_ext>  h_n  (   n  );
        Tensor2<C_ext,I_ext>  phi  (   wave_count_, n  ); 

        C_ext* boundary_conditions_ptr = boundary_conditions.data();

        phi.SetZero();

        if (*pdu_dn == NULL)
        {            
            Tensor2<C_ext,I_ext>  inc_coeff (  wave_chunk_count_, 4  );
            Tensor2<C_ext,I_ext>  incident_wave (  wave_count_, n );  //weak representation of the incident wave 
            
            C_ext* inc_coeff_ptr = inc_coeff.data();

            *pdu_dn           = (C_ext*)calloc(wave_count_ * n, sizeof(C_ext)); 

            // create weak representation of the negative incident wave
            for(I_ext i = 0 ; i < wave_chunk_count_ ; i++)
            {
                inc_coeff_ptr[4 * i + 0] = Zero;
                inc_coeff_ptr[4 * i + 1] = -I;
                inc_coeff_ptr[4 * i + 2] = One;
                inc_coeff_ptr[4 * i + 3] = Zero;
            }

            CreateIncidentWave_PL( One, inc_directions, wave_chunk_size_,
                                Zero, incident_wave.data(), wave_count_,
                                kappa, inc_coeff.data(), wave_count_, wave_chunk_size_
                                );
            

            DirichletToNeumann<solver_count>( kappa, incident_wave.data(), *pdu_dn, wave_chunk_count_, wave_chunk_size_, cg_tol, gmres_tol ); 
        }


        DotWithNormals_PL( h, h_n.data(), cg_tol );

            
        // CheckThis
        ParallelDo(
            [boundary_conditions_ptr,&h_n,&pdu_dn]( const I_ext i )
            {   
                ptr<C_ext> a = &(*pdu_dn)[i * solver_count];
                ptr<R_ext> b = h_n.data();
                mut<C_ext> c = &boundary_conditions_ptr[i * solver_count];
                
                
                LOOP_UNROLL(8)
                for(I_ext j = 0; j < solver_count; ++j )
                {
                    c[j] = - a[j] * b[i];
                }
            },
            I_ext(n), I_ext(CPU_thread_count)
        );
        
        // apply mass to the boundary conditions to get weak representation
        Mass.Dot(
            One, boundary_conditions.data(), wave_count_,
            Zero,  boundary_conditions_weak.data(), wave_count_,
            wave_count_
        );

        BoundaryPotential<solver_count>( kappa, coeff.data(), boundary_conditions_weak.data(), phi.data(), 
                                                            wave_chunk_count_, wave_chunk_size_, cg_tol, gmres_tol);

        ApplyFarFieldOperators_PL( One, phi.data(), wave_count_,
                            Zero, C_out, wave_count_,
                            kappa,coeff.data(), wave_count_, wave_chunk_size_
                            );

        ptoc(ClassName()+"::Derivative_FF");
    }

    template<size_t solver_count, typename I_ext, typename R_ext, typename C_ext>
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
        ptic(ClassName()+"::AdjointDerivative_FF");

        const C_ext One  = static_cast<C_ext>(Complex(1.0f,0.0f));
        const C_ext I    = static_cast<C_ext>(Complex(0.0f,1.0f));
        const C_ext Zero = static_cast<C_ext>(Complex(0.0f,0.0f));

        const I_ext  n                      = static_cast<I_ext>(VertexCount());
        const I_ext  wave_count_            = wave_chunk_count_ * wave_chunk_size_;


        Tensor2<C_ext,I_ext>  inc_coeff (  wave_chunk_count_, 4  );
        Tensor2<C_ext,I_ext>  herglotz_wave (  wave_count_, n  );     //weak representation of the herglotz wave
        Tensor2<C_ext,I_ext>  dv_dn (  wave_count_, n  );
        Tensor1<C_ext,I_ext>  wave_product (   n   );

        C_ext* inc_coeff_ptr = inc_coeff.data();

        dv_dn.SetZero();

        // create weak representation of the negative incident wave
        for(I_ext i = 0 ; i < wave_chunk_count_ ; i++)
        {
            inc_coeff_ptr[4 * i + 0] = Zero;
            inc_coeff_ptr[4 * i + 1] = -I;
            inc_coeff_ptr[4 * i + 2] = One;
            inc_coeff_ptr[4 * i + 3] = Zero;
        }

        if (*pdu_dn == NULL)
        {
            Tensor2<C_ext,I_ext>  incident_wave (  wave_count_, n );  //weak representation of the incident wave 
            
            *pdu_dn           = (C_ext*)calloc(wave_count_ * n, sizeof(C_ext)); 

            
            CreateIncidentWave_PL( One, inc_directions, wave_chunk_size_,
                                Zero, incident_wave.data(), wave_count_,
                                kappa, inc_coeff.data(), wave_count_, wave_chunk_size_
                                );
            

            DirichletToNeumann<solver_count>( kappa, incident_wave.data(), *pdu_dn, wave_chunk_count_, wave_chunk_size_, cg_tol, gmres_tol ); 
        }

        CreateHerglotzWave_PL(One, g, wave_count_,
                            Zero, herglotz_wave.data(), wave_count_,
                            kappa, inc_coeff.data(), wave_count_, wave_chunk_size_
                            );
        
        // solve for the normal derivatives of the near field solutions
        DirichletToNeumann<solver_count>( kappa, herglotz_wave.data(), dv_dn.data(), wave_chunk_count_, wave_chunk_size_, cg_tol, gmres_tol );
        
        // calculate du_dn .* dv_dn and sum over the leading dimension
        HadamardProduct( *pdu_dn, dv_dn.data(), wave_product.data(), n, wave_count_, true);

        // calculate (-1/wave_count)*Re(du_dn .* dv_dn).*normals
        MultiplyWithNormals_PL(wave_product.data(),C_out,-( 1 /static_cast<R_ext>(wave_count_) ), cg_tol);

        ptoc(ClassName()+"::AdjointDerivative_FF");
    }

    template<size_t solver_count, typename I_ext, typename R_ext, typename C_ext, typename M_T, typename P_T>
    void GaussNewtonStep(const R_ext* kappa, const I_ext& wave_chunk_count_, 
                    const R_ext* inc_directions,  const I_ext& wave_chunk_size_, 
                    M_T M, P_T P,
                    const R_ext* h, R_ext* C_out, 
                    C_ext** pdu_dn,                         //pdu_dn is the pointer to the Neumann data of the scattered wave
                    R_ext cg_tol, R_ext gmres_tol_inner , R_ext gmres_tol_outer
                    )
    {
        // Calculates a gauss newton step. Note that the metric M has to add the input to the result.
        const I_ext  n           = static_cast<I_ext>(VertexCount());
        const I_ext  m           = static_cast<I_ext>(GetMeasCount());
        const I_ext  wave_count_ = wave_chunk_count_ * wave_chunk_size_;

        GMRES<3,R_ext,size_t,Side::Left> gmres(n,30,CPU_thread_count);
        
        Tensor2<C_ext,I_ext>  DF (  wave_count_, m  );
        Tensor2<R_ext,I_ext>  y_strong (  3, n  );

        auto A = [&]( const R_ext * x, R_ext *y )
        {   
            Derivative_FF<solver_count>( kappa, wave_chunk_count_, inc_directions, wave_chunk_size_,
                        x, DF.data(), pdu_dn, cg_tol, gmres_tol_inner);
            AdjointDerivative_FF<solver_count>( kappa, wave_chunk_count_, inc_directions, wave_chunk_size_,
                        DF.data(), y_strong.data(), pdu_dn, cg_tol, gmres_tol_inner);

            Mass.Dot(
                Tools::Scalar::One<R_ext>, y_strong.data(), 3,
                Tools::Scalar::Zero<R_ext>, y, 3,
                3
            );

            M(x,y); // The metric m has to return y + M*y
        };

        zerofy_buffer(C_out, (size_t)(3 * n), CPU_thread_count);

        bool succeeded = gmres(A,P,h,3,C_out,3,gmres_tol_outer,3);
    }

    //----------------------------------------------------------------------------------------------------------------------------------------------------------------

    template<size_t solver_count, typename I_ext, typename R_ext, typename C_ext>
    void BoundaryPotential(const R_ext* kappa, C_ext* coeff, C_ext * wave, C_ext* phi, 
                            const I_ext& wave_chunk_count_, const I_ext& wave_chunk_size_, 
                            R_ext cg_tol, R_ext gmres_tol)
    {
        ptic(ClassName()+"::BoundaryPotential");

        const C_ext One  = static_cast<C_ext>(Complex(1.0f,0.0f));
        const C_ext Zero = static_cast<C_ext>(Complex(0.0f,0.0f));

        const Int  n     = VertexCount();

        const I_ext wave_count_ = wave_chunk_count_ * wave_chunk_size_;

        GMRES<solver_count,C_ext,size_t,Side::Left> gmres(n,30,CPU_thread_count);

        // setup the mass matrix Preconditionier P:=M^-1. P is also used for transf. into strong form

        auto P = [&]( const C_ext * x, C_ext *y )
        {
            ApplyMassInverse<solver_count>(x,y,wave_count_,cg_tol);
        };

        // set up the bdry operator and solve
        for(Int i = 0 ; i < wave_chunk_count_ ; i++)
        {
            coeff[4 * i + 0] = static_cast<C_ext>(Complex(0.5f,0.0f));
            coeff[4 * i + 1] = static_cast<C_ext>(Complex(0.0f,-kappa[i]));
            coeff[4 * i + 2] = One;
            coeff[4 * i + 3] = Zero;
        }

        kernel_list list = LoadKernel(kappa,coeff,wave_count_,wave_chunk_size_);

        auto A = [&]( const C_ext * x, C_ext *y )
        {   
            ApplyBoundaryOperators_PL(
                            wave_count_, One,x,Zero,y
                            );
        };

        bool succeeded = gmres(A,P,wave,wave_count_,phi,wave_count_,gmres_tol,10);

        DestroyKernel(&list);

        ptoc(ClassName()+"::BoundaryPotential");
    }


    template<size_t solver_count, typename I_ext, typename R_ext, typename C_ext>
    void DirichletToNeumann(const R_ext* kappa, C_ext * wave, C_ext* neumann_trace, 
                            const I_ext& wave_chunk_count_, const I_ext& wave_chunk_size_, 
                            R_ext cg_tol, R_ext gmres_tol)
    {
        ptic(ClassName()+"::DirichletToNeumann");
        const C_ext One  = static_cast<C_ext>(Complex(1.0f,0.0f));
        const C_ext I    = static_cast<C_ext>(Complex(0.0f,1.0f));
        const C_ext Zero = static_cast<C_ext>(Complex(0.0f,0.0f));

        const Int    n   = VertexCount();

        const I_ext wave_count_ = wave_chunk_count_ * wave_chunk_size_;

        C_ext*  coeff    = (C_ext*)malloc(wave_chunk_count_ * 4 * sizeof(C_ext));

        GMRES<solver_count,C_ext,size_t,Side::Left> gmres(n,30,CPU_thread_count);

        // setup the mass matrix Preconditionier P:=M^-1. P is also used for transf. into strong form

        auto P = [&]( const C_ext * x, C_ext *y )
        {
            ApplyMassInverse<solver_count>(x,y,wave_count_,cg_tol);
        };

        // set up the bdry operator and solve
        for(Int i = 0 ; i < wave_chunk_count_ ; i++)
        {
            coeff[4 * i + 0] = static_cast<C_ext>(Complex(0.5f,0.0f));
            coeff[4 * i + 1] = -I;
            coeff[4 * i + 2] = Zero;
            coeff[4 * i + 3] = One;
        }

        kernel_list list = LoadKernel(kappa,coeff,wave_count_,wave_chunk_size_);

        auto A = [&]( const C_ext * x, C_ext *y )
        {   
            ApplyBoundaryOperators_PL(
                             wave_count_,One,x,Zero,y
                            );
        };

        // solve for the normal derivatives of the near field solutions
        bool succeeded = gmres(A,P,wave,wave_count_,neumann_trace,wave_count_,gmres_tol,10);

        DestroyKernel(&list);

        free(coeff);

        ptoc(ClassName()+"::DirichletToNeumann");
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
        
        // make the input from PL to a PC function
        AvOp.Dot(
                Complex(1.0f,0.0f),  B_in,  1,
                Complex(0.0f,0.0f), B, 1,
                1
            );


        // pointwise multiplication of the STRONG FORM with the normals
        // CheckThis
        ParallelDo(
            [=]( const Int i )
            {
                Real a_i = B[i].real() / areas_ptr[i];
                LOOP_UNROLL(3)
                for(Int j = 0; j < 3; ++j )
                {
                    C[i * 3 + j] = normals_ptr[i * 4 + j] * a_i;
                }
            },
            m, CPU_thread_count
        );

        // retransf. from PC to PL
        AvOpTransp.Dot(
                factor, C, 3,
                Zero,  C_weak, 3,
                3
            );  

        ApplyMassInverse<3>(C_weak,C_out,3,cg_tol);

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
        
        // make the input from PL to a PC function
        AvOp.Dot( 
                1.0f,  B_in,  3,
                0.0f, B, 3,
                3
            );

        // pointwise multiplication of the STRONG FORM with the normals
        // CheckThis
        ParallelDo(
            [=]( const Int i )
            {
                Real a_i = 1.0f / areas_ptr[i];
                LOOP_UNROLL_FULL
                for(Int j = 0; j < 3; ++j )
                {
                    C[i] += normals_ptr[i * 4 + j] * B[i * 3 + j] * a_i;
                }
            },
            m, CPU_thread_count
        );

        // retransf. from PC to PL
        AvOpTransp.Dot(
                One, C, 1,
                Zero,  C_weak, 1,
                1
            ); 
        
        ApplyMassInverse<1>(C_weak,C_out,1,cg_tol);

        free(B);
        free(C);
        free(C_weak);
    }

    // template<typename I, typename T>
    // void ApplyMass(const T* B_in, T* C_out, const I ld)
    // {
    //     Mass.Dot(
    //         Tools::Scalar::One<T>, B_in, ld,
    //         Tools::Scalar::Zero<T>, C_out, ld,
    //         ld
    //     ); 
    // }

    template<size_t solver_count, typename I, typename T, typename R>
    void ApplyMassInverse(const T* B_in, T* C_out, const I ld, const R cg_tol)
    {
        const I   n = VertexCount();

        ConjugateGradient<solver_count,T,size_t> cg(n,200,CPU_thread_count);

        auto id = [&]( const T * x, T *y )
        {
            memcpy(y,x, ld * n * sizeof(T));
        };

        auto mass = [&]( const T * x, T *y )
        {
            Mass.Dot(
                Tools::Scalar::One<T>, x, ld,
                Tools::Scalar::Zero<T>, y, ld,
                ld
            );
        };

        zerofy_buffer(C_out, ld * static_cast<size_t>(n), CPU_thread_count);
        bool succeeded = cg(mass,id,B_in,ld,C_out,ld,cg_tol);  
    }
