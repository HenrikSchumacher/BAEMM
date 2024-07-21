public:

    template<Int EQ_COUNT, typename I_ext, typename R_ext, typename C_ext>
    void FarField(
        cptr<R_ext> kappa,
        const I_ext wave_chunk_count_,
        mptr<R_ext> inc_directions,
        const I_ext wave_chunk_size_,
        mptr<C_ext> C_out,
        const WaveType type,
        const R_ext cg_tol,
        const R_ext gmres_tol
    )
    {
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);
        
        std::string tag = ClassName()+"::FarField<" + ToString(EQ_COUNT)
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);
        
        // // Implement the bdry to Farfield map. 
        // wave ist the std incident wave defined pointwise by exp(i*kappa*<x,d>).
        // A = (1/2) * I - i * kappa * SL + DL
        // phi = A\wave is the bdry potential which will be mapped onto the far field.
        
        
        // No need to copy kappa to eta.
        
        FarField_parameters<EQ_COUNT>(
            kappa, wave_chunk_count_, inc_directions, wave_chunk_size_,
            C_out, type, kappa, cg_tol, gmres_tol
        );
        
        ptoc(tag);
    }

    template<Int EQ_COUNT, typename I_ext, typename R_ext, typename C_ext>
    void FarField_parameters(
        cptr<R_ext> kappa,
        const I_ext wave_chunk_count_,
        mptr<R_ext> inc_directions,
        const I_ext wave_chunk_size_,
        mptr<C_ext> C_out,
        const WaveType type,
        cptr<R_ext> eta,
        const R_ext cg_tol,
        const R_ext gmres_tol
    )
    {
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);
        
        std::string tag = ClassName()+"::FarField_parameters<"
            + "," + ToString(EQ_COUNT)
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);
        
        // Implement the bdry to Farfield map. wave ist the std incident wave defined pointwise by exp(i*kappa*<x,d>). A = (1/2) * I - i * kappa * SL + DL
        // phi = A\wave is the bdry potential which will be mapped onto the far field
        
        const C_ext One  = Scalar::One <C_ext>;
        const C_ext Zero = Scalar::Zero<C_ext>;

        const I_ext  n           = static_cast<I_ext>(VertexCount());
        const I_ext  wave_count_ = wave_chunk_count_ * wave_chunk_size_;   
        
        Tensor2<C_ext,I_ext>  inc_coeff ( wave_chunk_count_, 4 );
        Tensor2<C_ext,I_ext>  coeff (  wave_chunk_count_, 4  );
        Tensor2<C_ext,I_ext>  wave  (  n, wave_count_  );     //weak representation of the incident wave
        Tensor2<C_ext,I_ext>  phi   (  n, wave_count_  );

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

        CreateIncidentWave_PL(
            One,   inc_directions,   wave_chunk_size_,
            Zero,  wave.data(),      wave_count_,
            kappa, inc_coeff.data(), wave_count_, wave_chunk_size_,
            type
        );
        
        BoundaryPotential_parameters<EQ_COUNT>(
            kappa, coeff.data(), wave.data(), phi.data(),
            eta, wave_chunk_count_, wave_chunk_size_, cg_tol, gmres_tol
        );

        ApplyFarFieldOperators_PL( 
            One,   phi.data(),   wave_count_,
            Zero,  C_out,        wave_count_,
            kappa, 
            coeff.data(),
            wave_count_,
            wave_chunk_size_
        );
        
        ptoc(tag);
    }

    template<Int EQ_COUNT, typename I_ext, typename R_ext, typename C_ext>
    void Derivative_FF(
        cptr<R_ext> kappa,
        const I_ext wave_chunk_count_,
        cptr<R_ext> inc_directions,
        const I_ext wave_chunk_size_,
        cptr<R_ext> h,
        mptr<C_ext> C_out,
        C_ext * *   pdu_dn, //pdu_dn is the pointer to the Neumann data of the scattered wave
        const WaveType type,
        const R_ext cg_tol,
        const R_ext gmres_tol
    )
    {
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);
        
        std::string tag = ClassName()+"::Derivative_FF<"
            + "," + ToString(EQ_COUNT)
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);
        
        // Implement the action of the derivative of the bdry to Farfield map.
        // inc_wave is a linear combination of the std incident wave defined pointwise by exp(i*kappa*<x,d>) and its normal derivative
        // A := (1/2) * I - i * kappa * SL + DL' for the calculation of du/dn
        // B := (1/2) * I - i * kappa * SL + DL  for the calculation of the Farfield
        // dudn = A\incident_wave is the normal derivative of the solution with inc wave wave
        // phi is the bdry potential for the incident wave dudn *(<h , n>), the solution is the far field to this
        // Formulas follow from Thorsten's book.

        const C_ext One  = static_cast<C_ext>(Complex(1.0f,0.0f));
        const C_ext I    = static_cast<C_ext>(Complex(0.0f,1.0f));
        const C_ext Zero = static_cast<C_ext>(Complex(0.0f,0.0f));

        const I_ext  n           = static_cast<I_ext>(VertexCount());
        const I_ext  wave_count_ = wave_chunk_count_ * wave_chunk_size_;
        

        Tensor2<C_ext,I_ext>  coeff (  wave_chunk_count_, 4  );
        Tensor2<C_ext,I_ext>  boundary_conditions (    n, wave_count_  ); 
        Tensor2<C_ext,I_ext>  boundary_conditions_weak (   n, wave_count_ );
        Tensor1<R_ext,I_ext>  h_n  (   n  );
        Tensor2<C_ext,I_ext>  phi  (   n, wave_count_  ); 

        mptr<C_ext> boundary_conditions_ptr = boundary_conditions.data();

        phi.SetZero();

        if (*pdu_dn == NULL)
        {            
            Tensor2<C_ext,I_ext>  inc_coeff (  4, wave_chunk_count_  );
            Tensor2<C_ext,I_ext>  incident_wave (  n, wave_count_   );  //weak representation of the incident wave 
            
            C_ext* inc_coeff_ptr = inc_coeff.data();

            *pdu_dn = (C_ext*)calloc(wave_count_ * n, sizeof(C_ext)); 

            // create weak representation of the negative incident wave
            for(I_ext i = 0 ; i < wave_chunk_count_ ; i++)
            {
                inc_coeff_ptr[4 * i + 0] = Zero;
                inc_coeff_ptr[4 * i + 1] = C_ext(R_ext(0),-kappa[i]);
                inc_coeff_ptr[4 * i + 2] = One;
                inc_coeff_ptr[4 * i + 3] = Zero;
            }

            CreateIncidentWave_PL( 
                One,   inc_directions,       wave_chunk_size_,
                Zero,  incident_wave.data(), wave_count_,
                kappa, inc_coeff.data(),     wave_count_, wave_chunk_size_, type
            );
            

            DirichletToNeumann<EQ_COUNT>( kappa, incident_wave.data(), *pdu_dn, wave_chunk_count_, wave_chunk_size_, cg_tol, gmres_tol ); 
        }


        DotWithNormals_PL( h, h_n.data(), cg_tol );

            
        // CheckThis
        ParallelDo(
            [boundary_conditions_ptr,&h_n,&pdu_dn]( const I_ext i )
            {   
                cptr<C_ext> a = &(*pdu_dn)[i * EQ_COUNT];
                cptr<R_ext> b = h_n.data();
                mptr<C_ext> c = &boundary_conditions_ptr[i * EQ_COUNT];
                
                for(I_ext j = 0; j < EQ_COUNT; ++j )
                {
                    c[j] = - a[j] * b[i];
                }
            },
            I_ext(n), I_ext(CPU_thread_count)
        );
        
        // apply mass to the boundary conditions to get weak representation
        Mass.Dot(
            One,  boundary_conditions.data(),      wave_count_,
            Zero, boundary_conditions_weak.data(), wave_count_,
            wave_count_
        );

        BoundaryPotential<EQ_COUNT>( kappa, coeff.data(), boundary_conditions_weak.data(), phi.data(), 
                                                            wave_chunk_count_, wave_chunk_size_, cg_tol, gmres_tol);

        ApplyFarFieldOperators_PL( 
            One,  phi.data(),    wave_count_,
            Zero,  C_out,        wave_count_,
            kappa, 
            coeff.data(),
            wave_count_,
            wave_chunk_size_
        );

        ptoc(tag);
    }

    template<Int EQ_COUNT, typename I_ext, typename R_ext, typename C_ext>
    void AdjointDerivative_FF(
        cptr<R_ext> kappa,
        const I_ext wave_chunk_count_,
        cptr<R_ext> inc_directions,
        const I_ext wave_chunk_size_,
        cptr<C_ext> g, R_ext* C_out,
        C_ext** pdu_dn,                 //pdu_dn is the pointer to the Neumann data of the scattered wave
        const WaveType type,
        const R_ext cg_tol,
        const R_ext gmres_tol
    )
    {
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);
        
        // Implement the action of the adjoint to the derivative of the bdry to Farfield map.
        // incident_wave is a linear combination of the std incident wave defined pointwise by exp(i*kappa*<x,d>) and its normal derivative
        // herglotz_wave is a linear combination the herglotz wave with kernel g and its normal derivative
        // A := (1/2) * I - i * kappa * SL + DL'
        // dudn = A\incident_wave is the normal derivative of the solution with inc wave wave
        // anh phi_h = A\herglotz_wave is the normal derivative of the solution with inc wave herglotz_wave

        std::string tag = ClassName()+"::AdjointDerivative_FF<"
            + "," + ToString(EQ_COUNT)
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);

        const C_ext One  = static_cast<C_ext>(Complex(1.0f,0.0f));
        const C_ext I    = static_cast<C_ext>(Complex(0.0f,1.0f));
        const C_ext Zero = static_cast<C_ext>(Complex(0.0f,0.0f));

        const I_ext  n                      = static_cast<I_ext>(VertexCount());
        const I_ext  wave_count_            = wave_chunk_count_ * wave_chunk_size_;


        Tensor2<C_ext,I_ext>  inc_coeff (  wave_chunk_count_, 4  );
        Tensor2<C_ext,I_ext>  herglotz_wave (  n, wave_count_  );     //weak representation of the herglotz wave
        Tensor2<C_ext,I_ext>  dv_dn (  n, wave_count_  );
        Tensor1<C_ext,I_ext>  wave_product (   n   );

        C_ext* inc_coeff_ptr = inc_coeff.data();

        dv_dn.SetZero();

        // create weak representation of the negative incident wave
        for(I_ext i = 0 ; i < wave_chunk_count_ ; i++)
        {
            inc_coeff_ptr[4 * i + 0] = Zero;
            inc_coeff_ptr[4 * i + 1] = C_ext(R_ext(0),-kappa[i]);
            inc_coeff_ptr[4 * i + 2] = One;
            inc_coeff_ptr[4 * i + 3] = Zero;
        }

        if (*pdu_dn == NULL)
        {
            Tensor2<C_ext,I_ext>  incident_wave (  wave_count_, n );  //weak representation of the incident wave 
            
            *pdu_dn           = (C_ext*)calloc(wave_count_ * n, sizeof(C_ext)); 

            
            CreateIncidentWave_PL( One, inc_directions, wave_chunk_size_,
                                Zero, incident_wave.data(), wave_count_,
                                kappa, inc_coeff.data(), wave_count_, wave_chunk_size_,type
                                );
            

            DirichletToNeumann<EQ_COUNT>( kappa, incident_wave.data(), *pdu_dn, wave_chunk_count_, wave_chunk_size_, cg_tol, gmres_tol ); 
        }

        CreateHerglotzWave_PL(One, g, wave_count_,
                            Zero, herglotz_wave.data(), wave_count_,
                            kappa, inc_coeff.data(), wave_count_, wave_chunk_size_
                            );
        
        // solve for the normal derivatives of the near field solutions
        DirichletToNeumann<EQ_COUNT>( kappa, herglotz_wave.data(), dv_dn.data(), wave_chunk_count_, wave_chunk_size_, cg_tol, gmres_tol );
        
        // calculate du_dn .* dv_dn and sum over the leading dimension
        HadamardProduct( *pdu_dn, dv_dn.data(), wave_product.data(), n, wave_count_, true);

        // calculate (-1/wave_count)*Re(du_dn .* dv_dn).*normals
        MultiplyWithNormals_PL(wave_product.data(),C_out,-( 1 /static_cast<R_ext>(wave_count_) ), cg_tol);

        ptoc(tag);
    }

    template<Int EQ_COUNT, typename I_ext, typename R_ext, typename C_ext, typename M_T, typename P_T>
    I_ext GaussNewtonStep(
        cptr<R_ext> kappa,
        const I_ext wave_chunk_count_,
        cptr<R_ext> inc_directions,
        const I_ext wave_chunk_size_,
        M_T & M,
        P_T & P,
        cptr<R_ext> h,
        mptr<R_ext> C_out,
        C_ext** pdu_dn, //pdu_dn is the pointer to the Neumann data of the scattered wave
        const WaveType type,
        const R_ext cg_tol,
        const R_ext gmres_tol_inner,
        const R_ext gmres_tol_outer
    )
    {
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);
        
        std::string tag = ClassName()+"::GaussNewtonStep<"
            + "," + ToString(EQ_COUNT)
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);

        // Calculates a gauss newton step. Note that the metric M has to add the input to the result.
        const I_ext  n           = static_cast<I_ext>(VertexCount());
        const I_ext  m           = static_cast<I_ext>(GetMeasCount());
        const I_ext  wave_count_ = wave_chunk_count_ * wave_chunk_size_;

        GMRES<1,R_ext,Size_T,Side::Left> gmres(3 * n,30,1,CPU_thread_count);
        
        Tensor2<C_ext,I_ext>  DF (  m, wave_count_  );
        Tensor2<R_ext,I_ext>  y_strong (  n, 3  );

        auto A = [&]( const R_ext * x, R_ext *y )
        {   
            Derivative_FF<EQ_COUNT>( 
                kappa, wave_chunk_count_, inc_directions, wave_chunk_size_,
                x, DF.data(), pdu_dn, type, cg_tol, gmres_tol_inner
            );
            
            AdjointDerivative_FF<EQ_COUNT>( 
                kappa, wave_chunk_count_, inc_directions, wave_chunk_size_,
                DF.data(), y_strong.data(), pdu_dn, type, cg_tol, gmres_tol_inner
            );

            Mass.Dot(
                Tools::Scalar::One <R_ext>, y_strong.data(), 3,
                Tools::Scalar::Zero<R_ext>, y,               3,
                3
            );

            M(x,y); // The metric m has to return y + M*x
        };

        zerofy_buffer(C_out, static_cast<Size_T>(n * 3), CPU_thread_count);

        bool succeeded = gmres(A,P,h,1,C_out,1,gmres_tol_outer,3);

        // int iter, res;
        // iter = gmres.IterationCount();
        // res = gmres.RestartCount();

        ptoc(tag);
        
        return static_cast<I_ext>(succeeded);
    }

    //----------------------------------------------------------------------------------------------------------------------------------------------------------------

    template<Int EQ_COUNT, typename I_ext, typename R_ext, typename C_ext>
    void BoundaryPotential(
        cptr<R_ext> kappa,
        mptr<C_ext> coeff,
        mptr<C_ext> wave,
        mptr<C_ext> phi,
        const I_ext wave_chunk_count_,
        const I_ext wave_chunk_size_,
        const R_ext cg_tol,
        const R_ext gmres_tol
    )
    {
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);
        
        std::string tag = ClassName()+"::BoundaryPotential<"
            + "," + ToString(EQ_COUNT)
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);
        
        // No need to copy kappa to eta.

        BoundaryPotential_parameters<EQ_COUNT>(
            kappa, coeff, wave, phi, kappa,
            wave_chunk_count_, wave_chunk_size_, cg_tol, gmres_tol
        );
        
        ptoc(tag);
    }
        
    template<Int EQ_COUNT, typename I_ext, typename R_ext, typename C_ext>
    void BoundaryPotential_parameters(
        cptr<R_ext> kappa,
        mptr<C_ext> coeff,
        mptr<C_ext> wave,
        mptr<C_ext> phi,
        cptr<R_ext> eta,
        const I_ext wave_chunk_count_,
        const I_ext wave_chunk_size_,
        const R_ext cg_tol,
        const R_ext gmres_tol
    )
    {
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);
        
        std::string tag = ClassName()+"::BoundaryPotential_parameters<"
            + "," + ToString(EQ_COUNT)
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);
        
        const C_ext One  = Scalar::One <C_ext>;
        const C_ext Zero = Scalar::Zero<C_ext>;

        const Int  n     = VertexCount();

        const I_ext wave_count_ = wave_chunk_count_ * wave_chunk_size_;

        GMRES<EQ_COUNT,C_ext,Size_T,Side::Left> gmres(n,30,wave_count_,CPU_thread_count);
        
        // setup the mass matrix Preconditionier P:=M^-1. P is also used for transf. into strong form
        
        auto P = [&]( cptr<C_ext> x, mptr<C_ext> y )
        {
            if constexpr ( use_lumped_mass_as_precQ )
            {
                ApplyLumpedMassInverse<EQ_COUNT>(x,wave_count_,y,wave_count_,wave_count_);
            }
            else
            {
                ApplyMassInverse<EQ_COUNT>(x,wave_count_,y,wave_count_,wave_count_);
            }
        };

        // set up the bdry operator and solve
        for(Int i = 0 ; i < wave_chunk_count_ ; i++)
        {
            coeff[4 * i + 0] = static_cast<C_ext>(Complex(0.5f,0.0f));
            coeff[4 * i + 1] = C_ext(R_ext(0),-eta[i]);
            coeff[4 * i + 2] = One;
            coeff[4 * i + 3] = Zero;
        }

        kernel_list list = LoadKernel( kappa, coeff, wave_count_, wave_chunk_size_ );

        auto A = [&]( cptr<C_ext> x, mptr<C_ext> y )
        {
            ApplyBoundaryOperators_PL<true>( One, x, wave_count_, Zero, y, wave_count_ );
        };

        (void)gmres(A,P,wave,wave_count_,phi,wave_count_,gmres_tol,10);

        DestroyKernel(&list);

        ptoc(tag);
    }


    template<Int EQ_COUNT, typename I_ext, typename R_ext, typename C_ext>
    void DirichletToNeumann(
        cptr<R_ext> kappa,
        mptr<C_ext> wave,
        mptr<C_ext> neumann_trace,
        const I_ext wave_chunk_count_,
        const I_ext wave_chunk_size_,
        const R_ext cg_tol,
        const R_ext gmres_tol
    )
    {
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);
        
        std::string tag = ClassName()+"::DirichletToNeumann<"
            + "," + ToString(EQ_COUNT)
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);
        
        const C_ext One  = static_cast<C_ext>(Complex(1.0f,0.0f));
        const C_ext I    = static_cast<C_ext>(Complex(0.0f,1.0f));
        const C_ext Zero = static_cast<C_ext>(Complex(0.0f,0.0f));

        const Int    n   = VertexCount();

        const I_ext wave_count_ = wave_chunk_count_ * wave_chunk_size_;

        C_ext*  coeff    = (C_ext*)malloc(wave_chunk_count_ * 4 * sizeof(C_ext));

        GMRES<EQ_COUNT,C_ext,Size_T,Side::Left> gmres(n,30,wave_count_,CPU_thread_count);

        // setup the mass matrix Preconditionier P:=M^-1. P is also used for transf. into strong form

        auto P = [&]( const C_ext * x, C_ext *y )
        {
            if constexpr ( use_lumped_mass_as_precQ )
            {
                ApplyLumpedMassInverse<EQ_COUNT>(x,wave_count_,y,wave_count_,wave_count_);
            }
            else
            {
                ApplyMassInverse<EQ_COUNT>(x,wave_count_,y,wave_count_,wave_count_);
            }
        };

        // set up the bdry operator and solve
        for(Int i = 0 ; i < wave_chunk_count_ ; i++)
        {
            coeff[4 * i + 0] = static_cast<C_ext>(Complex(0.5f,0.0f));
            coeff[4 * i + 1] = C_ext(R_ext(0),-kappa[i]);
            coeff[4 * i + 2] = Zero;
            coeff[4 * i + 3] = One;
        }

        kernel_list list = LoadKernel(kappa,coeff,wave_count_,wave_chunk_size_);

        auto A = [&]( const C_ext * x, C_ext *y )
        {   
            ApplyBoundaryOperators_PL<true>( One, x, wave_count_, Zero, y, wave_count_ );
        };

        // solve for the normal derivatives of the near field solutions
        (void)gmres(A,P,wave,wave_count_,neumann_trace,wave_count_,gmres_tol,10);

        int iter, res;
        iter = gmres.IterationCount();
        res  = gmres.RestartCount();

        DestroyKernel(&list);

        free(coeff);

        ptoc(tag);
    }

    //------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    // calculate factor * Re(B_in) .* normals
    template<typename R_ext, typename C_ext>
    void MultiplyWithNormals_PL(
        cptr<C_ext> B_in, 
        mptr<R_ext> C_out,
        const R_ext factor,
        const R_ext cg_tol_
    )
    {
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);
        
        std::string tag = ClassName()+"::DotWithNormals_PL<"
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);
        
        const Int m = SimplexCount();
        
        Tensor1<Complex,Int> B      (m);
        Tensor2<Real   ,Int> C      (m, 3);
        Tensor2<R_ext  ,Int> C_weak (m, 3);
        
        // make the input from PL to a PC function
        AvOp.Dot(
            Scalar::One <Complex>, B_in,     1,
            Scalar::Zero<Complex>, B.data(), 1,
            1
        );

        // pointwise multiplication of the STRONG FORM with the normals
        // CheckThis
        ParallelDo(
            [&]( const Int i )
            {
                Real a_i = Re(B[i]) / areas_ptr[i];

                for(Int j = 0; j < 3; ++j )
                {
                    C(i,j) = normals_ptr[i * 4 + j] * a_i;
                }
            },
            m, CPU_thread_count
        );

        // retransf. from PC to PL
        // There is an overload of Dot for Tensor2.
        AvOpTransp.Dot(
            factor,              C.data(),
            Scalar::Zero<R_ext>, C_weak.data()
        );

        cg_tol = static_cast<Real>(cg_tol_);
        ApplyMassInverse<3>( C_weak.data(), 3, C_out, 3, 3 );
        
        ptoc(tag);
    }


    // calculate <B_in , normals>
    template<typename R_ext>
    void DotWithNormals_PL( 
        cptr<R_ext> B_in,
        mptr<R_ext> C_out,
        const R_ext cg_tol_
    )
    {
        ASSERT_REAL(R_ext);
        
        std::string tag = ClassName()+"::DotWithNormals_PL<"
            + "," + TypeName<R_ext>
            + ">";
        
        ptic(tag);
        
        const Int   m = SimplexCount();
        const Int   n = VertexCount();

        Real*       B = (Real*)malloc( 3 * m * sizeof(Real));
        Real*       C = (Real*)calloc( m, sizeof(Real));        
        R_ext* C_weak = (R_ext*)malloc( n * sizeof(R_ext));
        
        // make the input from PL to a PC function
        AvOp.Dot<3>( 
            Scalar::One <R_ext>, B_in,  3,
            Scalar::Zero<R_ext>, B,     3,
            3
        );

        // pointwise multiplication of the STRONG FORM with the normals
        // CheckThis
        ParallelDo(
            [=]( const Int i )
            {
                Real a_i = Inv<Real>(areas_ptr[i]);

                for(Int j = 0; j < 3; ++j )
                {
                    C[i] += normals_ptr[i * 4 + j] * B[i * 3 + j] * a_i;
                }
            },
            m, CPU_thread_count
        );

        // retransf. from PC to PL
        AvOpTransp.Dot(
            Scalar::One <R_ext>, C,      1,
            Scalar::Zero<R_ext>, C_weak, 1,
            1
        );
        
        cg_tol = static_cast<Real>(cg_tol_);
        ApplyMassInverse<1>(C_weak,1,C_out,1,1);

        free(B);
        free(C);
        free(C_weak);
        
        ptoc(tag);
    }
