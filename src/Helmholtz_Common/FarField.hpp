// The functions here interact with the outside world.
// Thus, the types for real, complex and integer numbers may deviate from the internally used ones.
// Therefore, most functions are templated on these types.

// WC = wave count at compile time.
// WC > 0 means the number is known. Compiler will try to use compile time optimizations wherever possible.
// WC = 0 means that the number of waves is computed from wave_chunk_count_ * wave_chunk_size_ at runtime. Certain optimizations won't be available.

public:
    
    template<Int WC, typename I_ext, typename R_ext, typename C_ext>
    void FarField(
        cptr<R_ext> kappa_,
        const I_ext wave_chunk_count_,
        mptr<R_ext> inc_directions,
        const I_ext wave_chunk_size_,
        mptr<C_ext> C_out,
        const WaveType type,
        const R_ext cg_tol,
        const R_ext gmres_tol
    )
    {
        ASSERT_INT(I_ext);
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);
        
        std::string tag = ClassName()+"::FarField<" + ToString(WC)
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);
        
        // // Implement the bdry to Farfield map. 
        // wave ist the std incident wave defined pointwise by exp(i*kappa*<x,d>).
        // A = (1/2) * I - i * kappa * SL + DL
        // phi = A\wave is the bdry potential which will be mapped onto the far field.
        
        FarField_parameters<WC>(
            kappa_, 
            int_cast<Int>(wave_chunk_count_),
            inc_directions,
            int_cast<Int>(wave_chunk_size_),
            C_out,
            type,
            kappa_,  // No need for copying kappa_ to eta.
            cg_tol,
            gmres_tol
        );
        
        ptoc(tag);
    }

public:

    template<Int WC, typename I_ext, typename R_ext, typename C_ext>
    void FarField_parameters(
        cptr<R_ext> kappa_,
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
        ASSERT_INT(I_ext);
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);
        
        std::string tag = ClassName()+"::FarField_parameters<"
            + "," + ToString(WC)
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);
        
        // Implement the bdry to Farfield map. wave ist the std incident wave defined pointwise by exp(i*kappa*<x,d>). A = (1/2) * I - i * kappa * SL + DL
        // phi = A\wave is the bdry potential which will be mapped onto the far field
        
        constexpr C_ext One  = Scalar::One <C_ext>;
        constexpr C_ext Zero = Scalar::Zero<C_ext>;
        
        const Int n   = VertexCount();
        const Int wcc = int_cast<Int>(wave_chunk_count_);
        const Int wcs = int_cast<Int>(wave_chunk_size_ );
        const Int wc  = wcc * wcs;
        
        Tensor2<C_ext,Int>  inc_coeff ( wcc, 4  );
        Tensor2<C_ext,Int>  coeff     ( wcc, 4  );
        Tensor2<C_ext,Int>  wave      ( n,   wc );     //weak representation of the incident wave
        Tensor2<C_ext,Int>  phi       ( n,   wc );
        
        phi.SetZero( CPU_thread_count );

        // create weak representation of the negative incident wave
        for( Int i = 0 ; i < wcc ; i++ )
        {
            inc_coeff(i,0) = Zero;
            inc_coeff(i,1) = -One;
            inc_coeff(i,2) = Zero;
            inc_coeff(i,3) = Zero;
        }

        CreateIncidentWave_PL(
            One,    inc_directions, wcs,
            Zero,   wave.data(),    wc,
            kappa_, inc_coeff.data(), wc, wcs, type
        );
        
        BoundaryPotential_parameters<WC>(
            kappa_, coeff.data(), wave.data(), phi.data(),
            eta, wcc, wcs, cg_tol, gmres_tol
        );

        ApplyFarFieldOperators_PL( 
            One,   phi.data(), wc,
            Zero,  C_out,      wc,
            kappa_, coeff.data(), wc, wcs
        );
        
        ptoc(tag);
    }

public:

    template<Int WC, typename I_ext, typename R_ext, typename C_ext>
    void Derivative_FF(
        cptr<R_ext> kappa_,
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
        ASSERT_INT(I_ext);
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);
        
        std::string tag = ClassName()+"::Derivative_FF<"
            + "," + ToString(WC)
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

        constexpr C_ext One  = Scalar::One <C_ext>;
        constexpr C_ext I    = Scalar::I   <C_ext>;
        constexpr C_ext Zero = Scalar::Zero<C_ext>;

        const Int n   = VertexCount();
        const Int wcc = int_cast<Int>(wave_chunk_count_);
        const Int wcs = int_cast<Int>(wave_chunk_size_ );
        const Int wc  = wcc * wcs;
        
        Tensor2<C_ext,Int>  coeff                    ( wcc, 4  );
        Tensor2<C_ext,Int>  boundary_conditions      ( n,   wc );
        Tensor2<C_ext,Int>  boundary_conditions_weak ( n,   wc );
        Tensor2<C_ext,Int>  phi                      ( n,   wc );
        
        Tensor1<R_ext,Int>  h_n                      ( n );
        
        phi.SetZero( CPU_thread_count );

        if (*pdu_dn == NULL)
        {            
            Tensor2<C_ext,I_ext>  inc_coeff     ( 4, wcc );
            Tensor2<C_ext,I_ext>  incident_wave ( n, wc  );  //weak representation of the incident wave

            *pdu_dn = (C_ext*)calloc(n * wc, sizeof(C_ext));

            // create weak representation of the negative incident wave
            for(I_ext i = 0 ; i < wcc ; i++)
            {
                inc_coeff(i,0) = Zero;
                inc_coeff(i,1) = C_ext(R_ext(0),-kappa_[i]);
                inc_coeff(i,2) = One;
                inc_coeff(i,3) = Zero;
            }

            CreateIncidentWave_PL( 
                One,    inc_directions,       wcs,
                Zero,   incident_wave.data(), wc,
                kappa_, inc_coeff.data(),     wc, wcs, type
            );
            
            DirichletToNeumann<WC>( kappa_, incident_wave.data(), *pdu_dn, wcc, wcs, cg_tol, gmres_tol );
        }

        DotWithNormals_PL( h, h_n.data(), cg_tol );
            
        mptr<C_ext> boundary_conditions_ptr = boundary_conditions.data();
        
        // CheckThis
        ParallelDo(
            [boundary_conditions_ptr,&h_n,&pdu_dn]( const Int i )
            {
                cptr<C_ext> a = &(*pdu_dn)[WC * i];
                mptr<C_ext> c = &boundary_conditions_ptr[WC * i];
                
                const R_ext b = -h_n[i];
                
                // TODO: This does not work for WC == VarSize.
                for( Int j = 0; j < WC; ++j )
                {
                    c[j] = a[j] * b;
                }
                
                
//                combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Zero,NRHS>(
//                    -h_n[i],             &(*pdu_dn)[WC * i],
//                    Scalar::Zero<C_ext>, &boundary_conditions_ptr[WC * i],
//                    nrhs
//                );
            },
            n, CPU_thread_count
        );
        
        // apply mass to the boundary conditions to get weak representation
        Mass.Dot(
            One,  boundary_conditions.data(),      wc,
            Zero, boundary_conditions_weak.data(), wc,
            wc
        );

        BoundaryPotential<WC>( 
            kappa_, coeff.data(), boundary_conditions_weak.data(), phi.data(),
            wcc, wcs, cg_tol, gmres_tol
        );

        ApplyFarFieldOperators_PL( 
            One,   phi.data(),    wc,
            Zero,  C_out,         wc,
            kappa_, coeff.data(), wc, wcs
        );

        ptoc(tag);
    }

public:

    template<Int WC, typename I_ext, typename R_ext, typename C_ext>
    void AdjointDerivative_FF(
        cptr<R_ext> kappa_,
        const I_ext wave_chunk_count_,
        cptr<R_ext> inc_directions,
        const I_ext wave_chunk_size_,
        cptr<C_ext> g, 
        mptr<R_ext> C_out,
        C_ext * * pdu_dn,                 //pdu_dn is the pointer to the Neumann data of the scattered wave
        const WaveType type,
        const R_ext cg_tol,
        const R_ext gmres_tol
    )
    {
        ASSERT_INT(I_ext);
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);
        
        // Implement the action of the adjoint to the derivative of the bdry to Farfield map.
        // incident_wave is a linear combination of the std incident wave defined pointwise by exp(i*kappa*<x,d>) and its normal derivative
        // herglotz_wave is a linear combination the herglotz wave with kernel g and its normal derivative
        // A := (1/2) * I - i * kappa * SL + DL'
        // dudn = A\incident_wave is the normal derivative of the solution with inc wave wave
        // anh phi_h = A\herglotz_wave is the normal derivative of the solution with inc wave herglotz_wave

        std::string tag = ClassName()+"::AdjointDerivative_FF<"
            + "," + ToString(WC)
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);

        constexpr C_ext One  = Scalar::One <C_ext>;
        constexpr C_ext I    = Scalar::I   <C_ext>;
        constexpr C_ext Zero = Scalar::Zero<C_ext>;

        const Int n   = VertexCount();
        const Int wcc = int_cast<Int>(wave_chunk_count_);
        const Int wcs = int_cast<Int>(wave_chunk_size_ );
        const Int wc  = wcc * wcs;
        
        Tensor2<C_ext,Int>  inc_coeff     ( wcc, 4  );
        Tensor2<C_ext,Int>  herglotz_wave ( n,   wc );     //weak representation of the herglotz wave
        Tensor2<C_ext,Int>  dv_dn         ( n,   wc );

        Tensor1<C_ext,Int>  wave_product  ( n );

        wave_product.SetZero();
        
        // Create weak representation of the negative incident wave.
        for( Int i = 0 ; i < wcc ; i++)
        {
            inc_coeff(i,0) = Zero;
            inc_coeff(i,1) = C_ext(R_ext(0),-kappa_[i]);
            inc_coeff(i,2) = One;
            inc_coeff(i,3) = Zero;
        }

        if (*pdu_dn == NULL)
        {
            Tensor2<C_ext,Int> incident_wave ( n, wc );  //weak representation of the incident wave
            
            *pdu_dn = (C_ext*)calloc(n * wc, sizeof(C_ext));

            CreateIncidentWave_PL( 
                One,  inc_directions,       wcs,
                Zero, incident_wave.data(), wc,
                kappa_, inc_coeff.data(), wc, wcs,type
            );
            
            DirichletToNeumann<WC>( kappa_, incident_wave.data(), *pdu_dn, wcc, wcs, cg_tol, gmres_tol );
        }

        CreateHerglotzWave_PL(
            One,  g,                    wc,
            Zero, herglotz_wave.data(), wc,
            kappa_, inc_coeff.data(), wc, wcs
        );
        
        // solve for the normal derivatives of the near field solutions
        DirichletToNeumann<WC>( kappa_, herglotz_wave.data(), dv_dn.data(), wcc, wcs, cg_tol, gmres_tol );
        
        // calculate du_dn .* dv_dn and sum over the leading dimension
        HadamardProduct( *pdu_dn, dv_dn.data(), wave_product.data(), n, wc, true );

        // calculate (-1/wave_count)*Re(du_dn .* dv_dn).*normals
        MultiplyWithNormals_PL(wave_product.data(),C_out,-Inv<R_ext>(wc), cg_tol);

        ptoc(tag);
    }

public:

    template<Int WC, typename I_ext, typename R_ext, typename C_ext, typename M_T, typename P_T>
    I_ext GaussNewtonStep(
        cptr<R_ext> kappa_,
        const I_ext wave_chunk_count_,
        cptr<R_ext> inc_directions,
        const I_ext wave_chunk_size_,
        M_T & M,
        P_T & P,
        cptr<R_ext> h,
        mptr<R_ext> C_out,
        C_ext * * pdu_dn, //pdu_dn is the pointer to the Neumann data of the scattered wave
        const WaveType type,
        const R_ext cg_tol,
        const R_ext gmres_tol_inner,
        const R_ext gmres_tol_outer
    )
    {
        ASSERT_INT(I_ext);
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);
        
        std::string tag = ClassName()+"::GaussNewtonStep<"
            + "," + ToString(WC)
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);

        // Calculates a Gauss-Newton step. Note that the metric M has to _add_ the input to the result.
        
        const Int n   = VertexCount();
        const Int m   = GetMeasCount();
        const Int wcc = int_cast<Int>(wave_chunk_count_);
        const Int wcs = int_cast<Int>(wave_chunk_size_ );
        const Int wc  = wcc * wcs;

        GMRES<1,R_ext,Size_T,Side::Left> gmres(n * Int(3),30,1,CPU_thread_count);
        
        Tensor2<C_ext,Int>  DF       ( m, wc     );
        Tensor2<R_ext,Int>  y_strong ( n, Int(3) );

        auto A = [&]( cptr<R_ext> x, mptr<R_ext> y )
        {
            Derivative_FF<WC>( 
                kappa_, wcc, inc_directions, wcs,
                x, DF.data(), pdu_dn, type, cg_tol, gmres_tol_inner
            );
            
            AdjointDerivative_FF<WC>( 
                kappa_, wcc, inc_directions, wcs,
                DF.data(), y_strong.data(), pdu_dn, type, cg_tol, gmres_tol_inner
            );

            Mass.Dot<3>(
                Tools::Scalar::One <R_ext>, y_strong.data(), 3,
                Tools::Scalar::Zero<R_ext>, y,               3,
                3
            );

            M(x,y); // The metric m has to return y + M*x
        };

        zerofy_buffer(C_out, static_cast<Size_T>(n * 3), CPU_thread_count);

        const Int succeeded = gmres(A,P,h,1,C_out,1,gmres_tol_outer,3);

        ptoc(tag);
        
        return static_cast<I_ext>(succeeded);
    }

    //----------------------------------------------------------------------------------------------------------------------------------------------------------------

public:

    template<Int WC, typename I_ext, typename R_ext, typename C_ext>
    void BoundaryPotential(
        cptr<R_ext> kappa_,
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
            + "," + ToString(WC)
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);
        
        const Int wcc = int_cast<Int>(wave_chunk_count_);
        const Int wcs = int_cast<Int>(wave_chunk_size_ );
        
        // No need to copy kappa to eta.

        BoundaryPotential_parameters<WC>(
            kappa_, coeff, wave, phi, kappa_, wcc, wcs, cg_tol, gmres_tol
        );
        
        ptoc(tag);
    }

private:

// Henrik: I think this function is only meant to be called internally.
// Therefore I made I it `private`.
        
    template<Int WC, typename I_ext, typename R_ext, typename C_ext>
    void BoundaryPotential_parameters(
        cptr<R_ext> kappa_,
        mptr<C_ext> coeff_,
        mptr<C_ext> wave,
        mptr<C_ext> phi,
        cptr<R_ext> eta,
        const I_ext wave_chunk_count_,
        const I_ext wave_chunk_size_,
        const R_ext cg_tol,
        const R_ext gmres_tol
    )
    {
        ASSERT_INT(I_ext);
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);
        
        std::string tag = ClassName()+"::BoundaryPotential_parameters<"
            + "," + ToString(WC)
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);
        
        constexpr C_ext One  = Scalar::One <C_ext>;
        constexpr C_ext Zero = Scalar::Zero<C_ext>;

        const Int n   = VertexCount();
        const Int wcc = int_cast<Int>(wave_chunk_count_);
        const Int wcs = int_cast<Int>(wave_chunk_size_ );
        const Int wc  = wcc * wcs;

        GMRES<WC,C_ext,Size_T,Side::Left> gmres(n,30,wc,CPU_thread_count);
        
        // Setup the mass matrix Preconditionier P:=M^-1.
        
        // P is also used for transf. into strong form.
        // Henrik is it?
        
        auto P = [&]( cptr<C_ext> x, mptr<C_ext> y )
        {
            ApplyLumpedMassInverse<WC>(x,wc,y,wc,wc);
        };

        // set up the bdry operator and solve
        for( Int i = 0 ; i < wcc; i++ )
        {
            coeff_[4 * i + 0] = static_cast<C_ext>(Complex(0.5f,0.0f));
            coeff_[4 * i + 1] = C_ext(R_ext(0),-eta[i]);
            coeff_[4 * i + 2] = One;
            coeff_[4 * i + 3] = Zero;
        }

        LoadBoundaryOperators_PL(kappa_,coeff_,wc,wcs);
        
        auto A = [&]( cptr<C_ext> x, mptr<C_ext> y )
        {
            ApplyBoundaryOperators_PL( One, x, wc, Zero, y, wc );
        };

        (void)gmres(A,P,wave,wc,phi,wc,gmres_tol,10);

        UnloadBoundaryOperators_PL();

        ptoc(tag);
    }

public:

// Henrik: Is this function ever called at all?

    template<Int WC, typename I_ext, typename R_ext, typename C_ext>
    void DirichletToNeumann(
        cptr<R_ext> kappa_,
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
            + "," + ToString(WC)
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);
        
        constexpr C_ext One  = Scalar::One <C_ext>;
        constexpr C_ext I    = Scalar::I   <C_ext>;
        constexpr C_ext Zero = Scalar::Zero<C_ext>;

        const Int n   = VertexCount();
        const Int wcc = int_cast<Int>(wave_chunk_count_);
        const Int wcs = int_cast<Int>(wave_chunk_size_ );
        const Int wc  = wcc * wcs;

        Tensor2<C_ext,Int> c_ ( wcc, 4);

        GMRES<WC,C_ext,Size_T,Side::Left> gmres(n,30,wc,CPU_thread_count);

        // Setup the mass matrix Preconditionier P:=M^-1.
        // P is also used for transf. into strong form
        // Henrik: Is it?

        auto P = [&]( cptr<C_ext> x, mptr<C_ext> y )
        {
            ApplyLumpedMassInverse<WC>(x,wc,y,wc,wc);
        };

        // set up the bdry operator and solve
        for( Int i = 0 ; i < wcc ; i++ )
        {
            c_(i,0) = static_cast<C_ext>(Complex(0.5f,0.0f));
            c_(i,1) = C_ext(R_ext(0),-kappa_[i]);
            c_(i,2) = Zero;
            c_(i,3) = One;
        }

        LoadBoundaryOperators_PL(kappa_,c_.data(),wc,wcs);

        auto A = [&]( const C_ext * x, C_ext *y )
        {   
            ApplyBoundaryOperators_PL( One, x, wc, Zero, y, wc );
        };

        // solve for the normal derivatives of the near field solutions
        (void)gmres(A,P,wave,wc,neumann_trace,wc,gmres_tol,10);

        UnloadBoundaryOperators_PL();

        ptoc(tag);
    }

    //------------------------------------------------------------------------------------------------------------------------------------------------------------------------

private:

// Henrik: I think this function is only meant to be called internally.
// Therefore I made I it `private`.

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
        
        const Int n = VertexCount();
        const Int m = SimplexCount();
        
        Tensor1<Complex,Int> B      (m);
        Tensor2<Real   ,Int> C      (m, 3);
        Tensor2<Real   ,Int> C_weak (n, 3);
        
        // Convert the input from PL to a PC function.
        // Also change precision to internal one.
        AvOp.Dot<1>(
            Scalar::One <Complex>, B_in,     1,
            Scalar::Zero<Complex>, B.data(), 1,
            1
        );
        
        // From here on we use internal precision (float).

        // Pointwise multiplication of the STRONG FORM with the normals.
        // CheckThis
        ParallelDo(
            [&]( const Int i )
            {
                const Real mul = Re(B[i]) / areas_ptr[i];

                for(Int j = 0; j < 3; ++j )
                {
                    C(i,j) = mul * normals_ptr[i * 4 + j];
                }
            },
            m, CPU_thread_count
        );
        
        // Convert from PC function to PL function.
        AvOpTransp.Dot<3>(
            static_cast<Real>(factor), C.data()     , 3,
            Scalar::Zero<Real>,        C_weak.data(), 3,
            3
        );

        // Set the tolerance parameter for ApplyMassInverse.
        cg_tol = static_cast<Real>(cg_tol_);
        
        // The solve here still uses internal precision (float).
        // On return the result is converted to external precision.
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
        
        const Int m = SimplexCount();
        const Int n = VertexCount();

        Tensor2<Real,Int> B      ( m, 3 );
        Tensor1<Real,Int> C      ( m );
        Tensor1<Real,Int> C_weak ( n );
        
        // Convert the input from PL to a PC function.
        // Also change precision to internal one.
        AvOp.Dot<3>(
            Scalar::One <R_ext>, B_in,     3,
            Scalar::Zero<R_ext>, B.data(), 3,
            3
        );
        
        // From here on we use internal precision (float).

        // TODO: Allocating C_weak might be obsolete, because we can use C_out.
        
        // Pointwise multiplication of the STRONG FORM with the normals.
        // CheckThis
        ParallelDo(
            [&C,&B,this]( const Int i )
            {
                Real factor = Inv<Real>(areas_ptr[i]);

                Real sum = 0;
                
                for(Int j = 0; j < 3; ++j )
                {
                    sum += normals_ptr[i * 4 + j] * B(i,j) * factor;
                }
                
                C[i] = sum;
            },
            m, CPU_thread_count
        );

        
        cg_tol = static_cast<Real>(cg_tol_);
        
        // Convert from PC function to PL function.
        AvOpTransp.Dot<1>(
            Scalar::One <R_ext>, C.data(),      1,
            Scalar::Zero<R_ext>, C_weak.data(), 1,
            1
        );
        
        // The solve here still uses internal precision (float).
        // On return the result is converted to external precision.
        ApplyMassInverse<1>( C_weak.data(), 1, C_out, 1, 1 );
        
        ptoc(tag);
    }
