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
        mptr<C_ext> Y_out,
        const WaveType type,
        const R_ext cg_tol,
        const R_ext gmres_tol
    )
    {
        CheckInteger<I_ext>();
        CheckScalars<R_ext,C_ext>();
        
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
            Y_out,
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
        mptr<C_ext> Y_out,
        const WaveType type,
        cptr<R_ext> eta,
        const R_ext cg_tol,
        const R_ext gmres_tol
    )
    {
        CheckInteger<I_ext>();
        CheckScalars<R_ext,C_ext>();
        
        // Implement the bdry to Farfield map. wave ist the std incident wave defined pointwise by exp(i*kappa*<x,d>). A = (1/2) * I - i * kappa * SL + DL
        // phi = A \ wave is the bdry potential which will be mapped onto the far field
        
        const Int n   = VertexCount();
        const Int wcc = int_cast<Int>(wave_chunk_count_);
        const Int wcs = int_cast<Int>(wave_chunk_size_ );
        const Int wc  = wcc * wcs;
        
        Tensor2<C_ext,Int>  inc_coeff ( wcc, 4  );
        Tensor2<C_ext,Int>  coeff     ( wcc, 4  );
        Tensor2<C_ext,Int>  wave      ( n,   wc );     //weak representation of the incident wave
        Tensor2<C_ext,Int>  phi       ( n,   wc );

        // Create weak representation of the negative incident wave.
        for( Int i = 0 ; i < wcc ; i++ )
        {
            inc_coeff(i,0) =  C_ext(0);
            inc_coeff(i,1) = -C_ext(1);
            inc_coeff(i,2) =  C_ext(0);
            inc_coeff(i,3) =  C_ext(0);
        }

        CreateIncidentWave_PL(
            C_ext(1), inc_directions, wcs,
            C_ext(0), wave.data(),    wc,
            kappa_, inc_coeff.data(), wc, wcs, type
        );
        
        BoundaryPotential_parameters<WC>(
            kappa_, coeff.data(), wave.data(), phi.data(),
            eta, wcc, wcs, cg_tol, gmres_tol
        );

        ApplyFarFieldOperators_PL( 
            C_ext(1), phi.data(), wc,
            C_ext(0), Y_out,      wc,
            kappa_, coeff.data(), wc, wcs
        );
    }

public:

    template<Int WC, typename I_ext, typename R_ext, typename C_ext>
    void Derivative_FF(
        cptr<R_ext> kappa_,
        const I_ext wave_chunk_count_,
        cptr<R_ext> inc_directions,
        const I_ext wave_chunk_size_,
        cptr<R_ext> X_in,
        mptr<C_ext> Y_out,
        C_ext * &   du_dn, //du_dn is the Neumann data of the scattered wave
        const WaveType type,
        const R_ext cg_tol,
        const R_ext gmres_tol
    )
    {
        CheckInteger<I_ext>();
        CheckScalars<R_ext,C_ext>();
        
        std::string tag = ClassName()+"::Derivative_FF<"
            + "," + ToString(WC)
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);
        
        // Implement the action of the derivative of the bdry to Farfield map.
        // `inc_wave` is a linear combination of the std incident wave defined pointwise by exp(i*kappa*<x,d>) and its normal derivative
        // A := (1/2) * I - i * kappa * SL + DL' for the calculation of du/dn
        // B := (1/2) * I - i * kappa * SL + DL  for the calculation of the Farfield
        // `du_dn` = A \ `inc_wave` is the normal derivative of the solution with inc wave `wave`
        // `phi` is the bdry potential for the incident wave `du_dn` *(<X_in , n>), the solution is the far field to this
        // Formulas follow from Thorsten's book.

        const Int n   = VertexCount();
        const Int wcc = int_cast<Int>(wave_chunk_count_);
        const Int wcs = int_cast<Int>(wave_chunk_size_ );
        const Int wc  = wcc * wcs;
        
        Tensor2<C_ext,Int>  coeff         ( wcc, 4  );
        Tensor2<C_ext,Int>  bdr_cond_buf  ( n,   wc );
        Tensor2<C_ext,Int>  bdr_cond_weak ( n,   wc );
        Tensor2<C_ext,Int>  phi           ( n,   wc );
        
        Tensor1<R_ext,Int>  X_n           ( n );
        
        if( du_dn == nullptr )
        {
            logprint( tag + ": du_dn == nullptr. Allocating and computing du_dn.");
            
            Tensor2<C_ext,Int> inc_coeff ( wcc, 4  );
            Tensor2<C_ext,Int> inc_wave  ( n,   wc );  //weak representation of the incident wave
            
            du_dn = (C_ext*)calloc(n * wc, sizeof(C_ext));

            // create weak representation of the negative incident wave
            for(Int i = 0 ; i < wcc ; i++)
            {
                inc_coeff(i,0) = C_ext(0);
                inc_coeff(i,1) = C_ext( R_ext(0), -kappa_[i] );
                inc_coeff(i,2) = C_ext(1);
                inc_coeff(i,3) = C_ext(0);
            }

            CreateIncidentWave_PL(
                C_ext(1), inc_directions,  wcs,
                C_ext(0), inc_wave.data(), wc,
                kappa_, inc_coeff.data(),  wc, wcs, type
            );
            
            DirichletToNeumann<WC>( kappa_, inc_wave.data(), du_dn, wcc, wcs, cg_tol, gmres_tol );
            
        }

        DotWithNormals_PL( X_in, X_n.data(), cg_tol );

        mptr<C_ext> bdr_cond = bdr_cond_buf.data();
        
        // CheckThis
        ParallelDo(
            [=,this,&X_n]( const Int i )
            {
                // bdr_cond[wc * i] = -X_n[i] * du_dn[wc * i]
                
                combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Zero,WC>(
                    -X_n[i],             &du_dn   [wc * i],
                    Scalar::Zero<C_ext>, &bdr_cond[wc * i],
                    wc
                );
            },
            n, CPU_thread_count
        );

        // apply mass to the boundary conditions to get weak representation
        Mass.Dot<WC>(
            Scalar::One <C_ext>, bdr_cond_buf.data(),  wc,
            Scalar::Zero<C_ext>, bdr_cond_weak.data(), wc,
            wc
        );
        
        BoundaryPotential<WC>(
            kappa_, coeff.data(), bdr_cond_weak.data(), phi.data(),
            wcc, wcs, cg_tol, gmres_tol
        );
        
        ApplyFarFieldOperators_PL(
            C_ext(0), phi.data(), wc,
            C_ext(1), Y_out,      wc,
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
        mptr<R_ext> Y_out,
        C_ext * & du_dn, //du_dn is the Neumann data of the scattered wave
        const WaveType type,
        const R_ext cg_tol,
        const R_ext gmres_tol
    )
    {
        CheckInteger<I_ext>();
        CheckScalars<R_ext,C_ext>();
        
        // Implement the action of the adjoint to the derivative of the bdry to Farfield map.
        // `inc_wave` is a linear combination of the standard incident wave defined pointwise by exp(i*kappa*<x,d>) and its normal derivative
        // `herglotz_wave` is a linear combination the Herglotz wave with kernel `g` and its normal derivative
        // A := (1/2) * I - i * kappa * SL + DL'
        // `du_dn` = A \ `inc_wave` is the normal derivative of the solution with incident wave `wave`
        // `phi_h = A \ `herglotz_wave` is the normal derivative of the solution with inc wave `herglotz_wave`

        std::string tag = ClassName()+"::AdjointDerivative_FF<"
            + "," + ToString(WC)
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);

        const Int n   = VertexCount();
        const Int wcc = int_cast<Int>(wave_chunk_count_);
        const Int wcs = int_cast<Int>(wave_chunk_size_ );
        const Int wc  = wcc * wcs;
        
        Tensor2<C_ext,Int>  herglotz_wave ( n, wc ); //weak representation of the herglotz wave
        Tensor2<C_ext,Int>  dv_dn_buf     ( n, wc );
        
        mptr<C_ext> dv_dn = dv_dn_buf.data();
        
        Tensor1<C_ext,Int> wave_product  ( n );

        // Create weak representation of the negative incident wave.
        Tensor2<C_ext,Int> inc_coeff ( wcc, 4 );
        
        for( Int i = 0 ; i < wcc ; i++)
        {
            inc_coeff(i,0) = C_ext(0);
            inc_coeff(i,1) = C_ext(R_ext(0),-kappa_[i]);
            inc_coeff(i,2) = C_ext(1);
            inc_coeff(i,3) = C_ext(0);
        }

        if( du_dn == nullptr )
        {
            Tensor2<C_ext,Int> inc_wave ( n, wc );  //weak representation of the incident wave
            
            du_dn = (C_ext*)calloc(n * wc, sizeof(C_ext));
            
            CreateIncidentWave_PL(
                C_ext(1), inc_directions,  wcs,
                C_ext(0), inc_wave.data(), wc,
                kappa_, inc_coeff.data(),  wc, wcs, type
            );
            
            DirichletToNeumann<WC>( kappa_, inc_wave.data(), du_dn, wcc, wcs, cg_tol, gmres_tol );
        }

        CreateHerglotzWave_PL(
            C_ext(1), g,                    wc,
            C_ext(0), herglotz_wave.data(), wc,
            kappa_, inc_coeff.data(), wc, wcs
        );
        
        // Solve for the normal derivatives of the near field solutions.
        DirichletToNeumann<WC>( kappa_, herglotz_wave.data(), dv_dn, wcc, wcs, cg_tol, gmres_tol );
        
        // Calculate du_dn .* dv_dn and sum over the leading dimension.
        HadamardProduct( du_dn, dv_dn, wave_product.data(), n, wc, true );

        // Calculate (-1/wave_count)*Re(du_dn .* dv_dn).*normals.
        MultiplyWithNormals_PL( wave_product.data(), Y_out, -Inv<R_ext>(wc), cg_tol );

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
        cptr<R_ext> X_in,
        mptr<R_ext> Y_out,
        C_ext * & du_dn, //du_dn is the Neumann data of the scattered wave
        const WaveType type,
        const R_ext cg_tol,
        const R_ext gmres_tol_inner,
        const R_ext gmres_tol_outer
    )
    {
        CheckInteger<I_ext>();
        CheckScalars<R_ext,C_ext>();
        
        std::string tag = ClassName()+"::GaussNewtonStep<"
            + "," + ToString(WC)
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);

        // Calculates a Gauss-Newton step. Note that the metric M has to _add_ the input to the result.
        
        constexpr Int DIM = 3; // Dimension of the ambient space.
        constexpr Int ldX = DIM;
        constexpr Int ldY = DIM;
        
        const Int n   = VertexCount();
        const Int m   = GetMeasCount();
        const Int wcc = int_cast<Int>(wave_chunk_count_);
        const Int wcs = int_cast<Int>(wave_chunk_size_ );
        const Int wc  = wcc * wcs;
        
        // The two optional booleans at the end of the template silence some messages.
        //                                  |     |
        //                                  v     v
        GMRES<DIM,R_ext,Size_T,Side::Left,false,false> gmres(
            n, gmres_max_iter, DIM, CPU_thread_count /*, true*/ );
        //                      ^                          ^
        //                      |                          |
        //               This argument is new.    This would activate use of initial guess.

        
        // A, M and P are matrices of size n x n.
        // They are applied to matrices of size n x DIM
        
        Tensor2<C_ext,Int>  DF       ( m, wc  );
        Tensor2<R_ext,Int>  y_strong ( n, DIM );

        auto A = [&]( cptr<R_ext> x, mptr<R_ext> y )
        {
            Derivative_FF<WC>( 
                kappa_, wcc, inc_directions, wcs,
                x, DF.data(), du_dn, type, cg_tol, gmres_tol_inner
            );
            
            AdjointDerivative_FF<WC>( 
                kappa_, wcc, inc_directions, wcs,
                DF.data(), y_strong.data(), du_dn, type, cg_tol, gmres_tol_inner
            );

            Mass.Dot<DIM>(
                Tools::Scalar::One <R_ext>, y_strong.data(), DIM,
                Tools::Scalar::Zero<R_ext>, y,               DIM,
                DIM
            );

            M(x,y); // The metric m has to return y + M*x
        };
        
        // Evaluating Y_out = R_ext(1) * A^{-1} . X_in + R_ext(0) * Y_out
        const Int succeeded = gmres(A,P,
            Scalar::One <C_ext>, X_in,  ldX,
            Scalar::Zero<C_ext>, Y_out, ldY,
            gmres_tol_outer, gmres_max_restarts
        );

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
        CheckInteger<I_ext>();
        CheckScalars<R_ext,C_ext>();
        
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
        CheckInteger<I_ext>();
        CheckScalars<R_ext,C_ext>();
        
        const Int n   = VertexCount();
        const Int wcc = int_cast<Int>(wave_chunk_count_);
        const Int wcs = int_cast<Int>(wave_chunk_size_ );
        const Int wc  = wcc * wcs;

        // The two boolean at the end of the template silence some messages.
        GMRES<WC,C_ext,Size_T,Side::Left,false,false> gmres(
            n, gmres_max_iter, wc, CPU_thread_count
        );

        // set up the bdry operator and solve
        for( Int i = 0 ; i < wcc; i++ )
        {
            coeff_[4 * i + 0] = C_ext(0.5,0.0);
            coeff_[4 * i + 1] = C_ext(R_ext(0),-eta[i]);
            coeff_[4 * i + 2] = C_ext(1);
            coeff_[4 * i + 3] = C_ext(0);
        }

        LoadBoundaryOperators_PL(kappa_,coeff_,wc,wcs);
        
        auto A = [this,wc]( cptr<C_ext> x, mptr<C_ext> y )
        {
            ApplyBoundaryOperators_PL( C_ext(1), x, wc, C_ext(0), y, wc );
        };
        
        
        // Setup the mass matrix Preconditionier P:=M^-1.
        // P is also used for transf. into strong form.
        // Henrik is it?
        
        auto P = [this,wc,cg_tol]( cptr<C_ext> x, mptr<C_ext> y )
        {
            if constexpr ( lumped_mass_as_prec_for_intopsQ )
            {
                ApplyLumpedMassInverse<WC>( x, wc, y, wc,         wc );
            }
            else
            {
                ApplyMassInverse      <WC>( x, wc, y, wc, cg_tol, wc );
            }
        };
        
        (void)gmres(A,P,
            Scalar::One <C_ext>, wave, wc,
            Scalar::Zero<C_ext>, phi,  wc,
            gmres_tol, gmres_max_restarts
        );

        UnloadBoundaryOperators_PL();
    }

public:

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
        CheckInteger<I_ext>();
        CheckScalars<R_ext,C_ext>();
        
        std::string tag = ClassName()+"::DirichletToNeumann<"
            + "," + ToString(WC)
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);
        
        const Int n   = VertexCount();
        const Int wcc = int_cast<Int>(wave_chunk_count_);
        const Int wcs = int_cast<Int>(wave_chunk_size_ );
        const Int wc  = wcc * wcs;

        Tensor2<C_ext,Int> c_ ( wcc, 4);

        // The two boolean at the end of the template silence some messages.
        GMRES<WC,C_ext,Size_T,Side::Left,false,false> gmres(
            n, gmres_max_iter, wc, CPU_thread_count
        );

        // Setup the mass matrix Preconditionier P:=M^-1.
        // P is also used for transf. into strong form
        // Henrik: Is it?

        auto P = [this,wc,cg_tol]( cptr<C_ext> x, mptr<C_ext> y )
        {
            if constexpr ( lumped_mass_as_prec_for_intopsQ )
            {
                ApplyLumpedMassInverse<WC>( x, wc, y, wc,         wc );
            }
            else
            {
                ApplyMassInverse<WC>      ( x, wc, y, wc, cg_tol, wc );
            }
        };

        // set up the bdry operator and solve
        for( Int i = 0 ; i < wcc ; i++ )
        {
            c_(i,0) = C_ext(0.5f,0.0f);
            c_(i,1) = C_ext( R_ext(0), -kappa_[i] );
            c_(i,2) = C_ext(0);
            c_(i,3) = C_ext(1);
        }

        LoadBoundaryOperators_PL(kappa_,c_.data(),wc,wcs);

        auto A = [this,wc]( cptr<C_ext> x, mptr<C_ext> y )
        {
            ApplyBoundaryOperators_PL( C_ext(1), x, wc, C_ext(0), y, wc );
        };
        
        // solve for the normal derivatives of the near field solutions
        (void)gmres(A,P,
            Scalar::One <C_ext>, wave,          wc,
            Scalar::Zero<C_ext>, neumann_trace, wc,
            gmres_tol, gmres_max_restarts
        );

        UnloadBoundaryOperators_PL();

        ptoc(tag);
    }

    //------------------------------------------------------------------------------------------------------------------------------------------------------------------------

private:

// Henrik: I think this function is only meant to be called internally.
// Therefore I made I it `private`.

    // calculate factor * Re(X_in) .* normals
    template<typename R_ext, typename C_ext>
    void MultiplyWithNormals_PL(
        cptr<C_ext> X_in, 
        mptr<R_ext> Y_out,
        const R_ext factor,
        const R_ext cg_tol
    )
    {
        CheckScalars<R_ext,C_ext>();
        
        static_assert( std::is_same_v<Scalar::Real<C_ext>, R_ext>, "" );
        
        std::string tag = ClassName()+"::DotWithNormals_PL<"
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);
        
        const Int n = VertexCount();
        const Int m = SimplexCount();
        
        Tensor1<Complex,Int> X      ( m         );
        Tensor2<Real   ,Int> Y      ( m, Int(3) );
        Tensor2<Real   ,Int> Y_weak ( n, Int(3) );
        
        // Convert the input from PL to a PC function.
        // Also change precision to internal one.
        AvOp.Dot<1>(
            Scalar::One <Complex>, X_in,     Int(1),
            Scalar::Zero<Complex>, X.data(), Int(1),
            Int(1)
        );
        
        // From here on we use internal precision (float).

        // Pointwise multiplication of the STRONG FORM with the normals.
        // CheckThis
        ParallelDo(
            [this,&Y,&X]( const Int i )
            {
                const Real mul = Re(X[i]) / areas_ptr[i];

                for(Int j = 0; j < 3; ++j )
                {
                    Y(i,j) = mul * normals_ptr[i * 4 + j];
                }
            },
            m, CPU_thread_count
        );
        
        // Convert from PC function to PL function.
        AvOpTransp.Dot<3>(
            static_cast<Real>(factor), Y.data()     , Int(3),
            Scalar::Zero<Real>,        Y_weak.data(), Int(3),
            Int(3)
        );

        ApplyMassInverse<3>( Y_weak.data(), 3, Y_out, 3, cg_tol, 3 );
        
        ptoc(tag);
    }


    // calculate <X_in , normals>
    template<typename R_ext>
    void DotWithNormals_PL( 
        cptr<R_ext> X_in,
        mptr<R_ext> Y_out,
        const R_ext cg_tol
    )
    {
        CheckReal<R_ext>();
        
        std::string tag = ClassName()+"::DotWithNormals_PL<"
            + "," + TypeName<R_ext>
            + ">";
        
        ptic(tag);
        
        const Int m = SimplexCount();
        const Int n = VertexCount();

        Tensor2<Real,Int> X      ( m, Int(3) );
        Tensor1<Real,Int> Y      ( m         );
        Tensor1<Real,Int> Y_weak ( n         );
        
        // Convert the input from PL to a PC function.
        // Also change precision to internal one.
        AvOp.Dot<3>(
            Scalar::One <R_ext>, X_in,     Int(3),
            Scalar::Zero<R_ext>, X.data(), Int(3),
            Int(3)
        );
        
        // From here on we use internal precision (float).
        
        // Pointwise multiplication of the STRONG FORM with the normals.
        // CheckThis
        ParallelDo(
            [&X,&Y,this]( const Int i )
            {
                Real factor = Inv<Real>(areas_ptr[i]);

                Real sum = 0;
                
                for(Int j = 0; j < 3; ++j )
                {
                    sum += normals_ptr[i * 4 + j] * X(i,j) * factor;
                }
                
                Y[i] = sum;
            },
            m, CPU_thread_count
        );
        
        // Convert from PC function to PL function.
        AvOpTransp.Dot<1>(
            Scalar::One <Real>, Y.data(),      Int(1),
            Scalar::Zero<Real>, Y_weak.data(), Int(1),
            Int(1)
        );
        
        // Set the tolerance parameter for ApplyMassInverse.
        ApplyMassInverse<1>( Y_weak.data(), 1, Y_out, 1, cg_tol, 1 );

        ptoc(tag);
    }
