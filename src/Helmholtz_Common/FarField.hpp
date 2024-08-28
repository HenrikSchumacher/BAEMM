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
        
        std::string tag = ClassName()+"::FarField" 
            + "<" + (WC <= VarSize ? std::string("VarSize") : ToString(WC) )
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

    template<Int WC = VarSize, typename I_ext, typename R_ext, typename C_ext>
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
        print("unload start");
        UnloadBoundaryOperators_PL();
        print("ff");
        ApplyFarFieldOperators_PL<WC>(
            C_ext(1), phi.data(), wc,
            C_ext(0), Y_out,      wc,
            kappa_, coeff.data(), wc, wcs
        );
    }

public:

    template<Int WC = VarSize, typename I_ext, typename R_ext, typename C_ext>
    void Derivative_FF_Helper(
        cptr<R_ext> kappa_,
        const I_ext wave_chunk_count_,
        cptr<R_ext> inc_directions,
        const I_ext wave_chunk_size_,
        mptr<C_ext> du_dn, //du_dn is the Neumann data of the scattered wave
        const WaveType type,
        const R_ext cg_tol,
        const R_ext gmres_tol
    )
    {
        CheckInteger<I_ext>();
        CheckScalars<R_ext,C_ext>();
        
        std::string tag = ClassName()+"::Derivative_FF_Helper<"
            + "," + (WC <= VarSize ? std::string("VarSize") : ToString(WC) )
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);
        
        // `du_dn` = A \ `inc_wave` is the normal derivative of the solution with inc wave `wave`
        // `phi` is the bdry potential for the incident wave `du_dn` *(<X_in , n>), the solution is the far field to this
        // Formulas follow from Thorsten's book.
        
        // du_dn needs to be preallocated of size n x (wave_chunk_count_ * wave_chunk_size_).

        const Int n   = VertexCount();
        const Int wcc = int_cast<Int>(wave_chunk_count_);
        const Int wcs = int_cast<Int>(wave_chunk_size_ );
        const Int wc  = wcc * wcs;
        
        Tensor2<C_ext,Int> inc_coeff ( wcc, 4  );
        Tensor2<C_ext,Int> inc_wave  ( n,   wc );  //weak representation of the incident wave
        
        // Create weak representation of the negative incident wave.
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
        
        // `inc_wave` is the incoming wave in weak form, i.e.,
        // multiplied by a mass-ish matrix.
        
        DirichletToNeumann<WC>( kappa_, inc_wave.data(), du_dn, wcc, wcs, cg_tol, gmres_tol );
        
        ptoc(tag);
    }

    template<Int WC = VarSize, typename I_ext, typename R_ext, typename C_ext>
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
            + "," + (WC <= VarSize ? std::string("VarSize") : ToString(WC) )
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
        Tensor2<C_ext,Int>  bdr_cond      ( n,   wc );
        Tensor2<C_ext,Int>  bdr_cond_weak ( n,   wc );
        Tensor2<C_ext,Int>  phi           ( n,   wc );
        
        Tensor1<R_ext,Int>  X_dot_normal  ( n );
        
        if( du_dn == nullptr )
        {
            du_dn = (C_ext*)calloc(n * wc, sizeof(C_ext));
            
            Derivative_FF_Helper<WC>(
                kappa_,         wcc,
                inc_directions, wcs,
                du_dn,
                type, cg_tol, gmres_tol
            );
        }
        
        DotWithNormals_PL( X_in, X_dot_normal.data(), cg_tol );
        
        // CheckThis
        ParallelDo(
            [&X_dot_normal,&bdr_cond,du_dn,wc]( const Int i )
            {
                // bdr_cond[wc * i] = -X_dot_normal[i] * du_dn[wc * i]
                
                combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Zero,WC>(
                    -X_dot_normal[i],    &du_dn[wc * i],
                    Scalar::Zero<C_ext>, bdr_cond.data(i),
                    wc
                );
            },
            n, CPU_thread_count
        );
        
        // apply mass to the boundary conditions to get weak representation
        MassOp.Dot<WC>(
            Scalar::One <C_ext>, bdr_cond.data(),      wc,
            Scalar::Zero<C_ext>, bdr_cond_weak.data(), wc,
            wc
        );
        
        BoundaryPotential<WC>(
            kappa_, coeff.data(), bdr_cond_weak.data(), phi.data(),
            wcc, wcs, cg_tol, gmres_tol
        );
        
        ApplyFarFieldOperators_PL<WC>(
            C_ext(1), phi.data(), wc,
            C_ext(0), Y_out,      wc,
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
            + "," + (WC <= VarSize ? std::string("VarSize") : ToString(WC) )
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);

        const Int n   = VertexCount();
        const Int wcc = int_cast<Int>(wave_chunk_count_);
        const Int wcs = int_cast<Int>(wave_chunk_size_ );
        const Int wc  = wcc * wcs;

        // Ensure presence of du_dn.
        if( du_dn == nullptr )
        {
            du_dn = (C_ext*)calloc(n * wc, sizeof(C_ext));
            
            Derivative_FF_Helper<WC>(
                kappa_,         wcc,
                inc_directions, wcs,
                du_dn,
                type, cg_tol, gmres_tol
            );
        }
        
        
        Tensor2<C_ext,Int>  herglotz_wave ( n, wc ); //weak representation of the herglotz wave
        Tensor2<C_ext,Int>  dv_dn_buf     ( n, wc );
        
        mptr<C_ext> dv_dn = dv_dn_buf.data();
        
        Tensor1<C_ext,Int> wave_product  ( n );
        
        // Create Herglotz wave.
        Tensor2<C_ext,Int> inc_coeff ( wcc, 4 );
        
        for( Int i = 0 ; i < wcc ; i++)
        {
            inc_coeff(i,0) = C_ext(0);
            inc_coeff(i,1) = C_ext(R_ext(0),-kappa_[i]);
            inc_coeff(i,2) = C_ext(1);
            inc_coeff(i,3) = C_ext(0);
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


    // Computes X = alpha (DF^T/DF + M)^{-1}.B + beta * X.

    template<Int WC,
        typename I_ext, typename R_ext, typename C_ext,
        typename M_T, typename P_T
    >
    bool GaussNewtonSolve(
        cptr<R_ext> kappa_,
        const I_ext wave_chunk_count_,
        cptr<R_ext> inc_directions,
        const I_ext wave_chunk_size_,
        M_T & M,
        P_T & P,
        // Supply the differential of the Tikonov functional here.
        const R_ext alpha, cptr<R_ext> B_in,  const I_ext ldB,
        // Write the seach direction here.
        const R_ext beta , mptr<R_ext> X_out, const I_ext ldX,
        //du_dn is the Neumann data of the scattered wave.
        C_ext * & du_dn,
        const WaveType type,
        const R_ext cg_tol,
        const R_ext gmres_tol_inner,
        const R_ext gmres_tol_outer
    )
    {
        CheckInteger<I_ext>();
        CheckScalars<R_ext,C_ext>();
        
        std::string tag = ClassName()+"::GaussNewtonSolve<"
            + "," + (WC <= VarSize ? std::string("VarSize") : ToString(WC) )
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);

        // Calculates a Gauss-Newton step.
        
        using GMRES_Scal = R_ext;
        
        constexpr auto one  = Scalar::One <GMRES_Scal>;
        constexpr auto zero = Scalar::Zero<GMRES_Scal>;
        
        constexpr Size_T DIM = 3; // Dimension of the ambient space.
        
        const Int n   = VertexCount();
        const Int m   = GetMeasCount();
        const Int wcc = int_cast<Int>(wave_chunk_count_);
        const Int wcs = int_cast<Int>(wave_chunk_size_ );
        const Int wc  = wcc * wcs;
        
        
        // Since B_in and X_out may be in scattered format,
        // we may need to copy the inputs and outputs.
        
        // Local buffers of contiguous memory.
        // Will only be allocated if necessary.
        Tensor2<GMRES_Scal,Size_T> B_loc;
        Tensor2<GMRES_Scal,Size_T> X_loc;
        
        if( ldB != DIM )
        {
            // External buffer B_in is scattered.
            // Allocate local, contiguous storage and copy input B_in into it.
            B_loc = Tensor2<GMRES_Scal,Size_T>( n, DIM );

//                         +--- First dimension of matrix is unknown at compile time.
//                         |      +--- Second dimension is known at compile time.
//                         |      |     +--- Allow parallel execution.
//                         v      v     v
            copy_matrix<VarSize,DIM,Parallel>(
                B_in, ldB, B_loc.data(), DIM, n, DIM, CPU_thread_count
            );
        }
        
        if( ldX != DIM )
        {
            // External buffer X_out is scattered.
            // Allocate local, contiguous storage and let X point to it.
            X_loc = Tensor2<GMRES_Scal,Size_T>( n, DIM );
        }
        
        GMRES<1,GMRES_Scal,Size_T,Side::Left,false,false> gmres(
            DIM * n, gmres_max_iter, 1, CPU_thread_count );

        
        // A, M and P are matrices of size n x n.
        // They are applied to matrices of size n x DIM
        // However, DF operates on _vector_ of size (n * DIM).
        
        Tensor2<C_ext,Int>  DF       ( m, wc  );
        Tensor2<R_ext,Int>  y_strong ( n, DIM );

        
        // A computes A.x = M.x + DF^*.DF.x;
        auto A = [&]( cptr<GMRES_Scal> x, mptr<GMRES_Scal> y )
        {
            M(x,y); // The metric m has to return M.x;
            
            Derivative_FF<WC>(
                kappa_, wcc, inc_directions, wcs,
                x, DF.data(), 
                du_dn, type, cg_tol, gmres_tol_inner
            );
            
            AdjointDerivative_FF<WC>(
                kappa_, wcc, inc_directions, wcs,
                DF.data(), y_strong.data(), 
                du_dn, type, cg_tol, gmres_tol_inner
            );

            MassOp.Dot<DIM>(
                one, y_strong.data(), DIM,
                one, y,               DIM,
                DIM
            );
        };
        
        // Computes X = (DF^T/DF + M)^{-1}.B.
        bool succeeded = gmres(A,P,
            one , (ldB!=DIM) ? B_loc.data() : B_in , Size_T(1),
            zero, (ldX!=DIM) ? X_loc.data() : X_out, Size_T(1),
            gmres_tol_outer, gmres_max_restarts
        );
        
        if( ldX != DIM )
        {
            // External buffer is scattered.
            // We need to copy the results from the local storage to it.
            
            constexpr auto G = Scalar::Flag::Generic;
            
            // X_out = alpha * X_loc + beta * X_out;
            //
            //               +--- alpha is not known to have a special value like 0, 1, -1.
            //               | +--- beta is not known to have a special value like 0, 1, -1.
            //               | |   +--- First dimension of matrix is unknown at compile time.
            //               | |   |      +--- Second dimension is known at compile time.
            //               | |   |      |     +--- Activate parallel execution.
            //               v v   v      v     v
            combine_matrices<G,G,VarSize,DIM,Parallel>(
                alpha, X_loc.data(), DIM, beta, X_out, ldX, n, DIM, CPU_thread_count
            );
        }
        
        ptoc(tag);
        
        return succeeded;
    }

    //----------------------------------------------------------------------------------------------------------------------------------------------------------------

public:

    template<Int WC = VarSize, typename I_ext, typename R_ext, typename C_ext>
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
        
        std::string tag = ClassName()+"::BoundaryPotential"
            + "<" + (WC <= VarSize ? std::string("VarSize") : ToString(WC) )
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
        
        // Using the external precision in the solver.
        using GMRES_Scal = C_ext;
        
//        // Using the internal precision in the solver.
//        using GMRES_Scal = Complex;
        
        // The two boolean at the end of the template silence some messages.
        GMRES<WC,GMRES_Scal,Size_T,Side::Left,false,false> gmres(
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

        auto A = [this,wc]( cptr<GMRES_Scal> x, mptr<GMRES_Scal> y )
        {
//            print("op");
            ApplyBoundaryOperators_PL<WC>(
                GMRES_Scal(1), x, wc,
                GMRES_Scal(0), y, wc
            );
        };
        
        
        
        // Setup the mass matrix Preconditionier P:=M^-1.
        // P is also used for transf. into strong form.
        // Henrik is it?
        
        auto P = [this,wc,cg_tol]( cptr<GMRES_Scal> x, mptr<GMRES_Scal> y )
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
            + "," + (WC <= VarSize ? std::string("VarSize") : ToString(WC) )
            + "," + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "," + TypeName<C_ext>
            + ">";
        
        ptic(tag);
        
        // Caution: `wave` is supposed to be in weak form!
        // Caution: `neumann_trace` will be in strong form!
        
        const Int n   = VertexCount();
        const Int wcc = int_cast<Int>(wave_chunk_count_);
        const Int wcs = int_cast<Int>(wave_chunk_size_ );
        const Int wc  = wcc * wcs;

        // Using the external precision in GMRES.
        using GMRES_Scal = C_ext;
        
//        // Using the internal (lower) precision in GMRES.
//        using GMRES_Scal = Complex;
        
        // The two boolean at the end of the template silence some messages.
        GMRES<WC,GMRES_Scal,Size_T,Side::Left,false,false> gmres(
            n, gmres_max_iter, wc, CPU_thread_count
        );

        // Setup the mass matrix Preconditionier P:=M^-1.

        auto P = [this,wc,cg_tol]( cptr<GMRES_Scal> x, mptr<GMRES_Scal> y )
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

        
        
        Tensor2<C_ext,Int> c_ ( wcc, 4);
        
        // set up the bdry operator and solve
        for( Int i = 0 ; i < wcc ; i++ )
        {
            c_(i,0) = C_ext(0.5f,0.0f);
            c_(i,1) = C_ext( R_ext(0), -kappa_[i] );
            c_(i,2) = C_ext(0);
            c_(i,3) = C_ext(1);
        }

        LoadBoundaryOperators_PL(kappa_,c_.data(),wc,wcs);

        auto A = [this,wc]( cptr<GMRES_Scal> x, mptr<GMRES_Scal> y )
        {
            ApplyBoundaryOperators_PL<WC>( 
                GMRES_Scal(1), x, wc,
                GMRES_Scal(0), y, wc
            );
        };
        
        // Solve for the normal derivatives of the near field solutions.
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

        // Write to output and convert to external precision.
        ApplyMassInverse<3>( 
            Y_weak.data(), Int(3),
            Y_out,         Int(3),
            cg_tol, Int(3)
        );
        
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
        
        std::string tag = ClassName()+"::DotWithNormals_PL"
            + "<" + TypeName<R_ext>
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
            Scalar::One <Real>, X_in,     Int(3),
            Scalar::Zero<Real>, X.data(), Int(3),
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
        ApplyMassInverse<1>( 
            Y_weak.data(), Int(1),
            Y_out,         Int(1),
            cg_tol, Int(1)
        );
        
        ptoc(tag);
    }
