public:

    template<Int NRHS = VarSize, typename X_T, typename Y_T, typename R_ext>
    void ApplyMassInverse(
        cptr<X_T> X, const Int ldX,
        mptr<Y_T> Y, const Int ldY,
        const R_ext cg_tol,
        const Int nrhs = NRHS
    )
    {
        // Uses CG algorithm to multiply with inverse mass matrix.
        // If NRHS > 0, then nrhs will be ignored and loops are unrolled and vectorized at compile time.
        // If NRHS == 0, then nrhs is used.
        
        ASSERT_REAL(R_ext);
        
        // Internally, the type `Real` is used, so `X_T` and `Y_T` must encode real types, too.
        
        static_assert( Scalar::RealQ   <X_T> == Scalar::RealQ   <Y_T>, "" );
        static_assert( Scalar::ComplexQ<X_T> == Scalar::ComplexQ<Y_T>, "" );

        std::string tag = ClassName()+"::ApplyMassInverse"
            + "<" + ToString(NRHS)
            + "," + TypeName<X_T>
            + "," + TypeName<Y_T>
            + "," + TypeName<R_ext>
            + ">("+ToString(nrhs)+")";
        
        if( nrhs > ldX )
        {
            eprint("nrhs > ldX");
            return;
        }
        
        if( nrhs > ldY )
        {
            eprint("nrhs > ldY");
            return;
        }
        
        ptic(tag);
        
        constexpr Int max_iter = 20;
        
        zerofy_matrix<NRHS>( Y, ldY, vertex_count, nrhs, CPU_thread_count );

        bool succeeded;
        
        if constexpr ( Scalar::ComplexQ<X_T> )
        {
            ConjugateGradient<NRHS,Complex,Size_T> cg( vertex_count, max_iter, nrhs, CPU_thread_count );

            auto A = [this,nrhs]( cptr<Complex> x, mptr<Complex> y )
            {
                Mass.Dot<2 * NRHS>(
                    Scalar::One <Real>, reinterpret_cast<const Real *>(x), 2 * nrhs,
                    Scalar::Zero<Real>, reinterpret_cast<      Real *>(y), 2 * nrhs,
                    2 * nrhs
                );
            };
            
            auto P = [this,nrhs]( cptr<Complex> x, mptr<Complex> y )
            {
                ApplyLumpedMassInverse<2 * NRHS>(
                    reinterpret_cast<const Real *>(x), 2 * nrhs,
                    reinterpret_cast<      Real *>(y), 2 * nrhs,
                    2 * nrhs
                );
            };
            
            succeeded = cg(A,P,X,ldX,Y,ldY,static_cast<Real>(cg_tol));
        }
        else
        {
            ConjugateGradient<NRHS,Real,Size_T> cg( vertex_count, max_iter, nrhs, CPU_thread_count );

            auto A = [this,nrhs]( cptr<Real> x, mptr<Real> y )
            {
                Mass.Dot<NRHS>(
                    Scalar::One <Real>, x, nrhs,
                    Scalar::Zero<Real>, y, nrhs,
                    nrhs
                );
            };
            
            auto P = [this,nrhs]( cptr<Real> x, mptr<Real> y )
            {
                ApplyLumpedMassInverse<NRHS>(x,nrhs,y,nrhs,nrhs);
            };
            
            succeeded = cg(A,P,X,ldX,Y,ldY,static_cast<Real>(cg_tol));
        }

        if( !succeeded )
        {
            wprint(ClassName()+"::ApplyMassInverse: CG algorithm did not converge.");
        }
        
        ptoc(tag);
    }

    template<Int NRHS = VarSize, typename X_T, typename Y_T>
    void ApplyLumpedMassInverse(
        cptr<X_T> X, const Int ldX,
        mptr<Y_T> Y, const Int ldY,
        const Int nrhs = NRHS
    )
    {
        // If NRHS > 0, then nrhs will be ignored and loops are unrolled and vectorized at compile time.
        // If NRHS == 0, then nrhs is used.
        
        std::string tag = ClassName()+"::ApplyLumpedMassInverse<"+ToString(NRHS)
            + "," + TypeName<X_T>
            + "," + TypeName<Y_T>
            + ">";
        ptic(tag);
        
        ParallelDo(
            [&,X,ldX,Y,ldY]( const Int i )
            {
                const Scalar::Real<Y_T> factor = static_cast<Scalar::Real<Y_T>>(areas_lumped_inv[i]);

                // Compute
                // `Y[ldY * i + j] = factor * X [ldX * i + j]`
                // for j in [0, `(NRHS>0) ? NRHS : nrhs`[.
                
                combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Zero,NRHS>(
                    factor,            &X[ldX * i],
                    Scalar::Zero<Y_T>, &Y[ldY * i],
                    nrhs
                );
            },
            vertex_count, CPU_thread_count
        );
        
        ptoc(tag);
    }
