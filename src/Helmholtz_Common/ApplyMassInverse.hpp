public:

    template<Int NRHS = VarSize, typename X_T, typename Y_T, typename R_ext>
    void ApplyMassInverse(
        cptr<X_T> X, const Int ldX,
        mptr<Y_T> Y, const Int ldY,
        const R_ext cg_tol,
        const Int nrhs = NRHS
    )
    {
        // Solves M.Y = X or Y = M^{-1}.X
        
        // Uses CG algorithm to multiply with inverse mass matrix.
        // If NRHS > 0, then nrhs will be ignored and loops are unrolled and vectorized at compile time.
        // If NRHS == 0, then nrhs is used.
        
        CheckReal<R_ext>();
        
        // Internally, the type `Real` is used, so `X_T` and `Y_T` must encode real types, too.
        
        static_assert( Scalar::RealQ   <X_T> == Scalar::RealQ   <Y_T>, "" );
        static_assert( Scalar::ComplexQ<X_T> == Scalar::ComplexQ<Y_T>, "" );

        std::string tag = ClassName()+"::ApplyMassInverse"
            + "<" + (NRHS <= VarSize ? std::string("VarSize") : ToString(NRHS) )
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
        
        // Per default `ConjugateGradient` assumes that the initial guess is 0.
        // zerofy_matrix<NRHS,Parallel>( Y, ldY, vertex_count, nrhs, CPU_thread_count );

        bool succeeded;
        
        if constexpr ( Scalar::ComplexQ<Y_T> )
        {
//            using CG_Scal = Complex;
//            using CG_Real = Real;
            
            using CG_Scal = Y_T;
            using CG_Real = Scalar::Real<CG_Scal>;
            
//            using CG_Scal = Complex64;
//            using CG_Real = Real64;
            
            // The two boolean at the end of the template silence some messages.
            ConjugateGradient<NRHS,CG_Scal,Size_T,false,false> cg(
                vertex_count, max_iter, nrhs, CPU_thread_count
            );
            
            // Here we use a dirty trick. std::complex<T> is implemented such
            // that
            //           reinterpret_cast<const T *>(x)[0] == x.real()
            // and
            //           reinterpret_cast<const T *>(x)[1] == x.imag()
            //
            // So, when we reinterpret_cast a pointer to an array of n std::complex<T> to
            // a pointer of T, then we implicitly get an array of 2*n elements
            // of type T, which contains the real and imaginary parts in interleaved
            // form.

            auto A = [this,nrhs]( cptr<CG_Scal> x, mptr<CG_Scal> y )
            {
                MassOp.Dot<2 * NRHS>(
                    Scalar::One <CG_Real>, reinterpret_cast<const CG_Real *>(x), 2 * nrhs,
                    Scalar::Zero<CG_Real>, reinterpret_cast<      CG_Real *>(y), 2 * nrhs,
                    2 * nrhs
                );
            };
            
            auto P = [this,nrhs]( cptr<CG_Scal> x, mptr<CG_Scal> y )
            {
                ApplyLumpedMassInverse<2 * NRHS>(
                    reinterpret_cast<const CG_Real *>(x), 2 * nrhs,
                    reinterpret_cast<      CG_Real *>(y), 2 * nrhs,
                    2 * nrhs
                );
            };
            
            succeeded = cg(
                A, P,
                Scalar::One <Y_T>, X, ldX,
                Scalar::Zero<Y_T>, Y, ldY,
                static_cast<CG_Real>(cg_tol), false
            );
        }
        else
        {
//            using CG_Scal = Real;
//            using CG_Real = Real;
            
            using CG_Scal = Y_T;
            using CG_Real = Y_T;
            
//            using CG_Scal = Real64;
//            using CG_Real = Real64;
            
            
            // The two boolean at the end of the template silence some messages.
            ConjugateGradient<NRHS,CG_Scal,Size_T,false,false> cg(
                vertex_count, max_iter, nrhs, CPU_thread_count
            );

            auto A = [this,nrhs]( cptr<CG_Scal> x, mptr<CG_Scal> y )
            {
                MassOp.Dot<NRHS>(
                    Scalar::One <CG_Scal>, x, nrhs,
                    Scalar::Zero<CG_Scal>, y, nrhs,
                    nrhs
                );
            };
            
            auto P = [this,nrhs]( cptr<CG_Scal> x, mptr<CG_Scal> y )
            {
                ApplyLumpedMassInverse<NRHS>(
                    x, nrhs,
                    y, nrhs,
                    nrhs
                );
            };
            
            succeeded = cg(
                A, P,
                Scalar::One <Y_T>, X, ldX,
                Scalar::Zero<Y_T>, Y, ldY,
                static_cast<CG_Real>(cg_tol), false
            );
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
        
        std::string tag = ClassName()+"::ApplyLumpedMassInverse"
            + "<" + (NRHS <= VarSize ? std::string("VarSize") : ToString(NRHS) )
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
                    factor, &X[ldX * i],
                    Y_T(0), &Y[ldY * i],
                    nrhs
                );
            },
            vertex_count, CPU_thread_count
        );
        
        ptoc(tag);
    }
