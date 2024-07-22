public:

    template<Int NRHS = VarSize, typename B_T, typename C_T>
    void ApplyMassInverse(
        cptr<B_T> B_in,  const Int ldB_in,
        mptr<C_T> C_out, const Int ldC_out,
        const Int nrhs = NRHS
    )
    {
        // Uses CG algorithm to multiply with inverse mass matrix.
        // If NRHS > 0, then nrhs will be ignored and loops are unrolled and vectorized at compile time.
        // If NRHS == 0, then nrhs is used.
        
        // Internally, the type `Real` is used, so `B_T` and `C_T` must encode real types, too.
        
        ASSERT_REAL(B_T)
        ASSERT_REAL(C_T)

        std::string tag = ClassName()+"::ApplyMassInverse<"+ToString(NRHS)
            + ">("+ToString(nrhs)+")";
        
        if( nrhs > ldB_in )
        {
            eprint("nrhs > ldB_in");
            return;
        }
        
        if( nrhs > ldC_out )
        {
            eprint("nrhs > ldC_out");
            return;
        }
        
        
        ptic(tag);
        
        auto P = [&,nrhs]( cptr<Real> x, mptr<Real> y )
        {
            ApplyLumpedMassInverse<NRHS>( x, nrhs, y, nrhs, nrhs );
        };

        auto A = [&,nrhs]( cptr<Real> x, mptr<Real> y )
        {
            Mass.Dot<NRHS>(
                Scalar::One <Real>, x, nrhs,
                Scalar::Zero<Real>, y, nrhs,
                nrhs
            );
        };
        
        constexpr Int max_iter = 20;
        
        ConjugateGradient<NRHS,Real,Size_T> cg( vertex_count, max_iter, nrhs, CPU_thread_count );

        zerofy_buffer(C_out, static_cast<std::size_t>(vertex_count * ldC_out), CPU_thread_count);

        bool succeeded = cg(A,P,B_in,ldB_in,C_out,ldC_out,cg_tol);

        if( !succeeded )
        {
            wprint(ClassName()+"::ApplyMassInverse: CG algorithm did not converge.");
        }
        
        ptoc(tag);
    }

    template<Int NRHS = VarSize, typename B_T, typename C_T>
    void ApplyLumpedMassInverse(
        cptr<B_T> B_in,  const Int ldB_in,
        mptr<C_T> C_out, const Int ldC_out,
        const Int nrhs = NRHS
    )
    {
        // If NRHS > 0, then nrhs will be ignored and loops are unrolled and vectorized at compile time.
        // If NRHS == 0, then nrhs is used.
        
        std::string tag = ClassName()+"::ApplyLumpedMassInverse<"+ToString(NRHS)
            + "," + TypeName<B_T>
            + "," + TypeName<C_T>
            + ">";
        ptic(tag);
        
        ParallelDo(
            [&,B_in,ldB_in,C_out,ldC_out]( const Int i )
            {
                const Scalar::Real<C_T> factor = static_cast<Scalar::Real<C_T>>(areas_lumped_inv[i]);

                // Compute
                // `C_out[ldC_out * i + j] = factor * B_in [ldB_in * i + j]`
                // for j in [0, `(NRHS>0) ? NRHS : nrhs`[.
                
                combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Zero,NRHS>(
                    factor,            &B_in [ldB_in  * i],
                    Scalar::Zero<C_T>, &C_out[ldC_out * i],
                    nrhs
                );
            },
            vertex_count, CPU_thread_count
        );
        
        ptoc(tag);
    }
