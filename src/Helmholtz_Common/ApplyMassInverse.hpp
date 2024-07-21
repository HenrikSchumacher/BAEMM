public:

    template<Int NRHS = VarSize, typename Scal>
    void ApplyMassInverse(
        cptr<Scal> B_in,  const Int ldB_in,
        mptr<Scal> C_out, const Int ldC_out,
        const Int nrhs = NRHS
    )
    {
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
        
        if constexpr( use_mass_choleskyQ )
        {
            if constexpr ( Scalar::ComplexQ<Scal> )
            {
                // In this case, Scal is of the form std::complex<T>
                // with T either float or double.
                //
                // Now we can employ a nasty trick because Mass is a matrix of float and
                // std::complex stores real and imaginary part consecutively:
                // A m x n matrix of std::complex<T> can be reinterpreted as
                // as a m x (2 * n) matrix of T.
                
                InverseMassMatrix().Solve(
                    reinterpret_cast<const Scalar::Real<Scal> *>( B_in  ), Int(2) * ldB_in,
                    reinterpret_cast<      Scalar::Real<Scal> *>( C_out ), Int(2) * ldC_out,
                    Int(2) * nrhs
                );
            }
            else
            {
                InverseMassMatrix().Solve(B_in,ldB_in,C_out,ldC_out,nrhs);
            }
        }
        else
        {
            auto P = [&,nrhs]( cptr<Scal> x, mptr<Scal> y )
            {
                if constexpr ( use_lumped_mass_as_precQ )
                {
                    ApplyLumpedMassInverse<NRHS>( x, nrhs, y, nrhs, nrhs );
                }
                else
                {
                    ParallelDo(
                        [&,x,y]( const Int i )
                        {
                           copy_buffer<NRHS>( &x[nrhs * i], &y[nrhs * i], nrhs );
                        },
                        vertex_count, CPU_thread_count
                    );
                }
                
            };

            auto A = [&,nrhs]( cptr<Scal> x, mptr<Scal> y )
            {
                Mass.Dot<NRHS>(
                    Scalar::One <Scal>, x, nrhs,
                    Scalar::Zero<Scal>, y, nrhs,
                    nrhs
                );
            };
            
            constexpr Int max_iter = use_lumped_mass_as_precQ ? 20 : 100;
            
            ConjugateGradient<NRHS,Scal,Size_T> cg( vertex_count, max_iter, nrhs, CPU_thread_count );

            zerofy_buffer(C_out, static_cast<std::size_t>(vertex_count * ldC_out), CPU_thread_count);

            bool succeeded = cg(A,P,B_in,ldB_in,C_out,ldC_out,cg_tol);

            if( !succeeded )
            {
                wprint(ClassName()+"::ApplyMassInverse: CG algorithm did not converge.");
            }
        }
        
        ptoc(tag);
    }

    template<Int NRHS = VarSize, typename Scal>
    void ApplyLumpedMassInverse(
        cptr<Scal> B_in,  const Int ldB_in,
        mptr<Scal> C_out, const Int ldC_out,
        const Int nrhs = NRHS
    )
    {
        std::string tag = ClassName()+"::ApplyLumpedMassInverse<"+ToString(NRHS)
            + "," + TypeName<Scal>
            + ">";
        ptic(tag);
        
        ParallelDo(
            [&,B_in,ldB_in,C_out,ldC_out]( const Int i )
            {
                const Scalar::Real<Scal> factor = areas_lumped_inv[i];
                
                for( Int j = 0; j < ((NRHS > VarSize) ? NRHS : nrhs); ++j )
                {
                    C_out[ldC_out * i + j] = factor * B_in[ldB_in * i + j];
                }
            },
            vertex_count, CPU_thread_count
        );
        
        ptoc(tag);
    }
