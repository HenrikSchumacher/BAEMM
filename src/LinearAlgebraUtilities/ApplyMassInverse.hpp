public:

    template<Int nrhs, typename Scal, typename R_ext>
    void ApplyMassInverse(
        cptr<Scal> B_in,  const Int ldB_in,
        mptr<Scal> C_out, const Int ldC_out,
        const R_ext cg_tol
    )
    {
        std::string tag = ClassName()+"::ApplyMassInverse<"+ToString(nrhs)
            + "," + TypeName<Scal>
            + ">";
        
        ptic(tag);
        
        if( nrhs != ldB_in )
        {
            eprint("nrhs != ldB_in");
        }
        
        if( nrhs != ldC_out )
        {
            eprint("nrhs != ldC_out");
        }
        
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
                    Int(2) * Int(nrhs)
                );
            }
            else
            {
                InverseMassMatrix().Solve(B_in,ldB_in,C_out,ldC_out,nrhs);
            }
        }
        else
        {
            ConjugateGradient<nrhs,Scal,size_t> cg(vertex_count,200,CPU_thread_count);

            auto P = [&]( cptr<Scal> x, mptr<Scal> y )
            {
                if constexpr ( use_lumped_mass_as_precQ )
                {
                    ApplyLumpedMassInverse<nrhs>( x, nrhs, y, nrhs );
                }
                else
                {
                    ParallelDo(
                        [&,x,y]( const Int i )
                        {
                            copy_buffer<nrhs>( &x[nrhs * i], &y[nrhs * i] );
                        },
                        vertex_count, CPU_thread_count
                    );
                }
                
            };

            auto A = [&]( cptr<Scal> x, mptr<Scal> y )
            {
                Mass.Dot<nrhs>(
                    Scalar::One <Scal>, x, nrhs,
                    Scalar::Zero<Scal>, y, nrhs
                );
            };

            zerofy_buffer(C_out, static_cast<size_t>(vertex_count * ldC_out), CPU_thread_count);

            bool succeeded = cg(A,P,B_in,ldB_in,C_out,ldC_out,cg_tol);

            if( !succeeded )
            {
                eprint(ClassName()+"::ApplyMassInverse_CG did not converge.");
            }
        }
        
        ptoc(tag);
    }

    template<Int nrhs, typename Scal>
    void ApplyLumpedMassInverse(
        cptr<Scal> B_in,  const Int ldB_in,
        mptr<Scal> C_out, const Int ldC_out
    )
    {
        std::string tag = ClassName()+"::ApplyLumpedMassInverse<"+ToString(nrhs)
            + "," + TypeName<Scal>
            + ">";
        ptic(tag);
        
        ParallelDo(
            [&,B_in,ldB_in,C_out,ldC_out]( const Int i )
            {
                const Scalar::Real<Scal> factor = areas_lumped_inv[i];
                
                for( Int j = 0; j < nrhs; ++j )
                {
                    C_out[ldC_out * i + j] = factor * B_in[ldB_in * i + j];
                }
            },
            vertex_count, CPU_thread_count
        );
        
        ptoc(tag);
    }
