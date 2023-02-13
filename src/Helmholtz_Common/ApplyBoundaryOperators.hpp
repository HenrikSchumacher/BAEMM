public:

    template<typename R_ext, typename C_ext, typename I_ext>
    void ApplyBoundaryOperators_PL(
        const C_ext alpha, ptr<C_ext> B_in,  const I_ext ldB_in,
        const C_ext beta,  mut<C_ext> C_out, const I_ext ldC_out,
        const R_ext kappa_,
        const C_ext coeff_0,
        const C_ext coeff_1,
        const C_ext coeff_2,
        const C_ext coeff_3,
        const I_ext wave_count_,
        const I_ext wave_chunk_size_
    )
    {
        // Computes
        //
        //     C_out = alpha * A * B_in + beta * C_out,
        //
        // where B_in and C_out out are matrices of size vertex_count x wave_count_ and
        // represent the vertex values of  wave_count_ piecewise-linear functions.
        // The operator A is a linear combination of several operators, depending on kappa:
        //
        // A =   coeff_0 * MassMatrix
        //     + coeff_1 * SingleLayerOp[kappa]
        //     + coeff_2 * DoubleLayerOp[kappa]
        //     + coeff_3 * AdjDblLayerOp[kappa]
        //
        //
        
        ASSERT_INT(I_ext);
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);
        
        LoadCoefficients(kappa_,coeff_0,coeff_1,coeff_2,coeff_3,wave_count_,wave_chunk_size_);
        
        ApplyBoundaryOperators_PL( alpha, B_in, ldB_in, beta, C_out, ldC_out );
    }


//    template<typename R_ext, typename C_ext, typename I_ext>
//    void ApplyBoundaryOperators_PL(
//        const C_ext alpha, ptr<C_ext> B_in,  const Int ldB_in,
//        const C_ext beta,  mut<C_ext> C_out, const Int ldC_out,
//        const Tensor1<R_ext,I_ext> & kappa_list,
//        const Tensor2<C_ext,I_ext> & coeff_list,
//        const Int wave_count_,
//        const Int wave_chunk_size_
//    )
//    {
//        //  The same as above, but with several wave numbers kappa_list and several coefficients.
//
//        ASSERT_INT(I_ext);
//        ASSERT_REAL(R_ext);
//        ASSERT_COMPLEX(C_ext);
//
//        LoadCoefficients(kappa_list.data(),coeff_list.data(),wave_count_,wave_chunk_size_);
//
//        ApplyBoundaryOperators_PL( alpha, B_in, ldB_in, beta, C_out, ldC_out );
//    }

    template<typename R_ext, typename C_ext, typename I_ext>
    void ApplyBoundaryOperators_PL(
        const C_ext alpha, ptr<C_ext> B_in,  const I_ext ldB_in,
        const C_ext beta,  mut<C_ext> C_out, const I_ext ldC_out,
        const R_ext * kappa_list,
        const C_ext * coeff_list,
        const I_ext wave_count_,
        const I_ext wave_chunk_size_
    )
    {
        //  The same as above, but with several wave numbers kappa_list and several coefficients.

        ASSERT_INT(I_ext);
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);

        LoadCoefficients(kappa_list, coeff_list, wave_count_, wave_chunk_size_);
        
        ApplyBoundaryOperators_PL( alpha, B_in, ldB_in, beta, C_out, ldC_out );
    }

    template<typename C_ext, typename I_ext>
    void ApplyBoundaryOperators_PL(
        const C_ext alpha, ptr<C_ext> B_in,  const I_ext ldB_in_,
        const C_ext beta,  mut<C_ext> C_out, const I_ext ldC_out_
    )
    {
        // The same as above, but assumes that
        ASSERT_INT(I_ext);
        ASSERT_COMPLEX(C_ext);
        
        ptic(ClassName()+"::ApplyBoundaryOperators_PL");

        const Int ldB_in  = int_cast<Int>(ldB_in_ );
        const Int ldC_out = int_cast<Int>(ldC_out_);
        
        RequireBuffers( wave_count );
        
        Scalar::Complex<C_ext> addTo = Scalar::Zero<C_ext>;

//        PrintBooleans();
        
        if( Re_single_layer || Im_single_layer ||
            Re_double_layer || Im_double_layer ||
            Re_adjdbl_layer || Im_adjdbl_layer
        )
        {
            AvOp.Dot(
                Scalar::One<Complex>,  B_in,  ldB_in,
                Scalar::Zero<Complex>, B_ptr, ldB,
                wave_count
            );

            B_loaded = true;
            ModifiedB( int_cast<LInt>(wave_count) * int_cast<LInt>(ldB) );

            BoundaryOperatorKernel_C( kappa );
            C_loaded = true;

            // TODO: Apply diagonal part of single layer boundary operator.
            
            // TODO: Is there some diagonal part of double layer and adjdbl boundary operator?
            
            AvOpTransp.Dot(
                alpha, C_ptr, ldC,
                beta,  C_out, ldC_out,
                wave_count
            );
            
            addTo = Scalar::One<C_ext>;
        }
        
        if( Re_mass_matrix || Im_mass_matrix )
        {
            if( wave_chunk_count >= 1 )
            {
                for( Int k = 0; k < wave_chunk_count-1; ++k )
                {
                    const Scalar::Complex<C_ext> factor = alpha * static_cast<C_ext>(c[k][0]);
                    
                    Mass.Dot(
                        factor, &B_in [wave_chunk_size * k], ldB_in,
                        addTo,  &C_out[wave_chunk_size * k], ldC_out,
                        wave_chunk_size
                    );
                }
                {
                    const Int k = wave_chunk_count-1;
                    const Scalar::Complex<C_ext> factor = alpha * static_cast<C_ext>(c[k][0]);
                    
                    Mass.Dot(
                        factor, &B_in [wave_chunk_size * k], ldB_in,
                        addTo,  &C_out[wave_chunk_size * k], ldC_out,
                        wave_count % wave_chunk_size
                    );
                }
            }
        }

        ptoc(ClassName()+"::ApplyBoundaryOperators_PL");
    }
