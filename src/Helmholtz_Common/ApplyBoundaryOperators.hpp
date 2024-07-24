public:

    template<typename R_ext, typename C_ext, typename I_ext>
    void ApplyBoundaryOperators_PL(
        const C_ext alpha, cptr<C_ext> B_in,  const I_ext ldB_in,
        const C_ext beta,  mptr<C_ext> C_out, const I_ext ldC_out,
        cptr<R_ext> kappa_list,
        cptr<C_ext> coeff_list,
        const I_ext wave_count_,
        const I_ext wave_chunk_size_
    )
    {
        ASSERT_INT(I_ext);
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);
        
        LoadBoundaryOperators_PL(kappa_list, coeff_list, wave_count_, wave_chunk_size_);
        
        ApplyBoundaryOperators_PL( alpha, B_in, ldB_in, beta, C_out, ldC_out );
        
        UnloadBoundaryOperators_PL();
    }

    template<typename R_ext, typename C_ext, typename I_ext>
    void LoadBoundaryOperators_PL(
        cptr<R_ext> kappa_list,
        cptr<C_ext> coeff_list,
        const I_ext wave_count_,
        const I_ext wave_chunk_size_
    )
    {
        LoadParameters(kappa_list, coeff_list, wave_count_, wave_chunk_size_);
        
        // Every Helmholtz_* class has to implement this.
        
        LoadBoundaryOperatorKernel_PL();
    }

    void UnloadBoundaryOperators_PL()
    {
        // Every Helmholtz_* class has to implement this.
        
        UnloadBoundaryOperatorKernel_PL();
    }

//    template<typename R_ext, typename C_ext, typename I_ext>
//    void ApplyBoundaryOperators_PL(
//        const C_ext alpha, cptr<C_ext> B_in,  const I_ext ldB_in,
//        const C_ext beta,  mptr<C_ext> C_out, const I_ext ldC_out,
//        const R_ext kappa_,
//        const C_ext coeff_0,
//        const C_ext coeff_1,
//        const C_ext coeff_2,
//        const C_ext coeff_3,
//        const I_ext wave_count_,
//        const I_ext wave_chunk_size_
//    )
//    {
//        // Computes
//        //
//        //     C_out = alpha * A * B_in + beta * C_out,
//        //
//        // where B_in and C_out out are matrices of size vertex_count x wave_count_ and
//        // represent the vertex values of  wave_count_ piecewise-linear functions.
//        // The operator A is a linear combination of several operators, depending on kappa:
//        //
//        // A =   coeff_0 * MassMatrix
//        //     + coeff_1 * SingleLayerOp[kappa]
//        //     + coeff_2 * DoubleLayerOp[kappa]
//        //     + coeff_3 * AdjDblLayerOp[kappa]
//        //
//        //
//        
//        ASSERT_INT(I_ext);
//        static_assert(FloatQ<R_ext>,"");
//        static_assert(FloatQ<C_ext>,"");
//        static_assert(Scalar::ComplexQ<C_ext>,"");
//        
//        ASSERT_REAL(R_ext);
//        ASSERT_COMPLEX(C_ext);
//        
//        LoadCoefficients(kappa_,coeff_0,coeff_1,coeff_2,coeff_3,wave_count_,wave_chunk_size_);
//        
//        ApplyBoundaryOperators_PL( alpha, B_in, ldB_in, beta, C_out, ldC_out );
//    }


    // Applies the boundary operators in the WEAK FORM to the input pointer.
    // Assumes that `LoadBoundaryOperators_PL` has been called before.
    template<typename C_ext, typename I_ext>
    void ApplyBoundaryOperators_PL(
        const C_ext alpha, cptr<C_ext> B_in,  const I_ext ldB_in_,
        const C_ext beta,  mptr<C_ext> C_out, const I_ext ldC_out_
    )
    {
        ASSERT_INT(I_ext);
        ASSERT_COMPLEX(C_ext);

        std::string tag = ClassName()+"::ApplyBoundaryOperators_PL";
        ptic(tag);
        
        if( wave_chunk_count < 1 )
        {
            ptoc(tag);
            return;
        }
    
        const Int ldB_in  = int_cast<Int>(ldB_in_ );
        const Int ldC_out = int_cast<Int>(ldC_out_);
        
        Scalar::Complex<C_ext> addTo = Scalar::Zero<C_ext>;

        RequireBuffers( wave_count );
        
        if( Re_single_layer || Im_single_layer ||
            Re_double_layer || Im_double_layer ||
            Re_adjdbl_layer || Im_adjdbl_layer
        )
        {
            // use averaging operator to get from PL to PC boundary functions
            AvOp.Dot(
                Scalar::One <Complex>, B_in,  ldB_in,
                Scalar::Zero<Complex>, B_ptr, ldB,
                wave_count
            );

            ModifiedB();
            C_loaded = true;

            // Apply off-diagonal part of integral operators.
            BoundaryOperatorKernel();

            // Apply diagonal of single layer operator.
            ApplySingleLayerDiagonal( kappa, c );
                        
            // TODO: Is there some diagonal part of double layer and adjdbl boundary operator?
            
            // use transpose averaging operator to get from PC to PL boundary functions
            AvOpTransp.Dot(
                alpha, C_ptr, ldC,
                beta,  C_out, ldC_out,
                wave_count
            );
            
            addTo = Scalar::One<C_ext>;
        }
        else
        {
            addTo = beta * Scalar::One<C_ext>;
        }
        
        // Apply mass matrix.
        if( Re_mass_matrix || Im_mass_matrix )
        {
            for( Int chunk = 0; chunk < wave_chunk_count - 1; ++chunk )
            {
                const Scalar::Complex<C_ext> factor
                        = alpha * static_cast<C_ext>(c[chunk][0]);
                
                Mass.Dot(
                    factor, &B_in [wave_chunk_size * chunk], ldB_in,
                    addTo,  &C_out[wave_chunk_size * chunk], ldC_out,
                    wave_chunk_size
                );
            }
            {
                const Int chunk = wave_chunk_count - 1;
                const Scalar::Complex<C_ext> factor
                        = alpha * static_cast<C_ext>(c[chunk][0]);
                
                Mass.Dot(
                    factor, &B_in [wave_chunk_size * chunk], ldB_in,
                    addTo,  &C_out[wave_chunk_size * chunk], ldC_out,
                    wave_count - wave_chunk_size*chunk
                );
            }
            
        }
        
        ptoc(tag);
    }
