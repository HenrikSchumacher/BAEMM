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
        
        LoadParameters(kappa_,coeff_0,coeff_1,coeff_2,coeff_3,wave_count_,wave_chunk_size_);
        
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
//        LoadParameters(kappa_list.data(),coeff_list.data(),wave_count_,wave_chunk_size_);
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
        
        LoadParameters(kappa_list, coeff_list, wave_count_, wave_chunk_size_);
        
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
        
        Scalar::Complex<C_ext> addTo = Scalar::Zero<C_ext>;

//        PrintBooleans();
        
        RequireBuffers( wave_count );
        
        // Apply off-diagonal part of integral operators.
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

            BoundaryOperatorKernel_C( kappa, c );
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
        
        
        // Apply diagonal of single layer boundary operators.
        if( (Re_single_layer || Im_single_layer) && (wave_chunk_count >= 1) )
        {
            Tensor1<Complex,Int> I_kappa ( wave_chunk_count );
            
            const Int border_size = wave_count - wave_chunk_size * (wave_chunk_count-1);
            
            for( Int chunk = 0; chunk < wave_chunk_count; ++chunk )
            {
                I_kappa[chunk] = Complex( - imag(kappa[chunk]), real(kappa[chunk]) );
            }

            #pragma omp parallel for num_threads( OMP_thread_count ) schedule( static )
            for( Int i = 0; i < simplex_count; ++i )
            {
                for( Int chunk = 0; chunk < wave_chunk_count-1; ++chunk )
                {
                    const Int pos = ldB * i + wave_chunk_size * chunk;
                    
                    const Complex factor = c(chunk,1) * (single_diag_ptr[i] + I_kappa[chunk] * areas_ptr[i]);
                    
                    combine_buffers<Scalar::Flag::Generic, Scalar::Flag::Plus>(
                        factor,               &B_ptr[pos],
                        Scalar::One<Complex>, &C_ptr[pos],
                        wave_chunk_size
                    );
                }
                
                {
                    const Int chunk = wave_chunk_count - 1;
                    
                    const Int pos = ldB * i + wave_chunk_size * chunk;
                    
                    const Complex factor = c(chunk,1) * (single_diag_ptr[i] + I_kappa[chunk] * areas_ptr[i]);
                    
                    combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Plus>(
                        factor,               &B_ptr[pos],
                        Scalar::One<Complex>, &C_ptr[pos],
                        border_size
                    );
                }
            }
        }
        
        
        // Apply mass matrix.
        if( Re_mass_matrix || Im_mass_matrix )
        {
            if( wave_chunk_count >= 1 )
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
        }
        

        ptoc(ClassName()+"::ApplyBoundaryOperators_PL");
    }
