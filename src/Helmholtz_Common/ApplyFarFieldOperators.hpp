public:

    template<Int WC = VarSize, typename R_ext, typename C_ext, typename I_ext>
    void ApplyFarFieldOperators_PL(
        const C_ext alpha, cptr<C_ext> B_in,  const I_ext ldB_in,
        const C_ext beta,  mptr<C_ext> C_out, const I_ext ldC_out,
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
        
        CheckInteger<I_ext>();
        CheckScalars<R_ext,C_ext>();
        
        LoadParameters(kappa_,coeff_0,coeff_1,coeff_2,coeff_3,wave_count_,wave_chunk_size_);
        
        ApplyFarFieldOperators_PL<WC>( alpha, B_in, ldB_in, beta, C_out, ldC_out );
    }

    template<Int WC = VarSize, typename R_ext, typename C_ext, typename I_ext>
    void ApplyFarFieldOperators_PL(
        const C_ext alpha, cptr<C_ext> B_in,  const I_ext ldB_in,
        const C_ext beta,  mptr<C_ext> C_out, const I_ext ldC_out,
        const R_ext * kappa_list,
        const C_ext * coeff_list,
        const I_ext wave_count_,
        const I_ext wave_chunk_size_
    )
    {
        //  The same as above, but with several wave numbers kappa_list and several coefficients.

        CheckInteger<I_ext>();
        CheckScalars<R_ext,C_ext>();
        
        LoadParameters(kappa_list, coeff_list, wave_count_, wave_chunk_size_);
        
        ApplyFarFieldOperators_PL<WC>( alpha, B_in, ldB_in, beta, C_out, ldC_out );
    }


    // Applies the boundary to farfield operators to the input pointer
    template<Int WC = VarSize, typename C_ext, typename I_ext>
    void ApplyFarFieldOperators_PL(
        const C_ext alpha, cptr<C_ext> B_in,  const I_ext ldB_in_,
        const C_ext beta,  mptr<C_ext> C_out, const I_ext ldC_out_
    )
    {
        CheckInteger<I_ext>();
        CheckComplex<C_ext>();

        std::string tag = ClassName()+"::ApplyFarFieldOperators_PL" +
            + "<" + ToString(VarSize)
            + "," + TypeName<C_ext>
            + "," + TypeName<I_ext>
            + ">";
        
        ptic(tag);
        
        if( wave_chunk_count < 1 )
        {
            ptoc(tag);
            return;
        }
        
        if( (WC > VarSize) && (WC != wave_count) )
        {
            eprint( tag + "WC != wave_count. Doing nothing." );
            ptoc(tag);
            return;
        }
            
        const Int ldB_in  = int_cast<Int>(ldB_in_ );
        const Int ldC_out = int_cast<Int>(ldC_out_);
        
        RequireBuffersFarField( wave_count );
        
        if( Re_single_layer || Im_single_layer || Re_double_layer || Im_double_layer )
        {
            // use averaging operator to get from PL to PC boundary functions
            AvOp.Dot<WC>(
                Scalar::One <Complex>, B_in,  ldB_in,
                Scalar::Zero<Complex>, B_ptr, ldB,
                wave_count
            );

            ModifiedB();
            C_loaded = true;
            
            // Apply off-diagonal part of integral operators.
            FarFieldOperatorKernel( kappa, c );
    
            // TODO: Is there some diagonal part of double layer and adjdbl boundary operator?

            // TODO: Are these dimensions correct?

//            combine_matrices(
//                alpha, C_ptr, ldC,
//                beta , C_out, ldC_out,
//                meas_count, wave_count, CPU_thread_count
//            );
            
            combine_matrices_auto<VarSize,WC,Parallel>(
                alpha, C_ptr, ldC,
                beta , C_out, ldC_out,
                meas_count, wave_count, CPU_thread_count
            );
        }
        ptoc(tag);
    }
