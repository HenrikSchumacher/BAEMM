public:

    template<typename R_ext, typename C_ext, typename I_ext>
    void CreateHerglotzWave_PL(
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
        
        R_ext factor = static_cast<R_ext>(four_pi / meas_count);
        LoadCoefficients(kappa_,coeff_0,coeff_1,coeff_2,coeff_3,wave_count_,wave_chunk_size_, factor);
        
        CreateHerglotzWave_PL( alpha, B_in, ldB_in, beta, C_out, ldC_out );
    }

    template<typename R_ext, typename C_ext, typename I_ext>
    void CreateHerglotzWave_PL(
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
        
        R_ext factor = static_cast<R_ext>(four_pi / meas_count);
        LoadParameters(kappa_list, coeff_list, wave_count_, wave_chunk_size_, factor);
        
        CreateHerglotzWave_PL( alpha, B_in, ldB_in, beta, C_out, ldC_out );
    }


    // creates a herglotz wave with kernel conj(B_in) in the WEAK FORM
    template<typename C_ext, typename I_ext>
    void CreateHerglotzWave_PL(
        const C_ext alpha, cptr<C_ext> B_in,  const I_ext ldB_in_,
        const C_ext beta,  mptr<C_ext> C_out, const I_ext ldC_out_
    )
    {
        // The same as above, but assumes that
        CheckInteger<I_ext>();
        CheckComplex<C_ext>();

        ptic(ClassName()+"::CreateHerglotzWave_PL");
        
        if( wave_chunk_count < 1 )
        {
            ptoc(ClassName()+"::CreateHerglotzWave_PL");
            return;
        }
        
        const Int ldB_in  = int_cast<Int>(ldB_in_);
        const Int ldC_out = int_cast<Int>(ldC_out_);
        
        RequireBuffersHerglotzWave( wave_count );
        
        if( Re_single_layer || Im_single_layer || Re_double_layer || Im_double_layer )
        {
            copy_matrix<VarSize,VarSize,Parallel>
            (
                B_in,  ldB_in,
                B_ptr, wave_count,
                meas_count, wave_count, CPU_thread_count
            );

            ModifiedB();
            C_loaded = true;    

            HerglotzWaveKernel( kappa, c );
        
            // use transpose averaging operator to get from PC to PL boundary functions
            AvOpTransp.Dot(
                alpha, C_ptr, ldC,
                beta,  C_out, ldC_out,
                wave_count
            );
        }
        ptoc(ClassName()+"::CreateHerglotzWave_PL");
    }
