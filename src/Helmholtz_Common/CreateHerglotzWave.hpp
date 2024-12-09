public:
    /**
     * Creates wave_count herglotz waves with some parsed kernel in the WEAK FORM (in the sense of piecewise linear continuous functions). 
     * Computes C_out = alpha * A * B_in + beta * C_out,
     *
     * where B_in and C_out out are matrices of size vertex_count x wave_count_ and
     * represent the vertex values of  wave_count_ piecewise-linear functions.
    * The operator A is a linear combination of several operators, depending on kappa:
     *
     * A = coeff_(-,1) * HerglotzWave
     *     + coeff(-,2) * dHerglotzWave/dn
     * 
     * The canonical choices would be alpha = 1 and beta = 0.
     * 
     * @tparam I_ext External integer type.
     * @tparam R_ext External Real type.
     * @tparam C_ext External Complex type.
     * @param B_in Input array of size meas_count*wave_count_ - Herglotz wave kernel.
     * @param ldB_in Leading dimension of input. Usually wave_count_.
     * @param C_out Output array.
     * @param ldC_out Leading dimension of output. Usually wave_count_.
     * @param kappa_list An (wave_count_/wave_chunk_size_) x 1 Complex array representing the wavenumbers.
     * @param coeff_list An (wave_count_/wave_chunk_size_) x 4 Complex array representing the used combination of Dirichlet- and Neumann-data (by the second and third columns).
     */
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


    /**
     * @brief Creates wave_count herglotz waves with some parsed kernel in the WEAK FORM (in the sense of piecewise linear continuous functions) under the assumption that Assumes that 'LoadParameters' has been called before.
     */
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
