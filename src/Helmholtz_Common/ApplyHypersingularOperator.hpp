public:

    template<typename R_ext, typename C_ext, typename I_ext>
    void ApplyHypersingularOperator_PL(
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
        // The operator A is the hypersingular operator.
        
        ASSERT_INT(I_ext);
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);
        
        LoadParameters3(kappa_,coeff_0,coeff_1,coeff_2,coeff_3,wave_count_,wave_chunk_size_);
        
        ApplyHypersingularOperator_PL( alpha, B_in, ldB_in, beta, C_out, ldC_out );
    }


    template<typename R_ext, typename C_ext, typename I_ext>
    void ApplyHypersingularOperator(
        const C_ext alpha, cptr<C_ext> B_in,  const I_ext ldB_in,
        const C_ext beta,  mptr<C_ext> C_out, const I_ext ldC_out,
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
        
        LoadParameters3(kappa_list, coeff_list, wave_count_, wave_chunk_size_);
        
        ApplyHypersingularOperator_PL( alpha, B_in, ldB_in, beta, C_out, ldC_out );
    }

    template<typename C_ext, typename I_ext>
    void ApplyHypersingularOperator_PL(
        const C_ext alpha, cptr<C_ext> B_in,  const I_ext ldB_in_,
        const C_ext beta,  mptr<C_ext> C_out, const I_ext ldC_out_
    )
    {
        if( wave_chunk_count < 1 )
        {
            return;
        }
        
        // The same as above, but assumes that
        ASSERT_INT(I_ext);
        ASSERT_COMPLEX(C_ext);
        
        ptic(ClassName()+"::ApplyHypersingularOperator");

        const Int ldB_in  = int_cast<Int>(ldB_in_ );
        const Int ldC_out = int_cast<Int>(ldC_out_);
        
        Scalar::Complex<C_ext> addTo = Scalar::Zero<C_ext>;

//        PrintBooleans();
        
        const Int wave_count_3       = 3 * wave_count;
        const Int wave_chunk_count_3 = 3 * wave_chunk_count;
        
        RequireBuffers( wave_count_3 );
        
        CurlOp.Dot(
            Scalar::One<Complex>,  B_in,  ldB_in,
            Scalar::Zero<Complex>, B_ptr, ldB,
            wave_count
        );

        ModifiedB();
        C_loaded = true;
        
        // TODO: Check whether kappa3 and c3 are set up correctly.
        
        BoundaryOperatorKernel_C( kappa3, c3 );
        
        addTo = Scalar::One<C_ext>;
        
        // We have tp apply also the diagonal of the single layer boundary operator.
        
        // TODO: Take ApplySingleLayerDiagonal.hpp and adapt it.
        // Maybe this does the job already:
        // TODO: Test this!
        ApplySingleLayerDiagonal( kappa3, c3 );
        
        CurlOpTransp.Dot(
            alpha, C_ptr, ldC,
            beta,  C_out, ldC_out,
            wave_count
        );

        ptoc(ClassName()+"::ApplyHypersingularOperator_PL");
    }
