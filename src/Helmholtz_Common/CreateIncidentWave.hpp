public:

    template<typename R_ext, typename C_ext, typename I_ext>
    void CreateIncidentWave_PL(
        const C_ext alpha, ptr<R_ext> incident_directions,  const I_ext inc_count,
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
        
        if (inc_count != wave_chunk_size)
        {
            if (inc_count > wave_chunk_size)
            {
                print("number of inident waves has to match the size of the wave chunks. excess incident directions will be ignored");
            }
            if (inc_count < wave_chunk_size)
            {
                eprint("error: number of inident waves has to match the size of the wave chunks");
            }
        }

        LoadCoefficients(kappa_,coeff_0,coeff_1,coeff_2,coeff_3,wave_count_,wave_chunk_size_,four_pi);
        
        CreateIncidentWave_PL( alpha, incident_directions, beta, C_out, ldC_out );
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
    void CreateIncidentWave_PL(
        const C_ext alpha, ptr<R_ext> incident_directions,  const I_ext inc_count,
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
        
        if (inc_count != wave_chunk_size)
        {
            if (inc_count > wave_chunk_size)
            {
                print("number of inident waves has to match the size of the wave chunks. excess incident directions will be ignored");
            }
            if (inc_count < wave_chunk_size)
            {
                eprint("error: number of inident waves has to match the size of the wave chunks");
            }
        }

        LoadParameters(kappa_list, coeff_list, wave_count_, wave_chunk_size_,four_pi);
        
        CreateIncidentWave_PL( alpha, incident_directions, beta, C_out, ldC_out );
    }

    template<typename R_ext,typename C_ext, typename I_ext>
    void CreateIncidentWave_PL(
        const C_ext alpha, ptr<R_ext> incident_directions,
        const C_ext beta,  mut<C_ext> C_out, const I_ext ldC_out_
    )
    {
        // The same as above, but assumes that
        ASSERT_INT(I_ext);
        ASSERT_COMPLEX(C_ext);

        ptic(ClassName()+"::CreateIncidentWave_PL");
        
        if( wave_chunk_count < 1 )
        {
            ptoc(ClassName()+"::CreateIncidentWave_PL");
            return;
        }
        
        const Int ldC = wave_count;
        const Int ldC_out = int_cast<Int>(ldC_out_);

        if( Re_single_layer || Im_single_layer ||
            Re_double_layer || Im_double_layer
        )
        {   
            Complex* C = (Complex*)malloc(ldC * simplex_count * sizeof(Complex));
            // Apply off-diagonal part of integral operators.

            IncidentWaveKernel_C( kappa, c , reinterpret_cast<const Real*>(incident_directions),C);
           
            AvOpTransp.Dot(
                alpha, C, ldC,
                beta,  C_out, ldC_out,
                wave_count
            );

            free(C);
        }
        ptoc(ClassName()+"::CreateIncidentWave_PL");
    }
