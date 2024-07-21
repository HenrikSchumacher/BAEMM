public:

    template<typename R_ext, typename C_ext, typename I_ext>
    void CreateIncidentWave_PL(
        const C_ext alpha, cptr<R_ext> incident_directions,  const I_ext inc_count,
        const C_ext beta,  mptr<C_ext> C_out, const I_ext ldC_out,
        const R_ext kappa_,
        const C_ext coeff_0,
        const C_ext coeff_1,
        const C_ext coeff_2,
        const C_ext coeff_3,
        const I_ext wave_count_,
        const I_ext wave_chunk_size_,
        WaveType type = WaveType::Plane
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
        
        if (inc_count != wave_chunk_size_)
        {
            if (inc_count > wave_chunk_size_)
            {
                print("number of inident waves has to match the size of the wave chunks. excess incident directions will be ignored");
            }
            if (inc_count < wave_chunk_size_)
            {
                eprint("error: number of inident waves has to match the size of the wave chunks");
            }
        }

        LoadCoefficients(kappa_,coeff_0,coeff_1,coeff_2,coeff_3,wave_count_,wave_chunk_size_,static_cast<R_ext>(four_pi));

        CreateIncidentWave_PL( alpha, incident_directions, beta, C_out, ldC_out, type );
    }

    template<typename R_ext, typename C_ext, typename I_ext>
    void CreateIncidentWave_PL(
        const C_ext alpha, cptr<R_ext> incident_directions,  const I_ext inc_count,
        const C_ext beta,  mptr<C_ext> C_out, const I_ext ldC_out,
        const R_ext * kappa_list,
        const C_ext * coeff_list,
        const I_ext wave_count_,
        const I_ext wave_chunk_size_,
        WaveType type = WaveType::Plane
    )
    {
        //  The same as above, but with several wave numbers kappa_list and several coefficients.

        ASSERT_INT(I_ext);
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);
        
        if (inc_count != wave_chunk_size_)
        {
            if (inc_count > wave_chunk_size_)
            {
                print("number of inident waves has to match the size of the wave chunks. excess incident directions will be ignored");
            }
            if (inc_count < wave_chunk_size_)
            {
                eprint("error: number of inident waves has to match the size of the wave chunks");
            }
        }

        LoadParameters(kappa_list, coeff_list, wave_count_, wave_chunk_size_,static_cast<R_ext>(four_pi));

        CreateIncidentWave_PL( alpha, incident_directions, beta, C_out, ldC_out, type );
    }

// creates wave_count incident waves for the scattering problem in the WEAK FORM
    template<typename R_ext,typename C_ext, typename I_ext>
    void CreateIncidentWave_PL(
        const C_ext alpha, cptr<R_ext> incident_directions_,
        const C_ext beta,  mptr<C_ext> C_out, const I_ext ldC_out_,
        WaveType type
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
            Tensor2<Real,Int> incident_directions ( wave_chunk_size, 3 );
            
            Tensor2<Complex,Int> C ( simplex_count, ldC );
            
            incident_directions.Read(incident_directions_);
            
//            type_cast(incident_directions, incident_directions_, 3 * wave_chunk_size, 1);

            if (type == WaveType::Plane)
            {
                IncidentWaveKernel_Plane_C( kappa, c, incident_directions.data(), C.data() );
            }
            else
            {
                IncidentWaveKernel_Radial_C( kappa, c, incident_directions.data(), C.data() );
            }

            // use transpose averaging operator to get from PC to PL boundary functions
            AvOpTransp.Dot(
                alpha, C.data(), ldC,
                beta,  C_out,    ldC_out,
                wave_count
            );
        }
        ptoc(ClassName()+"::CreateIncidentWave_PL");
    }
