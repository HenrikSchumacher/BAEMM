public:

// TODO: Is this one ever called? If not, we should discard it.
//
//    template<typename I_ext, typename R_ext, typename C_ext>
//    void CreateIncidentWave_PL(
//        const C_ext alpha, cptr<R_ext> incident_directions,  const I_ext inc_count,
//        const C_ext beta,  mptr<C_ext> C_out,                const I_ext ldC_out,
//        const R_ext kappa_,
//        const C_ext coeff_0,
//        const C_ext coeff_1,
//        const C_ext coeff_2,
//        const C_ext coeff_3,
//        const I_ext wave_count_,
//        const I_ext wave_chunk_size_,
//        WaveType type = WaveType::Plane
//    )
//    {
//        // Computes
//        //
//        //     C_out = alpha * A * B_in + beta * C_out,
//        //
//        // where B_in and C_out our are matrices of size vertex_count x wave_count_ and
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
//        
//        CheckInteger<I_ext>();
//        CheckScalars<R_ext,C_ext>();
//
//        
//        if (inc_count != wave_chunk_size_)
//        {
//            if (inc_count > wave_chunk_size_)
//            {
//                eprint("number of incident waves has to match the size of the wave chunks. excess incident directions will be ignored");
//            }
//            if (inc_count < wave_chunk_size_)
//            {
//                eprint("number of inident waves has to match the size of the wave chunks");
//            }
//        }
//
//        LoadCoefficients(kappa_,coeff_0,coeff_1,coeff_2,coeff_3,wave_count_,wave_chunk_size_,static_cast<R_ext>(four_pi));
//
//        CreateIncidentWave_PL( alpha, incident_directions, beta, C_out, ldC_out, type );
//    }

    template<typename R_ext, typename C_ext, typename I_ext>
    void CreateIncidentWave_PL(
        const C_ext alpha, cptr<R_ext> incident_directions, const I_ext inc_count,
        const C_ext beta,  mptr<C_ext> C_out,               const I_ext ldC_out,
        const R_ext * kappa_list,
        const C_ext * coeff_list,
        const I_ext wave_count_,
        const I_ext wave_chunk_size_,
        WaveType type = WaveType::Plane
    )
    {
        CheckInteger<I_ext>();
        CheckScalars<R_ext,C_ext>();
        
        if (inc_count != wave_chunk_size_)
        {
            if (inc_count > wave_chunk_size_)
            {
                wprint(ClassName() + "::CreateIncidentWave_PL : Number of incident waves has to match the size of the wave chunks. Excess incident directions will be ignored");
            }
            if (inc_count < wave_chunk_size_)
            {
                eprint(ClassName() + "::CreateIncidentWave_PL : Number of incident waves has to match the size of the wave chunks");
            }
        }

        LoadParameters(kappa_list, coeff_list, wave_count_, wave_chunk_size_,static_cast<R_ext>(four_pi));

        CreateIncidentWave_PL( alpha, incident_directions, beta, C_out, ldC_out, type );
    }

    // Creates wave_count incident waves for the scattering problem in the WEAK FORM
    template<typename R_ext,typename C_ext, typename I_ext>
    void CreateIncidentWave_PL(
        const C_ext alpha, cptr<R_ext> incident_directions_,
        const C_ext beta,  mptr<C_ext> C_out, const I_ext ldC_out_,
        WaveType type
    )
    {
        // Computes
        //
        //     C_out = alpha * A * B_in + beta * C_out,
        //
        // where B_in and C_out our are matrices of size vertex_count x wave_count_ and
        // represent the vertex values of  wave_count_ piecewise-linear functions.
        // The operator A is a linear combination of several operators, depending on kappa:
        //
        // A =   coeff_0 * MassMatrix
        //     + coeff_1 * SingleLayerOp[kappa]
        //     + coeff_2 * DoubleLayerOp[kappa]
        //     + coeff_3 * AdjDblLayerOp[kappa]
        //
        
        CheckInteger<I_ext>();
        CheckScalars<R_ext,C_ext>();

        std::string tag = ClassName() + "::CreateIncidentWave_PL"
            + "<" + TypeName<I_ext>
            + "," + TypeName<R_ext>
            + "m" + TypeName<C_ext>
            + ">";
        
        ptic(tag);
        
        if( wave_chunk_count < 1 )
        {
            ptoc(tag);
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

            if (type == WaveType::Plane)
            {
                IncidentWaveKernel_Plane( kappa, c, incident_directions.data(), C.data() );
            }
            else
            {
                IncidentWaveKernel_Radial( kappa, c, incident_directions.data(), C.data() );
            }

            // Use transpose averaging operator to get from PC to PL boundary functions
            AvOpTransp.Dot(
                alpha, C.data(), ldC,
                beta,  C_out,    ldC_out,
                wave_count
            );
        }
        
        ptoc(tag);
    }
