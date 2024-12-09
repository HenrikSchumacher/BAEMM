public:

    /* Kernel code for the incident wave in the classic inverse obstacle scattering problem with the Helmholtz equation
    *calculates element wise (c0 + i*k*c1*<d,n>)*e^(i*k*<d,x>) for wave number k, wave vector d, mid points(!) x, and simplex normals n
    */
    void IncidentWaveKernel_Plane(
        const WaveNumberContainer_T  & kappa_,
        const CoefficientContainer_T & c_,
        const Real    * incident_directions,
              Complex * C
    )
    {
        ptic( ClassName() + "::IncidentWaveKernel_Plane");
        
        switch(wave_count)
        {
            case 1:
            {
                incidentWaveKernel_Plane<1,1>(kappa_,c_,incident_directions,C);
                break;
            }
            case 4:
            {
                switch( wave_chunk_count )
                {
                    case 1:
                    {
                        incidentWaveKernel_Plane<1,4>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 2:
                    {
                        incidentWaveKernel_Plane<2,2>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 4:
                    {
                        incidentWaveKernel_Plane<4,1>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    default:
                    {
                        eprint(ClassName()+"::IncidentWaveKernel: wave_chunk_count must be a power of 2 that is smaller or equal to 8.");
                        break;
                    }
                }
                break;
            }
            case 8:
            {
                switch( wave_chunk_count )
                {
                    case 1:
                    {
                        incidentWaveKernel_Plane<1,8>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 2:
                    {
                        incidentWaveKernel_Plane<2,4>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 4:
                    {
                        incidentWaveKernel_Plane<4,2>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 8:
                    {
                        incidentWaveKernel_Plane<8,1>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    default:
                    {
                        eprint(ClassName()+"::IncidentWaveKernel: wave_chunk_count must be a power of 2 that is smaller or equal to 8.");
                        break;
                    }
                }
                break;
            }
            case 16:
            {
                switch( wave_chunk_count )
                {
                    case 1:
                    {
                        incidentWaveKernel_Plane<1,16>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 2:
                    {
                        incidentWaveKernel_Plane<2,8>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 4:
                    {
                        incidentWaveKernel_Plane<4,4>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 8:
                    {
                        incidentWaveKernel_Plane<8,2>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    default:
                    {
                        eprint(ClassName()+"::IncidentWaveKernel: wave_chunk_count must be a power of 2 that is smaller or equal to 8.");
                        break;
                    }
                }
                break;
            }
            case 32:
            {
                switch( wave_chunk_count )
                {
                    case 1:
                    {
                        incidentWaveKernel_Plane<1,32>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 2:
                    {
                        incidentWaveKernel_Plane<2,16>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 4:
                    {
                        incidentWaveKernel_Plane<4,8>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 8:
                    {
                        incidentWaveKernel_Plane<8,4>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    default:
                    {
                        eprint(ClassName()+"::IncidentWaveKernel: wave_chunk_count must be a power of 2 that is smaller or equal to 8.");
                        break;
                    }
                }
                break;
            }
            case 64:
            {
                switch( wave_chunk_count )
                {
                    case 1:
                    {
                        incidentWaveKernel_Plane<1,64>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 2:
                    {
                        incidentWaveKernel_Plane<2,32>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 4:
                    {
                        incidentWaveKernel_Plane<4,16>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 8:
                    {
                        incidentWaveKernel_Plane<8,8>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    default:
                    {
                        eprint(ClassName()+"::IncidentWaveKernel: wave_chunk_count must be a power of 2 that is smaller or equal to 8.");
                        break;
                    }
                }
                break;
            }
            default:
            {
                eprint(ClassName()+"::IncidentWaveKernel: wave_count must be a power of 2 that is bigger than 8 and smaller or equal to 64.");
                break;
            }
        }
        
        ptoc( ClassName() + "::IncidentWaveKernel_Plane");
    }

private:

    template<Int wave_chunk_count,Int wave_chunk_size>
    void incidentWaveKernel_Plane(
        const WaveNumberContainer_T  & kappa_,
        const CoefficientContainer_T & c_ ,
        const Real    * incident_directions,
              Complex * C
    )
    {
        ParallelDo(
            [=,this]( const Int i )
            {
                for( Int chunk = 0; chunk < wave_chunk_count; ++chunk )
                {
                    Real Kappa = kappa_[chunk];
                    Complex Coeff[2] = {c_[chunk][1],c_[chunk][2]};

                    for (Int j = 0; j < wave_chunk_size; ++j )
                    {
                        Real w_vec[3] = {
                            incident_directions[3*j + 0],
                            incident_directions[3*j + 1],
                            incident_directions[3*j + 2]
                        };

                        Real dot_mid = mid_points_ptr[4*i + 0] * w_vec[0]
                                     + mid_points_ptr[4*i + 1] * w_vec[1]
                                     + mid_points_ptr[4*i + 2] * w_vec[2];
                        
                        Complex exponent(0.0f, Kappa * dot_mid);

                        Real dot_norm = normals_ptr[4*i + 0] * w_vec[0]
                                      + normals_ptr[4*i + 1] * w_vec[1]
                                      + normals_ptr[4*i + 2] * w_vec[2];
                        
                        Complex dexponent(0.0f, Kappa * dot_norm);

                        Complex factor = Coeff[0] + Coeff[1] * dexponent;

                        C[wave_count * i + wave_chunk_size * chunk + j] = factor * std::exp(exponent);
                    }
                }
            },
            simplex_count, CPU_thread_count
        );
    }
