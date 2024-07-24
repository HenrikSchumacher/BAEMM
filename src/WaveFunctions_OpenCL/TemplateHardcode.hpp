public:

    // hardcoded template cases for wave_count and wave_chunk_count for loop unrolling in IncidentWaveKernel
    void IncidentWaveKernel_Plane_C_temp(
        const WaveNumberContainer_T  & kappa_,
        const CoefficientContainer_T & c_,
        const Real * incident_directions,
        Complex* C
    )
    {
        switch(wave_count)
        {
            case 1:
            {
                incidentWaveKernel_Plane_C<1,1>(kappa_,c_,incident_directions,C);
                break;
            }
            case 4:
            {
                switch( wave_chunk_count )
                {
                    case 1:
                    {
                        incidentWaveKernel_Plane_C<1,4>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 2:
                    {
                        incidentWaveKernel_Plane_C<2,2>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 4:
                    {
                        incidentWaveKernel_Plane_C<4,1>(kappa_,c_,incident_directions,C);
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
                        incidentWaveKernel_Plane_C<1,8>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 2:
                    {
                        incidentWaveKernel_Plane_C<2,4>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 4:
                    {
                        incidentWaveKernel_Plane_C<4,2>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 8:
                    {
                        incidentWaveKernel_Plane_C<8,1>(kappa_,c_,incident_directions,C);
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
                        incidentWaveKernel_Plane_C<1,16>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 2:
                    {
                        incidentWaveKernel_Plane_C<2,8>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 4:
                    {
                        incidentWaveKernel_Plane_C<4,4>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 8:
                    {
                        incidentWaveKernel_Plane_C<8,2>(kappa_,c_,incident_directions,C);
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
                        incidentWaveKernel_Plane_C<1,32>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 2:
                    {
                        incidentWaveKernel_Plane_C<2,16>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 4:
                    {
                        incidentWaveKernel_Plane_C<4,8>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 8:
                    {
                        incidentWaveKernel_Plane_C<8,4>(kappa_,c_,incident_directions,C);
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
                        incidentWaveKernel_Plane_C<1,64>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 2:
                    {
                        incidentWaveKernel_Plane_C<2,32>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 4:
                    {
                        incidentWaveKernel_Plane_C<4,16>(kappa_,c_,incident_directions,C);
                        break;
                    }
                    case 8:
                    {
                        incidentWaveKernel_Plane_C<8,8>(kappa_,c_,incident_directions,C);
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
    }
