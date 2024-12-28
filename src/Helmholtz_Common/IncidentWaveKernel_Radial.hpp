public:
    // kernel code for the incident wave in the classic inverse obstacle scattering problem with the Helmholtz equation
    // calculates element wise (c0 + i*k*c1*<d,n>)*e^(i*k*<d,x>) for wave number k, wave vector d, mid points(!) x, and simplex normals n

    void IncidentWaveKernel_Radial(
        const WaveNumberContainer_T  & kappa_,
        const CoefficientContainer_T & c_ ,
        const Real    * point_sources, 
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
                        const Real w_vec[3] = {
                            point_sources[3*j + 0],
                            point_sources[3*j + 1],
                            point_sources[3*j + 2]
                        };
                        
                        const Real delta [3] = {
                            mid_points_ptr[4*i + 0] - w_vec[0],
                            mid_points_ptr[4*i + 1] - w_vec[1],
                            mid_points_ptr[4*i + 2] - w_vec[2]
                        };
                    
                        const Real R2 = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
                        
                        Real R = std::sqrt(R2);
                        Real one_over_R = 1/R;
                        Real one_over_R2 = 1/R2;

                        Complex exponent(0.0f, Kappa * R);

                        Real dot_norm = normals_ptr[4*i + 0] * delta[0] 
                                      + normals_ptr[4*i + 1] * delta[1]
                                      + normals_ptr[4*i + 2] * delta[2];

                        Complex A(one_over_four_pi * one_over_R);
                        Complex B( -dot_norm *  one_over_R2, dot_norm * Kappa * one_over_R);

                        Complex factor = Coeff[0] * A + Coeff[1] * A * B;

                        C[wave_count * i + wave_chunk_size * chunk + j] = factor * std::exp(exponent);
                    }
                }
            },
            simplex_count, CPU_thread_count
        );
    }
    