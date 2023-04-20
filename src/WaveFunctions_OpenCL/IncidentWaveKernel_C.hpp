public:

#include "TemplateHardcode.hpp"

    void IncidentWaveKernel_C(
        const WaveNumberContainer_T  & kappa_,
        const CoefficientContainer_T & c_,
        const Real * incident_directions,
        Complex* C
    )
    {
        IncidentWaveKernel_C_temp(
        kappa_,c_,incident_directions,C
        );
    }


    template<Int wave_chunk_count,Int wave_chunk_size>
    void incidentWaveKernel_C(
        const WaveNumberContainer_T  & kappa_,
        const CoefficientContainer_T & c_ ,
        const Real * incident_directions, 
        Complex* C
    )
    {
        Int i,chunk,j;

        #pragma omp parallel for num_threads( OMP_thread_count ) schedule( static ) private(i) private(chunk) private(j)
        for( i = 0; i < simplex_count; ++i )
        {
            LOOP_UNROLL_FULL
            for( chunk = 0; chunk < wave_chunk_count; ++chunk )
            {
                Real Kappa = kappa_[chunk];
                Complex Coeff[2] = {c_[chunk][1],c_[chunk][2]};

                LOOP_UNROLL_FULL
                for ( j = 0; j < wave_chunk_size; ++j )
                {
                    Real w_vec[3] = { incident_directions[3*j + 0], incident_directions[3*j + 1]
                                    , incident_directions[3*j + 2] };
                
                    Real dot_mid = mid_points_ptr[4*i + 0] * w_vec[0] + mid_points_ptr[4*i + 1] * w_vec[1]
                    + mid_points_ptr[4*i + 2] * w_vec[2];
                    Complex exponent(0.0f, Kappa * dot_mid);

                    Real dot_norm = normals_ptr[4*i + 0] * w_vec[0] + normals_ptr[4*i + 1] * w_vec[1]
                    + normals_ptr[4*i + 2] * w_vec[2];
                    Complex dexponent(0.0f, Kappa * dot_norm);

                    Complex factor = Coeff[0] + Coeff[1] * dexponent;

                    C[wave_count * i + wave_chunk_size * chunk + j] = factor * std::exp(exponent);
                }
            }
        }
    }


    // template<Int wave_chunk_count, Int wave_chunk_size>
    // void incidentWaveKernel_C(
    //     const WaveNumberContainer_T  & kappa_,
    //     const CoefficientContainer_T & c_ ,
    //     const Real * incident_directions,
    //     Complex * C 
    // )
    // {
    //     #pragma omp parallel for num_threads( OMP_thread_count ) schedule( static )
    //     for( Int i = 0; i < simplex_count; ++i )
    //     {
    //         LOOP_UNROLL_FULL
    //         for ( Int j = 0; j < wave_chunk_size; ++j )
    //         {
    //             Real w_vec[3] = { incident_directions[3*j + 0], incident_directions[3*j + 1]
    //                                 , incident_directions[3*j + 2] };

    //             Real dot_mid = mid_points_ptr[4*i + 0] * w_vec[0] + mid_points_ptr[4*i + 1] * w_vec[1]
    //                         + mid_points_ptr[4*i + 2] * w_vec[2];

    //             Real dot_norm = normals_ptr[4*i + 0] * w_vec[0] + normals_ptr[4*i + 1] * w_vec[1]
    //                         + normals_ptr[4*i + 2] * w_vec[2];
                                
    //             LOOP_UNROLL_FULL
    //             for( Int chunk = 0; chunk < wave_chunk_count; ++chunk )
    //             {
    //                 Real Kappa = kappa_[chunk];
    //                 Complex Coeff[2] = {c_[chunk][1],c_[chunk][2]};

    //                 Complex exponent(0.0f, Kappa * dot_mid);
    //                 Complex dexponent(0.0f, Kappa * dot_norm);

    //                 Complex factor = Coeff[0] + Coeff[1] * dexponent;

    //                 C[wave_count * i + wave_chunk_size * chunk + j] = factor * std::exp(exponent);
    //             }
    //         }
    //     }
    // }