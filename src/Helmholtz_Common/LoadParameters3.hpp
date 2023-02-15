public:

    template<typename R_ext, typename C_ext, typename I_ext>
    void LoadParameters3(
        const R_ext * kappa_list,
        const C_ext * coeff_list,
        const I_ext wave_count_,
        const I_ext wave_chunk_size_
    )
    {
        ASSERT_INT(I_ext);
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);

        SetWaveCount    ( int_cast<Int>(wave_count_)      );
        SetWaveChunkSize( int_cast<Int>(wave_chunk_size_) );
        
        wave_chunk_count = GetWaveChunkCount(wave_count);
        
        if( c3.Dimension(0) != 3 * wave_chunk_count )
        {
            c3 = CoefficientContainer_T( 3 * wave_chunk_count, 4 );
        }
        
        if( kappa3.Dimension(0) != 3 * wave_chunk_count )
        {
            kappa3 = WaveNumberContainer_T( 3 * wave_chunk_count );
        }
        
        Re_mass_matrix  = false;
        Im_mass_matrix  = false;
        Re_single_layer = false;
        Im_single_layer = false;
        Re_double_layer = false;
        Im_double_layer = false;
        Re_adjdbl_layer = false;
        Im_adjdbl_layer = false;
        
        for( Int k = 0; k < wave_chunk_count; ++k )
        {
            const Real kappa_ = kappa_list[k];
            kappa[3*k+0] = kappa_;
            kappa[3*k+1] = kappa_;
            kappa[3*k+2] = kappa_;
            
            Complex z;
            
            // We have to process the coefficients anyways.
            // Hence we can already multiply by one_over_four_pi so that the kernels don't have to do that each time. (The performance gain is not measureable, though.)
            
            z = static_cast<Complex>(coeff_list[4*k+0]);
            c[3*k+0][0] = z;
            c[3*k+1][0] = z;
            c[3*k+2][0] = z;
            Re_mass_matrix  = Re_mass_matrix  || (real(z) != Scalar::Zero<Real>);
            Im_mass_matrix  = Im_mass_matrix  || (imag(z) != Scalar::Zero<Real>);
            
            z = static_cast<Complex>(coeff_list[4*k+1]) * one_over_four_pi;
            c[3*k+0][1] = z;
            c[3*k+1][1] = z;
            c[3*k+2][1] = z;
            Re_single_layer = Re_single_layer || (real(z) != Scalar::Zero<Real>);
            Im_single_layer = Im_single_layer || (imag(z) != Scalar::Zero<Real>);
            
            z = static_cast<Complex>(coeff_list[4*k+2]) * one_over_four_pi;
            c[3*k+0][2] = z;
            c[3*k+1][2] = z;
            c[3*k+2][2] = z;
            Re_double_layer = Re_double_layer || (real(z) != Scalar::Zero<Real>);
            Im_double_layer = Im_double_layer || (imag(z) != Scalar::Zero<Real>);
            
            z = static_cast<Complex>(coeff_list[4*k+3]) * one_over_four_pi;
            c[3*k+0][3] = z;
            c[3*k+1][3] = z;
            c[3*k+2][3] = z;
            Re_adjdbl_layer = Re_adjdbl_layer || (real(z) != Scalar::Zero<Real>);
            Im_adjdbl_layer = Im_adjdbl_layer || (imag(z) != Scalar::Zero<Real>);
        }
    }


    template<typename R_ext, typename C_ext, typename I_ext>
    void LoadCoefficients3(
        const R_ext kappa_,
        const C_ext coeff_0,
        const C_ext coeff_1,
        const C_ext coeff_2,
        const C_ext coeff_3,
        const Int wave_count_,
        const Int wave_chunk_size_
    )
    {
        ASSERT_INT(I_ext);
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);

        SetWaveCount    ( int_cast<Int>(wave_count_)      );
        SetWaveChunkSize( int_cast<Int>(wave_chunk_size_) );

        wave_chunk_count = GetWaveChunkCount(wave_count);

        if( c3.Dimension(0) != 3 * wave_chunk_count )
        {
            c3 = CoefficientContainer_T( 3 * wave_chunk_count, 4 );
        }

        if( kappa3.Dimension(0) != 3 * wave_chunk_count )
        {
            kappa3 = WaveNumberContainer_T( 3 * wave_chunk_count, static_cast<Real>(kappa_) );
        }

        // We have to process the coefficients anyways.
        // Hence we can already multiply by one_over_four_pi so that the kernels don't have to do that each time. (The performance gain is not measureable, though.)
        Complex z [4] {
            static_cast<Complex>(coeff_0),
            static_cast<Complex>(coeff_1) * one_over_four_pi,
            static_cast<Complex>(coeff_2) * one_over_four_pi,
            static_cast<Complex>(coeff_3) * one_over_four_pi
        };

        Re_mass_matrix  = (real(z[0]) != Scalar::Zero<Real>);
        Im_mass_matrix  = (imag(z[0]) != Scalar::Zero<Real>);
        Re_single_layer = (real(z[1]) != Scalar::Zero<Real>);
        Im_single_layer = (imag(z[1]) != Scalar::Zero<Real>);
        Re_double_layer = (real(z[2]) != Scalar::Zero<Real>);
        Im_double_layer = (imag(z[2]) != Scalar::Zero<Real>);
        Re_adjdbl_layer = (real(z[3]) != Scalar::Zero<Real>);
        Im_adjdbl_layer = (imag(z[3]) != Scalar::Zero<Real>);

        for( Int k = 0; k < wave_chunk_count; ++k )
        {
            copy_buffer<4>( z, c3[3*k+0] );
            copy_buffer<4>( z, c3[3*k+1] );
            copy_buffer<4>( z, c3[3*k+2] );
            copy_buffer<4>( z, c3[3*k+3] );
        }
    }
