public:
    template<typename R_ext, typename C_ext, typename I_ext>
    void LoadParameters(
        cptr<R_ext> kappa_list,
        cptr<C_ext> coeff_list,
        const I_ext wave_count_,
        const I_ext wave_chunk_size_
    )
    {
        LoadParameters(
            kappa_list,
            coeff_list,
            wave_count_,
            wave_chunk_size_,
            static_cast<R_ext>(1)
        );
    }


    template<typename R_ext, typename C_ext, typename I_ext>
    void LoadParameters(
        cptr<R_ext> kappa_list,
        cptr<C_ext> coeff_list,
        const I_ext wave_count_,
        const I_ext wave_chunk_size_,
        const R_ext factor
    )
    {
        ASSERT_INT(I_ext);
        ASSERT_REAL(R_ext);
        ASSERT_COMPLEX(C_ext);

        SetWaveCount    ( int_cast<Int>(wave_count_)      );
        SetWaveChunkSize( int_cast<Int>(wave_chunk_size_) );
        
        wave_chunk_count = GetWaveChunkCount(wave_count);
        
        if( c.Dimension(0) != wave_chunk_count )
        {
            c = CoefficientContainer_T( wave_chunk_count, 4 );
        }
        
        if( kappa.Dimension(0) != wave_chunk_count )
        {
            kappa = WaveNumberContainer_T( wave_chunk_count );
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
            kappa[k] = kappa_list[k];
            
            Complex z;
            
            // We have to process the coefficients anyways.
            // Hence we can already multiply by one_over_four_pi so that the kernels don't have to do that each time. (The performance gain is not measureable, though.)
            
            z = c[k][0] = static_cast<Complex>(coeff_list[4*k+0]);
            Re_mass_matrix  = Re_mass_matrix  || (real(z) != Scalar::Zero<Real>);
            Im_mass_matrix  = Im_mass_matrix  || (imag(z) != Scalar::Zero<Real>);
            
            z = c[k][1] = static_cast<Complex>(coeff_list[4*k+1]) * one_over_four_pi * static_cast<Real>(factor);
            Re_single_layer = Re_single_layer || (real(z) != Scalar::Zero<Real>);
            Im_single_layer = Im_single_layer || (imag(z) != Scalar::Zero<Real>);
            
            z = c[k][2] = static_cast<Complex>(coeff_list[4*k+2]) * one_over_four_pi * static_cast<Real>(factor);
            Re_double_layer = Re_double_layer || (real(z) != Scalar::Zero<Real>);
            Im_double_layer = Im_double_layer || (imag(z) != Scalar::Zero<Real>);
            
            z = c[k][3] = static_cast<Complex>(coeff_list[4*k+3]) * one_over_four_pi * static_cast<Real>(factor);
            Re_adjdbl_layer = Re_adjdbl_layer || (real(z) != Scalar::Zero<Real>);
            Im_adjdbl_layer = Im_adjdbl_layer || (imag(z) != Scalar::Zero<Real>);
        }
    }

//    template<typename R_ext, typename C_ext, typename I_ext>
//    void LoadCoefficients(
//        const R_ext kappa_,
//        const C_ext coeff_0,
//        const C_ext coeff_1,
//        const C_ext coeff_2,
//        const C_ext coeff_3,
//        const I_ext wave_count_,
//        const I_ext wave_chunk_size_
//    )
//    {
//        LoadCoefficients(
//            kappa_,
//            coeff_0,
//            coeff_1,
//            coeff_2,
//            coeff_3,
//            wave_count_,
//            wave_chunk_size_,
//            static_cast<R_ext>(1)
//        );
//    }
//
//
//    template<typename R_ext, typename C_ext, typename I_ext>
//    void LoadCoefficients(
//        const R_ext kappa_,
//        const C_ext coeff_0,
//        const C_ext coeff_1,
//        const C_ext coeff_2,
//        const C_ext coeff_3,
//        const I_ext wave_count_,
//        const I_ext wave_chunk_size_,
//        const R_ext factor
//    )
//    {
//        ASSERT_INT(I_ext);
//        ASSERT_REAL(R_ext);
//        ASSERT_COMPLEX(C_ext);
//        
//        SetWaveCount    ( int_cast<Int>(wave_count_)      );
//        SetWaveChunkSize( int_cast<Int>(wave_chunk_size_) );
//        
//        wave_chunk_count = GetWaveChunkCount(wave_count);
//
//        if( c.Dimension(0) != wave_chunk_count )
//        {
//            c = CoefficientContainer_T( wave_chunk_count, 4 );
//        }
//        
//        if( kappa.Dimension(0) != wave_chunk_count )
//        {
//            kappa = WaveNumberContainer_T( wave_chunk_count, static_cast<Real>(kappa_) );
//        }
//        
//        // We have to process the coefficients anyways.
//        // Hence we can already multiply by one_over_four_pi so that the kernels don't have to do that each time. (The performance gain is not measureable, though.)
//        Complex z [4] {
//            static_cast<Complex>(coeff_0),
//            static_cast<Complex>(coeff_1) * one_over_four_pi * static_cast<Real>(factor),
//            static_cast<Complex>(coeff_2) * one_over_four_pi * static_cast<Real>(factor),
//            static_cast<Complex>(coeff_3) * one_over_four_pi * static_cast<Real>(factor)
//        };
//        
//        Re_mass_matrix  = (real(z[0]) != Scalar::Zero<Real>);
//        Im_mass_matrix  = (imag(z[0]) != Scalar::Zero<Real>);
//        Re_single_layer = (real(z[1]) != Scalar::Zero<Real>);
//        Im_single_layer = (imag(z[1]) != Scalar::Zero<Real>);
//        Re_double_layer = (real(z[2]) != Scalar::Zero<Real>);
//        Im_double_layer = (imag(z[2]) != Scalar::Zero<Real>);
//        Re_adjdbl_layer = (real(z[3]) != Scalar::Zero<Real>);
//        Im_adjdbl_layer = (imag(z[3]) != Scalar::Zero<Real>);
//        
//        for( Int k = 0; k < wave_chunk_count; ++k )
//        {
//            c[k][0] = z[0];
//            c[k][1] = z[1];
//            c[k][2] = z[2];
//            c[k][3] = z[3];
//        }
//    }


public:

    void PrintBooleans() const
    {
        dump(Re_mass_matrix );
        dump(Im_mass_matrix );
        dump(Re_single_layer);
        dump(Im_single_layer);
        dump(Re_double_layer);
        dump(Im_double_layer);
        dump(Re_adjdbl_layer);
        dump(Im_adjdbl_layer);
    }
