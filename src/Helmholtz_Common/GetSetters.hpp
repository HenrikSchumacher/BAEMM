public:
    
    Int VertexCount() const
    {
        return vertex_count;
    }
    
    Int SimplexCount() const
    {
        return simplex_count;
    }
    
    const Sparse_T & MassMatrix() const
    {
        return Mass;
    }

    ptr<Real> Areas() const
    {
        return areas_ptr;
    }

public:

    Int GetWaveCount() const
    {
        return wave_count;
    }

    void SetWaveCount( const Int wave_count_ )
    {
        wave_count = wave_count_;
        B_loaded = false;
        C_loaded = false;
    }

    Int GetWaveChunkSize() const
    {
        return wave_chunk_size;
    }

    void SetWaveChunkSize( const Int wave_chunk_size_ )
    {
        wave_chunk_size = wave_chunk_size_;
        B_loaded = false;
        C_loaded = false;
    }

    Int GetWaveChunkCount( const Int wave_count_ ) const
    {
        return CeilDivide(wave_count_, wave_chunk_size);
    }

    Int GetBlockSize() const
    {
        return block_size;
    }

    void SetBlockSize( const Int block_size_ )
    {
        block_size    = block_size_;
        block_count   = CeilDivide( simplex_count, block_size);
        rows_rounded  = block_count * block_size;
        
        B_loaded = false;
        C_loaded = false;
    }

public:


    template<typename R_ext, typename C_ext, typename I_ext>
    void LoadCoefficients(
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
            
            z = c[k][1] = static_cast<Complex>(coeff_list[4*k+1]) * one_over_four_pi;
            Re_single_layer = Re_single_layer || (real(z) != Scalar::Zero<Real>);
            Im_single_layer = Im_single_layer || (imag(z) != Scalar::Zero<Real>);
            
            z = c[k][2] = static_cast<Complex>(coeff_list[4*k+2]) * one_over_four_pi;
            Re_double_layer = Re_double_layer || (real(z) != Scalar::Zero<Real>);
            Im_double_layer = Im_double_layer || (imag(z) != Scalar::Zero<Real>);
            
            z = c[k][3] = static_cast<Complex>(coeff_list[4*k+3]) * one_over_four_pi;
            Re_adjdbl_layer = Re_adjdbl_layer || (real(z) != Scalar::Zero<Real>);
            Im_adjdbl_layer = Im_adjdbl_layer || (imag(z) != Scalar::Zero<Real>);
        }
    }

//    template<typename R_ext, typename C_ext, typename I_ext>
//    void LoadCoefficients(
//        const Tensor1<R_ext,I_ext> & kappa_list,
//        const Tensor2<C_ext,I_ext> & coeff_list,
//        const Int wave_count_,
//        const Int wave_chunk_size_
//    )
//    {
//        ASSERT_REAL(R_ext);
//        ASSERT_COMPLEX(C_ext);
//
//        SetWaveCount    ( int_cast<Int>(wave_count_)      );
//        SetWaveChunkSize( int_cast<Int>(wave_chunk_size_) );
//
//        wave_chunk_count = GetWaveChunkCount(wave_count);
//
//        if( kappa_list.Dimension(0) != GetWaveChunkCount(wave_count_) )
//        {
//            eprint(ClassName()+"::LoadCoefficients: kappa_list.Dimension(0) != GetWaveChunkCount(wave_count_). Aborting.");
//            return;
//        }
//
//        if( coeff_list.Dimension(1) != 4 )
//        {
//            eprint(ClassName()+"::LoadCoefficients: coeff_list.Dimension(1) != 4. Aborting.");
//            return;
//        }
//
//        if( coeff_list.Dimension(0) != GetWaveChunkCount(wave_count_) )
//        {
//            eprint(ClassName()+"::LoadCoefficients: coeff_list.Dimension(0) != GetWaveChunkCount(wave_count_). Aborting.");
//            return;
//        }
//
//        LoadCoefficients(kappa_list.data(), coeff_list.data(), wave_count_, wave_chunk_size_ );
//    }



    template<typename R_ext, typename C_ext, typename I_ext>
    void LoadCoefficients(
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
        
        if( c.Dimension(0) != wave_chunk_count )
        {
            c = CoefficientContainer_T( wave_chunk_count, 4 );
        }
        
        if( kappa.Dimension(0) != wave_chunk_count )
        {
            kappa = WaveNumberContainer_T( wave_chunk_count, static_cast<Real>(kappa_) );
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
            c[k][0] = z[0];
            c[k][1] = z[1];
            c[k][2] = z[2];
            c[k][3] = z[3];
        }
    }


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
