private:

    /**
     * Applies the diagonal of the (singular!) single-layer boundary operator. Refer to Kirkups book for the formula.
     */
    void ApplySingleLayerDiagonal(
        const WaveNumberContainer_T  & kappa_,
        const CoefficientContainer_T & c_
    )
    {
        const Int wave_chunk_count_ = kappa_.Size();
        
        if( use_diagonal
            &&
            (Re_single_layer || Im_single_layer)
            &&
            (wave_chunk_count_ >= 1)
        )
        {
            Tensor1<Complex,Int> I_kappa ( wave_chunk_count_ );
            
            const Int border_size = wave_count - wave_chunk_size * (wave_chunk_count_ - 1 );
            
            for( Int chunk = 0; chunk < wave_chunk_count_; ++chunk )
            {
                I_kappa[chunk] = Complex( Scalar::Zero<Real>, kappa[chunk] );
            }

            ParallelDo(
                [=]( const Int i )
                {
                    for( Int chunk = 0; chunk < wave_chunk_count_ - 1; ++chunk )
                    {
                        const Int pos = ldB * i + wave_chunk_size * chunk;
                        
                        const Complex factor = c_(chunk,1) * (single_diag_ptr[i] + I_kappa[chunk]);
                        
                        combine_buffers<Scalar::Flag::Generic, Scalar::Flag::Plus>(
                            factor,               &B_ptr[pos],
                            Scalar::One<Complex>, &C_ptr[pos],
                            wave_chunk_size
                        );
                    }
                    
                    {
                        const Int chunk = wave_chunk_count_ - 1;
                        
                        const Int pos = ldB * i + wave_chunk_size * chunk;
                        
                        const Complex factor = c(chunk,1) * (single_diag_ptr[i] + I_kappa[chunk]);
                        
                        combine_buffers<Scalar::Flag::Generic,Scalar::Flag::Plus>(
                            factor,               &B_ptr[pos],
                            Scalar::One<Complex>, &C_ptr[pos],
                            border_size
                        );
                    }
                },
                simplex_count,
                CPU_thread_count
            );
        }
    }
