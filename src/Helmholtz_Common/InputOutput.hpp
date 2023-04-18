public:
    
    Complex & B( const Int i, const Int k )
    {
        return B_ptr[ldB * i + k];
    }
    
    const Complex & B( const Int i, const Int k ) const
    {
        return B_ptr[ldB * i + k];
    }
    
    
//        Complex * C_ptr()
//        {
//            return reinterpret_cast<Complex *>(C_buf->contents());
//        }
//
//        const Complex * C_ptr() const
//        {
//            return reinterpret_cast<Complex *>(C_buf->contents());
//        }
    
    Complex & C( const Int i, const Int k )
    {
        return C_ptr[ldC * i + k];
    }
    
    const Complex & C( const Int i, const Int k ) const
    {
        return C_ptr[ldC * i + k];
    }
    
    void ReadB( ptr<Complex> input, const Int ld_input, const Int wave_count_ )
    {
        RequireBuffers( wave_count_ );
        #pragma omp parallel for num_threads( OMP_thread_count )
        for( Int i = 0; i < simplex_count; ++i )
        {
            copy_buffer( &input[ld_input * i], &B_ptr[ldB * i], wave_count );
        }

        ModifiedB();
    }
    
    void ReadB( ptr<Complex> input, const Int wave_count_ )
    {
        ReadB( input, wave_count_, wave_count_ );
    }
    
    void WriteB( mut<Complex> output, const Int ld_output )
    {
        #pragma omp parallel for num_threads( OMP_thread_count )
        for( Int i = 0; i < simplex_count; ++i )
        {
            copy_buffer( &B_ptr[ldB * i], &output[ld_output * i], wave_count );
        }
    }
    
    void ReadC( ptr<Complex> input, const Int ld_input, const Int wave_count_ )
    {
        RequireBuffers( wave_count_ );
        
        #pragma omp parallel for num_threads( OMP_thread_count )
        for( Int i = 0; i < simplex_count; ++i )
        {
            copy_buffer( &input[ld_input * i], &C_ptr[ldC * i], wave_count );
        }
        
        ModifiedC();
    }
    
    void ReadC( ptr<Complex> input, const Int wave_count_ )
    {
        ReadC( input, wave_count_, wave_count_ );
    }
    
    void WriteC( mut<Complex> output, const Int ld_output )
    {
        #pragma omp parallel for num_threads( OMP_thread_count )
        for( Int i = 0; i < simplex_count; ++i )
        {
            copy_buffer( &C_ptr[ldC * i], &output[ld_output * i], wave_count );
        }
    }
