public:

    
    
    Complex & B( const Int i, const Int k )
    {
        return B_ptr[ldB * i + k];
    }
    
    const Complex & B( const Int i, const Int k ) const
    {
        return B_ptr[ldB * i + k];
    }
    
    Complex & C( const Int i, const Int k )
    {
        return C_ptr[ldC * i + k];
    }
    
    const Complex & C( const Int i, const Int k ) const
    {
        return C_ptr[ldC * i + k];
    }
    
    void ReadB( cptr<Complex> input, const Int ld_input, const Int wave_count_ )
    {
        RequireBuffers( wave_count_ );
        
        //CheckThis
        ParallelDo(
            [=,this]( const Int i )
            {
                copy_buffer( &input[ld_input * i], &B_ptr[ldB * i], wave_count );
            },
            simplex_count, CPU_thread_count
        );

        ModifiedB();
    }
    
    void ReadB( cptr<Complex> input, const Int wave_count_ )
    {
        ReadB( input, wave_count_, wave_count_ );
    }
    
    void WriteB( mptr<Complex> output, const Int ld_output ) const
    {
        //CheckThis
        ParallelDo(
            [=,this]( const Int i )
            {
                copy_buffer( &B_ptr[ldB * i], &output[ld_output * i], wave_count );
            },
            simplex_count, CPU_thread_count
        );
    }
    
    void ReadC( cptr<Complex> input, const Int ld_input, const Int wave_count_ )
    {
        RequireBuffers( wave_count_ );
        
        //CheckThis
        ParallelDo(
            [=,this]( const Int i )
            {
                copy_buffer( &input[ld_input * i], &C_ptr[ldC * i], wave_count );
            },
            simplex_count, CPU_thread_count
        );
        
        ModifiedC();
    }
    
    void ReadC( cptr<Complex> input, const Int wave_count_ )
    {
        ReadC( input, wave_count_, wave_count_ );
    }
    
    void WriteC( mptr<Complex> output, const Int ld_output ) const
    {
        //CheckThis
        ParallelDo(
            [=,this]( const Int i )
            {
                copy_buffer( &C_ptr[ldC * i], &output[ld_output * i], wave_count );
            },
            simplex_count, CPU_thread_count
        );
    }
