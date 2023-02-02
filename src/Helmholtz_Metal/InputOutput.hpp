public:
    
    void RequireBuffers( const Int wave_count_, const Int block_size_, const Int wave_chunk_size_ )
    {
        const Int new_ld          = RoundUpTo( wave_count_, wave_chunk_size_ );
        const Int new_block_count = DivideRoundUp(simplex_count, block_size_ );
        const Int new_n_rounded   = new_block_count * block_size_;
        const Int new_size        = new_n_rounded * new_ld;
        
        if( new_size > ldB * n_rounded )
        {
            print("Reallocating size "+ToString(new_size) );
            B_buf = device->newBuffer(new_size * sizeof(Complex), MTL::ResourceStorageModeManaged);
            C_buf = device->newBuffer(new_size * sizeof(Complex), MTL::ResourceStorageModeManaged);
            
            B_ptr = reinterpret_cast<Complex *>(B_buf->contents());
            C_ptr = reinterpret_cast<Complex *>(C_buf->contents());
        }
        
        wave_count      = wave_count_;
        wave_chunk_size = wave_chunk_size_;
        ldB = ldC       = new_ld;
        block_size      = block_size_;
        block_count     = new_block_count;
        n_rounded       = new_n_rounded;
    }
    
    void RequireBuffers( const Int wave_count_ )
    {
        RequireBuffers( wave_count_, block_size, wave_chunk_size );
    }

//        Complex * B_ptr()
//        {
//            return reinterpret_cast<Complex *>(B_buf->contents());
//        }
//
//        const Complex * B_ptr() const
//        {
//            return reinterpret_cast<Complex *>(B_buf->contents());
//        }
    
    Complex & B( const Int i, const Int k )
    {
        return B_ptr[ldB * i + k];
    }
    
    const Complex & B( const Int i, const Int k ) const
    {
        return B_ptr[ldB * i + k];
    }
    
    void ModifiedB()
    {
        B_buf->didModifyRange({0, B_buf->allocatedSize()});
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
    
    void ModifiedC()
    {
        C_buf->didModifyRange({0, B_buf->allocatedSize()});
    }
    
    void ReadB( ptr<Complex> input, const Int ld_input, const Int wave_count_ )
    {
//            tic("ReadB");
        
        RequireBuffers( wave_count_ );

        #pragma omp parallel for num_threads( OMP_thread_count )
        for( Int i = 0; i < simplex_count; ++i )
        {
            copy_buffer( &input[ld_input * i], &B_ptr[ldB * i], wave_count );
        }
        
        ModifiedB();
        
//            toc("ReadB");
    }
    
    void ReadB( ptr<Complex> input, const Int wave_count_ )
    {
        ReadB( input, wave_count_, wave_count_ );
    }
    
    void WriteB( mut<Complex> output, const Int ld_output )
    {
        Complex * B_ = reinterpret_cast<Complex *>(B_buf->contents());
        
        #pragma omp parallel for num_threads( OMP_thread_count )
        for( Int i = 0; i < simplex_count; ++i )
        {
            copy_buffer( &B_[ldB * i], &output[ld_output * i], wave_count );
        }
    }
    
    void ReadC( ptr<Complex> input, const Int ld_input, const Int wave_count_ )
    {
//            tic("ReadC");
        RequireBuffers( wave_count_ );
        
        #pragma omp parallel for num_threads( OMP_thread_count )
        for( Int i = 0; i < simplex_count; ++i )
        {
            copy_buffer( &input[ld_input * i], &C_ptr[ldC * i], wave_count );
        }
        
        ModifiedC();
        
//            toc("ReadC");
    }
    
    void ReadC( ptr<Complex> input, const Int wave_count_ )
    {
        ReadB( input, wave_count_, wave_count_ );
    }
    
    void WriteC( mut<Complex> output, const Int ld_output )
    {
//            tic("WriteC");
        
        #pragma omp parallel for num_threads( OMP_thread_count )
        for( Int i = 0; i < simplex_count; ++i )
        {
            copy_buffer( &C_ptr[ldC * i], &output[ld_output * i], wave_count );
        }
        
//            toc("WriteC");
    }
