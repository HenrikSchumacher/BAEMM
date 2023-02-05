void RequireBuffers( const Int wave_count_, const Int block_size_, const Int wave_chunk_size_ )
{
    const Int new_ld          = RoundUpTo( wave_count_, wave_chunk_size_ );
    const Int new_block_count = DivideRoundUp(simplex_count, block_size_ );
    const Int new_n_rounded   = new_block_count * block_size;
    const Int new_size        = new_n_rounded * new_ld;
    
    if( new_size > ldB * n_rounded )
    {
        B_loaded = false;
        C_loaded = false;
        
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

void RequireBuffers( const Int wave_count_, const Int wave_chunk_size_ )
{
    RequireBuffers( wave_count_, block_size, wave_chunk_size_ );
}

void RequireBuffers( const Int wave_count_ )
{
    RequireBuffers( wave_count_, wave_chunk_size );
}

void ModifiedB()
{
    B_buf->didModifyRange({0, B_buf->length()});
}

void ModifiedC()
{
    C_buf->didModifyRange({0, C_buf->length()});
}
