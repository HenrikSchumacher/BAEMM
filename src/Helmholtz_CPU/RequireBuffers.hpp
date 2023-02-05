void RequireBuffers( const Int wave_count_, const Int wave_chunk_size_ )
{
    const Int new_ld = RoundUpTo( wave_count_, wave_chunk_size_ );
    
    if( wave_count_ > wave_count )
    {
        B_loaded = false;
        C_loaded = false;
        
        B_buf = Tensor2<Complex,Int> (simplex_count, wave_count_);
        C_buf = Tensor2<Complex,Int> (simplex_count, wave_count_);
        
        B_ptr = B_buf.data();
        C_ptr = C_buf.data();
    }
    
    wave_count      = wave_count_;
    wave_chunk_size = wave_chunk_size_;
    ldB = ldC       = new_ld;
}

void RequireBuffers( const Int wave_count_ )
{
    RequireBuffers( wave_count_, wave_chunk_size );
}

void ModifiedB() {}

void ModifiedC() {}
