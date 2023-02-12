void RequireBuffers( const Int wave_count_ )
{
    wave_count       = wave_count_;
    wave_chunk_count = GetWaveChunkCount( wave_count );
    
    const Int new_ld = wave_chunk_count * wave_chunk_size;
    
    if( wave_count_ > wave_count )
    {
        B_loaded = false;
        C_loaded = false;
        
        B_buf = Tensor2<Complex,Int> (simplex_count, wave_count);
        C_buf = Tensor2<Complex,Int> (simplex_count, wave_count);
        
        B_ptr = B_buf.data();
        C_ptr = C_buf.data();
    }
    
    ldB = ldC        = new_ld;
}

void ModifiedB() {}

void ModifiedB( const Int n ) {}

void ModifiedC() {}

void ModifiedC( const Int n ) {}
