LInt BufferSize() const
{
    // Number of instances of Complex that fit into B_buf and C_buf.
    return std::min( B_buf.Size(), B_buf.Size() );
}

void RequireBuffers( const Int wave_count_ )
{
    wave_count       = wave_count_;
    wave_chunk_count = GetWaveChunkCount(wave_count);
    ldB = ldC        = wave_chunk_count * wave_chunk_size;
    
    if( simplex_count * wave_count > B_buf.Size() )
    {
//        print(ClassName()+"::RequireBuffers: Reallocating buffer to size "+ToString(simplex_count)+" * "+ToString(wave_count)+" = "+ToString(simplex_count * wave_count)+".");
        
        B_loaded = false;
        C_loaded = false;
        
        B_buf = Tensor2<Complex,Int> (simplex_count, wave_count);
        C_buf = Tensor2<Complex,Int> (simplex_count, wave_count);
        
        B_ptr = B_buf.data();
        C_ptr = C_buf.data();
    }
}

void ModifiedB() {}

void ModifiedB( const LInt n ) {}

void ModifiedC() {}

void ModifiedC( const LInt n ) {}
