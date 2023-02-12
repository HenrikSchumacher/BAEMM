Int BufferSize() const
{
    // Number of instances of Complex that fit into B_buf and C_buf.
    return rows_rounded * ldB;
}

void RequireBuffers( const Int wave_count_  )
{
    wave_count       = wave_count_;
    wave_chunk_count = CeilDivide( wave_chunk_size, wave_count );
    
    const Int new_ld = wave_chunk_count * wave_chunk_size;
    
    if( new_ld > ldB )
    {
        const Int new_size = rows_rounded * new_ld;
        print(ClassName()+"::RequireBuffers: Reallocating buffer to size "+ToString(new_size)+".");
        
        B_loaded = false;
        C_loaded = false;
        
        B_buf = device->newBuffer(new_size * sizeof(Complex), MTL::ResourceStorageModeManaged);
        C_buf = device->newBuffer(new_size * sizeof(Complex), MTL::ResourceStorageModeManaged);
        
        B_ptr = reinterpret_cast<Complex *>(B_buf->contents());
        C_ptr = reinterpret_cast<Complex *>(C_buf->contents());
     
        // Clearing out the right border and the bottom of the buffers is obsolete, because newBuffer already zerofies all bytes.
    }
    
    ldB = ldC        = new_ld;
}

void ModifiedB()
{
    B_buf->didModifyRange({0, B_buf->length()});
}

void ModifiedB( const Int n )
{
    B_buf->didModifyRange({0, static_cast<uint>(n * sizeof(Complex))});
}

void ModifiedC()
{
    C_buf->didModifyRange({0, C_buf->length()});
}

void ModifiedC( const Int n )
{
    C_buf->didModifyRange({0, static_cast<uint>(n * sizeof(Complex))});
}
