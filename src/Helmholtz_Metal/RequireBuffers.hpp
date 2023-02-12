LInt BufferSize() const
{
    // Number of instances of Complex that fit into B_buf and C_buf.
    return (B_buf == nullptr) || (C_buf == nullptr)
        ?
        LInt(0)
        :
        std::min( int_cast<Int>(B_buf->length()), int_cast<Int>(C_buf->length())) / int_cast<Int>(sizeof(Complex));
}

void RequireBuffers( const Int wave_count_  )
{
    wave_count       = wave_count_;
    wave_chunk_count = GetWaveChunkCount(wave_count);
    ldB = ldC        = wave_chunk_count * wave_chunk_size;
    
    const LInt new_size = int_cast<LInt>(rows_rounded) * int_cast<LInt>(ldB) * sizeof(Complex);
    
    if(
       (B_buf == nullptr) || (C_buf == nullptr)
       ||
       (new_size > std::min( B_buf->length(), C_buf->length() ) )
    )
    {
        print(ClassName()+"::RequireBuffers: Reallocating buffer to size "+ToString(rows_rounded)+" * "+ToString(ldB)+" = "+ToString(rows_rounded * ldB)+".");
        
        B_loaded = false;
        C_loaded = false;
        
        B_buf = device->newBuffer(new_size, MTL::ResourceStorageModeManaged);
        C_buf = device->newBuffer(new_size, MTL::ResourceStorageModeManaged);
        
        B_ptr = reinterpret_cast<Complex *>(B_buf->contents());
        C_ptr = reinterpret_cast<Complex *>(C_buf->contents());
     
        // Clearing out the right border and the bottom of the buffers is obsolete, because newBuffer already zerofies all bytes.
    }
}

void ModifiedB()
{
    B_buf->didModifyRange({0, B_buf->length()});
}

void ModifiedB( const LInt n )
{
    B_buf->didModifyRange({0, n * sizeof(Complex)});
}

void ModifiedC()
{
    C_buf->didModifyRange({0, C_buf->length()});
}

void ModifiedC( const LInt n )
{
    C_buf->didModifyRange({0, n * sizeof(Complex)});
}
