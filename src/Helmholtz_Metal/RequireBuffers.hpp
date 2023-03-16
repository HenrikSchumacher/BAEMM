LInt BufferSize() const
{
    // Number of instances of Complex that fit into B_buf and C_buf.
    return (B_buf.get() == nullptr) || (C_buf.get() == nullptr)
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
       (B_buf.get() == nullptr) || (C_buf.get() == nullptr)
       ||
       (new_size > std::min( B_buf->length(), C_buf->length() ) )
    )
    {
//        print(ClassName()+"::RequireBuffers: Reallocating buffer to size "+ToString(rows_rounded)+" * "+ToString(ldB)+" = "+ToString(rows_rounded * ldB)+".");
        
        B_loaded = false;
        C_loaded = false;
        
        B_buf = NS::TransferPtr(device->newBuffer(new_size, Managed));
        C_buf = NS::TransferPtr(device->newBuffer(new_size, Managed));
        
        B_ptr = reinterpret_cast<Complex *>(B_buf->contents());
        C_ptr = reinterpret_cast<Complex *>(C_buf->contents());
     
        // Clearing out the right border and the bottom of the buffers is obsolete, because newBuffer already zerofies all bytes.
    }
    else
    {
        zerofy_buffer( B_ptr, int_cast<LInt>(rows_rounded) * int_cast<LInt>(ldB) );
        zerofy_buffer( C_ptr, int_cast<LInt>(rows_rounded) * int_cast<LInt>(ldC) );
    }
}

void ModifiedB()
{
    B_buf->didModifyRange({0, int_cast<LInt>(rows_rounded) * int_cast<LInt>(ldB) });
    B_loaded = true;
}

void ModifiedC()
{
    C_buf->didModifyRange({0, int_cast<LInt>(rows_rounded) * int_cast<LInt>(ldC) });
    C_loaded = true;
}
