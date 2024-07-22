// After a increasement of the size of either the input or output we nee to reallocate the space  on both, the host and device
// The size of the allocated space depends on the requested kernel
// For details on the allocation process refer to "InitializeBuffers.hpp"

void RequireBuffers( const Int wave_count_  )
{
    wave_count       = wave_count_;
    wave_chunk_count = GetWaveChunkCount(wave_count);
    ldB = ldC        = wave_chunk_count * wave_chunk_size;
    
    const LInt new_size = int_cast<LInt>(rows_rounded) * int_cast<LInt>(ldB) * sizeof(Complex);

    if(
       (B_buf == nullptr) || (C_buf == nullptr)
       ||
       (new_size > std::min(B_size, C_size ) )
    )
    {        
//        print(ClassName()+"::RequireBuffers: Reallocating buffer to size "+ToString(rows_rounded)+" * "+ToString(ldB)+" = "+ToString(rows_rounded * ldB)+".");
        B_size = C_size = new_size;

        // remove the connection between host- and device buffer if existing
        if( B_ptr != nullptr )
        {
            ret = clEnqueueUnmapMemObject(command_queue,B_buf_pin,(void*)B_ptr,0,nullptr,nullptr);
        }
        if( C_ptr != nullptr )
        {
            ret = clEnqueueUnmapMemObject(command_queue,C_buf_pin,(void*)C_ptr,0,nullptr,nullptr);
        }
        clFinish(command_queue); // ensure the queue to be done before reallocating

        B_loaded = false;
        C_loaded = false;

        B_buf_pin = clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_ALLOC_HOST_PTR, new_size, nullptr, &ret);
        C_buf_pin = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, new_size, nullptr, &ret);

        B_ptr = (Complex*)clEnqueueMapBuffer(command_queue, B_buf_pin, CL_TRUE, CL_MAP_WRITE, 0, new_size, 0, nullptr, nullptr, nullptr);
        C_ptr = (Complex*)clEnqueueMapBuffer(command_queue, C_buf_pin, CL_TRUE, CL_MAP_READ,  0, new_size, 0, nullptr, nullptr, nullptr);
        
        B_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,  new_size, nullptr, &ret);
        C_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, new_size, nullptr, &ret);
    }
}

void RequireBuffersFarField( const Int wave_count_  )
{
    wave_count       = wave_count_;
    wave_chunk_count = GetWaveChunkCount(wave_count);
    ldB = ldC        = wave_chunk_count * wave_chunk_size;
    Int rows = block_size * ((meas_count - 1)/block_size + 1);

    const LInt new_size_B = int_cast<LInt>(rows_rounded) * int_cast<LInt>(ldB) * sizeof(Complex);
    const LInt new_size_C = int_cast<LInt>(rows) * int_cast<LInt>(ldC) * sizeof(Complex);
    
    if(
       (B_buf == nullptr) || (C_buf == nullptr)
       ||
       (new_size_C > C_size) || (new_size_B > B_size)
    )
    {
        B_size = new_size_B;
        C_size = new_size_C;

        if( B_ptr != nullptr )
        {
            ret = clEnqueueUnmapMemObject(command_queue,B_buf_pin,(void*)B_ptr,0,nullptr,nullptr);
        }
        if( C_ptr != nullptr )
        {
            ret = clEnqueueUnmapMemObject(command_queue,C_buf_pin,(void*)C_ptr,0,nullptr,nullptr);
        }
        clFinish(command_queue);

        B_loaded = false;
        C_loaded = false;

        B_buf_pin = clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_ALLOC_HOST_PTR, new_size_B, nullptr, &ret);
        C_buf_pin = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, new_size_C, nullptr, &ret);

        B_ptr = (Complex*)clEnqueueMapBuffer(command_queue, B_buf_pin, CL_TRUE, CL_MAP_WRITE, 0, new_size_B, 0, nullptr, nullptr, nullptr);
        C_ptr = (Complex*)clEnqueueMapBuffer(command_queue, C_buf_pin, CL_TRUE, CL_MAP_READ,  0, new_size_C, 0, nullptr, nullptr, nullptr);
     
        B_buf = clCreateBuffer(context, CL_MEM_READ_ONLY,  new_size_B, nullptr, &ret);
        C_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, new_size_C, nullptr, &ret);
    }
}

void RequireBuffersHerglotzWave( const Int wave_count_  )
{
    wave_count       = wave_count_;
    wave_chunk_count = GetWaveChunkCount(wave_count);
    ldB = ldC        = wave_chunk_count * wave_chunk_size;
    Int rows = block_size * ((meas_count - 1)/block_size + 1);

    const LInt new_size_C = int_cast<LInt>(rows_rounded) * int_cast<LInt>(ldC) * sizeof(Complex);
    const LInt new_size_B = int_cast<LInt>(rows) * int_cast<LInt>(ldB) * sizeof(Complex);
    
    if(
       (B_buf == nullptr) || (C_buf == nullptr)
       ||
       (new_size_C > C_size) || (new_size_B > B_size)
    )
    {        
//        print(ClassName()+"::RequireBuffers: Reallocating buffer to size "+ToString(rows_rounded)+" * "+ToString(ldB)+" = "+ToString(rows_rounded * ldB)+".");
        B_size = new_size_B;
        C_size = new_size_C;

        if( B_ptr != nullptr )
        {
            ret = clEnqueueUnmapMemObject(command_queue,B_buf_pin,(void*)B_ptr,0,nullptr,nullptr);
        }
        if( C_ptr != nullptr )
        {
            ret = clEnqueueUnmapMemObject(command_queue,C_buf_pin,(void*)C_ptr,0,nullptr,nullptr);
        }
        clFinish(command_queue);

        B_loaded = false;
        C_loaded = false;

        B_buf_pin = clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_ALLOC_HOST_PTR, new_size_B, nullptr, &ret);
        C_buf_pin = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, new_size_C, nullptr, &ret);

        B_ptr = (Complex*)clEnqueueMapBuffer(command_queue, B_buf_pin, CL_TRUE, CL_MAP_WRITE, 0, new_size_B, 0, nullptr, nullptr, nullptr);
        C_ptr = (Complex*)clEnqueueMapBuffer(command_queue, C_buf_pin, CL_TRUE, CL_MAP_READ , 0, new_size_C, 0, nullptr, nullptr, nullptr);
     
        B_buf = clCreateBuffer(context, CL_MEM_READ_ONLY , new_size_B, nullptr, &ret);
        C_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, new_size_C, nullptr, &ret);
    }
}

void RequireBuffersNearField( const Int wave_count_, const Int evaluation_count_  )
{
    wave_count       = wave_count_;
    wave_chunk_count = GetWaveChunkCount(wave_count);
    ldB = ldC        = wave_chunk_count * wave_chunk_size;
    Int rows = block_size * ((evaluation_count_ - 1)/block_size + 1);

    const LInt new_size_B = int_cast<LInt>(rows_rounded) * int_cast<LInt>(ldB) * sizeof(Complex);
    const LInt new_size_C = int_cast<LInt>(rows) * int_cast<LInt>(ldC) * sizeof(Complex);
    
    if(
       (B_buf == nullptr) || (C_buf == nullptr)
       ||
       (new_size_C > C_size) || (new_size_B > B_size)
    )
    {        
//        print(ClassName()+"::RequireBuffers: Reallocating buffer to size "+ToString(rows_rounded)+" * "+ToString(ldB)+" = "+ToString(rows_rounded * ldB)+".");
        B_size = new_size_B;
        C_size = new_size_C;

        if( B_ptr != nullptr )
        {
            ret = clEnqueueUnmapMemObject(command_queue,B_buf_pin,(void*)B_ptr,0,nullptr,nullptr);
        }
        if( C_ptr != nullptr )
        {
            ret = clEnqueueUnmapMemObject(command_queue,C_buf_pin,(void*)C_ptr,0,nullptr,nullptr);
        }
        clFinish(command_queue);

        B_loaded = false;
        C_loaded = false;

        B_buf_pin = clCreateBuffer(context, CL_MEM_READ_ONLY  | CL_MEM_ALLOC_HOST_PTR, new_size_B, nullptr, &ret);
        C_buf_pin = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, new_size_C, nullptr, &ret);

        B_ptr = (Complex*)clEnqueueMapBuffer(command_queue, B_buf_pin, CL_TRUE, CL_MAP_WRITE, 0, new_size_B, 0, nullptr, nullptr, nullptr);
        C_ptr = (Complex*)clEnqueueMapBuffer(command_queue, C_buf_pin, CL_TRUE, CL_MAP_READ , 0, new_size_C, 0, nullptr, nullptr, nullptr);
     
        B_buf = clCreateBuffer(context, CL_MEM_READ_ONLY , new_size_B, nullptr, &ret);
        C_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, new_size_C, nullptr, &ret);
    }
}

void ModifiedB()
{
    B_loaded = true;
}

void ModifiedC()
{
    C_loaded = true;
}
