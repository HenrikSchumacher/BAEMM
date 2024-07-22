private:

    // The idea of this host code is to keep the compiled program and uploaded buffers as long as possible to accelerate
    // the use in a linear solver where a flexibility in sizes and coefficients is not needed during the iterations.
    int BoundaryOperatorKernel_C()
    {
        std::string tag = ClassName()+"::BoundaryOperatorKernel_C";
        
        ptic(tag);
        
        // write to buffers
        clEnqueueWriteBuffer(command_queue, B_buf, CL_FALSE, 0, rows_rounded * wave_count * sizeof(Complex), B_ptr, 0, NULL, NULL);
        clFinish(command_queue);
        
        // Execute the OpenCL kernel on the list
        std::size_t local_item_size  = block_size;
        std::size_t global_item_size = rows_rounded;
        ret = clEnqueueNDRangeKernel(command_queue, bdr_kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

        // Read the memory buffer C on the device to the local variable C
        ret = clEnqueueReadBuffer(command_queue, C_buf, CL_TRUE, 0, wave_count * simplex_count * sizeof(Complex), C_ptr, 0, NULL, NULL);

        clFinish(command_queue);
        
        ptoc(tag);
        
        return 0;
    }

    void LoadBoundaryOperatorKernel_PL()
    {
        std::string tag = ClassName()+"::LoadBoundaryOperatorKernel_PL";
        ptic(tag);
        
        // Loads kappa and c to the device.
        // Compiles the kernel.
        
        RequireBuffers( wave_count );
        
        int n = static_cast<int>(simplex_count);

        std::size_t max_work_group_size; //check for maximal size of work group
        ret = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(std::size_t), &max_work_group_size, NULL);

        if (block_size > max_work_group_size)
        {
            SetBlockSize(static_cast<Int>(max_work_group_size));
        }

        ReleaseParameters();
        
        // Create the rest of the memory buffers on the device for each vector
        d_kappa      = clCreateBuffer(context, CL_MEM_READ_ONLY,     wave_chunk_count * sizeof(Real),    NULL, &ret);
        d_coeff      = clCreateBuffer(context, CL_MEM_READ_ONLY, 4 * wave_chunk_count * sizeof(Complex), NULL, &ret);
        d_n          = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Int),                            NULL, &ret);
        d_wave_count = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Int),                            NULL, &ret);
    
        clEnqueueWriteBuffer(command_queue, d_kappa,      CL_FALSE, 0,     wave_chunk_count * sizeof(Real),    kappa.data(), 0, NULL, NULL);
        clEnqueueWriteBuffer(command_queue, d_coeff,      CL_FALSE, 0, 4 * wave_chunk_count * sizeof(Complex), c.data(),     0, NULL, NULL);
        clEnqueueWriteBuffer(command_queue, d_n,          CL_FALSE, 0, sizeof(Int),                            &n,           0, NULL, NULL);
        clEnqueueWriteBuffer(command_queue, d_wave_count, CL_FALSE, 0, sizeof(Int),                            &wave_count,  0, NULL, NULL);

        // Create kernel source
        std::string source = CreateSourceString(
#include "BoundaryOperatorKernel_C.cl"
            ,block_size,wave_chunk_size
        );
            
        const char * source_str = source.c_str();
        std::size_t source_size = source.size();
        
        // Create a program from the kernel source
        cl_program program = clCreateProgramWithSource(context, 1, &source_str, (const std::size_t *)&source_size, &ret);
        
        if (ret != 0)
        {
            eprint( tag + ": Call to clCreateProgramWithSource failed.");
        }
        
        const char * opts = "-cl-fast-relaxed-math";
//        const char * opts = NULL;
        
        // Build the program
        ret = clBuildProgram(program, 1, &device_id, opts, NULL, NULL);
        
        
        if (ret != 0)
        {
            eprint( tag + ": Call to clBuildProgram failed.");
            
            char result[16384];
            std::size_t size;
            ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(result), &result, &size);
        
            
            print(result);
        }
                
        // Create the OpenCL kernel
        bdr_kernel = clCreateKernel(program, "BoundaryOperatorKernel_C", &ret);
        
        if (ret != 0)
        {
            eprint( tag + ": Call to BoundaryOperatorKernel_C failed.");
        }
        
        // Set the arguments of the kernel
        ret = clSetKernelArg(bdr_kernel, 0, sizeof(cl_mem), (void *)&mid_points);
        ret = clSetKernelArg(bdr_kernel, 1, sizeof(cl_mem), (void *)&normals);
        ret = clSetKernelArg(bdr_kernel, 2, sizeof(cl_mem), (void *)&B_buf);
        ret = clSetKernelArg(bdr_kernel, 3, sizeof(cl_mem), (void *)&C_buf);
        ret = clSetKernelArg(bdr_kernel, 4, sizeof(cl_mem), (void *)&d_kappa);
        ret = clSetKernelArg(bdr_kernel, 5, sizeof(cl_mem), (void *)&d_coeff);
        ret = clSetKernelArg(bdr_kernel, 6, sizeof(cl_mem), (void *)&d_n);
        ret = clSetKernelArg(bdr_kernel, 7, sizeof(cl_mem), (void *)&d_wave_count);
        
        clFinish(command_queue);
        
        // clean up
        ret = clReleaseProgram(program);
        
        if (ret != 0)
        {
            eprint( tag + ": Call to clReleaseProgram failed.");
        }

        ptoc(tag);
        
    }

    void UnloadBoundaryOperatorKernel_PL()
    {
        std::string tag = ClassName()+"::UnloadBoundaryOperatorKernel_PL";
        
        ptic(tag);
        
        // Clean up
        ret = clFinish(command_queue);
        ret = clFlush(command_queue);
        ret = clReleaseKernel(bdr_kernel);
        
        ReleaseParameters();
        
        ptoc(tag);
    }


    void ReleaseParameters()
    {
        if( d_kappa != NULL )
        {
            ret = clReleaseMemObject(d_kappa);
            
            d_kappa = NULL;
        }
        
        if( d_coeff != NULL )
        {
            ret = clReleaseMemObject(d_coeff);
            
            d_coeff = NULL;
        }
        
        if( d_n != NULL )
        {
            ret = clReleaseMemObject(d_n);
            
            d_n = NULL;
        }
        
        if( d_wave_count != NULL )
        {
            ret = clReleaseMemObject(d_wave_count);
            
            d_wave_count = NULL;
        }
    }
