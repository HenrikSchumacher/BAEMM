private:

    // The idea of this host code is to keep the compiled program and uploaded buffers as long as possible to accelerate
    // the use in a linear solver where a flexibility in sizes and coefficients is not needed during the iterations.
    int BoundaryOperatorKernel()
    {
        std::string tag = ClassName()+"::BoundaryOperatorKernel";
        
        ptic(tag);
        
        // Write to buffers.
        clEnqueueWriteBuffer(command_queue, B_buf, CL_FALSE, 0, rows_rounded * wave_count * sizeof(Complex), B_ptr, 0, nullptr, nullptr);
        clFinish(command_queue);
        
        // Execute the OpenCL kernel on the list
        std::size_t local_item_size  = block_size;
        std::size_t global_item_size = rows_rounded;
        ret = clEnqueueNDRangeKernel(command_queue, bdr_kernel, 1, nullptr, &global_item_size, &local_item_size, 0, nullptr, nullptr);

        // Read the memory buffer C on the device to the local variable C
        ret = clEnqueueReadBuffer(command_queue, C_buf, CL_TRUE, 0, wave_count * simplex_count * sizeof(Complex), C_ptr, 0, nullptr, nullptr);

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
        ret = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(std::size_t), &max_work_group_size, nullptr);

        if (block_size > static_cast<Int>(max_work_group_size) )
        {
            SetBlockSize(static_cast<Int>(max_work_group_size));
        }

        ReleaseParameters();
        
        Real* Kappa = (Real*)malloc(wave_chunk_count *  sizeof(Real));
            kappa.Write(Kappa);
        Complex* Coeff = (Complex*)malloc(wave_chunk_count * 4 * sizeof(Complex));
            c.Write(Coeff);

        ptic(tag + ": clCreateBuffer");
        // Create the rest of the memory buffers on the device for each vector
        d_kappa      = clCreateBuffer(context, CL_MEM_READ_ONLY,     wave_chunk_count * sizeof(Real),    nullptr, &ret);
        d_coeff      = clCreateBuffer(context, CL_MEM_READ_ONLY, 4 * wave_chunk_count * sizeof(Complex), nullptr, &ret);
        d_n          = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Int),                            nullptr, &ret);
        d_wave_count = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Int),                            nullptr, &ret);
        ptoc(tag + ": clCreateBuffer");
        
        ptic(tag + ": clEnqueueWriteBuffer");
        clEnqueueWriteBuffer(command_queue, d_kappa,      CL_FALSE, 0,     wave_chunk_count * sizeof(Real),    Kappa, 0, nullptr, nullptr);
        clEnqueueWriteBuffer(command_queue, d_coeff,      CL_FALSE, 0, 4 * wave_chunk_count * sizeof(Complex), Coeff,     0, nullptr, nullptr);
        clEnqueueWriteBuffer(command_queue, d_n,          CL_FALSE, 0, sizeof(Int),                            &n,           0, nullptr, nullptr);
        clEnqueueWriteBuffer(command_queue, d_wave_count, CL_FALSE, 0, sizeof(Int),                            &wave_count,  0, nullptr, nullptr);
        ptoc(tag + ": clEnqueueWriteBuffer");
        
        // Create kernel source
        std::string source = CreateSourceString(
#include "BoundaryOperatorKernel.cl"
            ,block_size,wave_chunk_size
        );
        
        if constexpr( print_kernel_codeQ )
        {
            logprint("");
            logprint("");
            logprint("BoundaryOperatorKernel.cl");
            logprint("");
            logprint(source);
            logprint("");
            logprint("");
        }
            
        const char * source_str = source.c_str();
        std::size_t source_size = source.size();
        
        ptic(tag + ": clCreateProgramWithSource");
        // Create a program from the kernel source
        cl_program program = clCreateProgramWithSource(context, 1, &source_str, (const std::size_t *)&source_size, &ret);
        cl_check_ret( tag, "clCreateProgramWithSource" );
        ptoc(tag + ": clCreateProgramWithSource");
        
        // Build the program
        ptic(tag + ": clBuildProgram");
        ret = clBuildProgram(program, 1, &device_id, clBuildOpts, nullptr, nullptr);
        if (ret != 0)
        {
            cl_check_ret( tag, "clBuildProgram" );
            
            char result[16384];
            std::size_t size;
            ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(result), &result, &size);
            
            print(result);
        }
        ptoc(tag + ": clBuildProgram");
                
        // Create the OpenCL kernel
        ptic(tag + ": clCreateKernel");
        bdr_kernel = clCreateKernel(program, "BoundaryOperatorKernel", &ret);
        cl_check_ret( tag, "clCreateKernel" );
        ptoc(tag + ": clCreateKernel");
        
        // Set the arguments of the kernel
        ptic(tag + ": clSetKernelArg");
        ret = clSetKernelArg(bdr_kernel, 0, sizeof(cl_mem), (void *)&mid_points);
        ret = clSetKernelArg(bdr_kernel, 1, sizeof(cl_mem), (void *)&normals);
        ret = clSetKernelArg(bdr_kernel, 2, sizeof(cl_mem), (void *)&B_buf);
        ret = clSetKernelArg(bdr_kernel, 3, sizeof(cl_mem), (void *)&C_buf);
        ret = clSetKernelArg(bdr_kernel, 4, sizeof(cl_mem), (void *)&d_kappa);
        ret = clSetKernelArg(bdr_kernel, 5, sizeof(cl_mem), (void *)&d_coeff);
        ret = clSetKernelArg(bdr_kernel, 6, sizeof(cl_mem), (void *)&d_n);
        ret = clSetKernelArg(bdr_kernel, 7, sizeof(cl_mem), (void *)&d_wave_count);
        ptoc(tag + ": clSetKernelArg");
        
        ptic(tag + ": clFinish");
        clFinish(command_queue);
        ptoc(tag + ": clFinish");
        
        // clean up
        ptic(tag + ": clReleaseProgram");
        ret = clReleaseProgram(program);
        cl_check_ret( tag, "clReleaseProgram" );
        ptoc(tag + ": clReleaseProgram");

        free(Kappa);
        free(Coeff);

        ptoc(tag);
        
    }

    void UnloadBoundaryOperatorKernel_PL()
    {
        std::string tag = ClassName()+"::UnloadBoundaryOperatorKernel_PL";
        
        ptic(tag);
        
        // Clean up

        ptic(tag + ": clFinish");
        ret = clFinish(command_queue);
        cl_check_ret( tag, "clFinish" );
        ptoc(tag + ": clFinish");

        ptic(tag + ": clFlush");
        ret = clFlush(command_queue);
        cl_check_ret( tag, "clFlush" );
        ptoc(tag + ": clFlush");

        ptic(tag + ": clReleaseKernel");
        ret = clReleaseKernel(bdr_kernel);
        cl_check_ret( tag, "clReleaseKernel" );
        ptoc(tag + ": clReleaseKernel");

        ReleaseParameters();
        
        ptoc(tag);
    }


    void ReleaseParameters()
    {
        ptic( ClassName() + "::ReleaseParameters" );
        
        if( d_kappa != nullptr )
        {
            ret = clReleaseMemObject(d_kappa);
            d_kappa = nullptr;
        }

        if( d_coeff != nullptr )
        {
            ret = clReleaseMemObject(d_coeff);
            d_coeff = nullptr;
        }

        if( d_n != nullptr )
        {
            ret = clReleaseMemObject(d_n);
            d_n = nullptr;
        }

        if( d_wave_count != nullptr )
        {
            ret = clReleaseMemObject(d_wave_count);
            d_wave_count = nullptr;
        }

        ptoc( ClassName() + "::ReleaseParameters" );
    }
