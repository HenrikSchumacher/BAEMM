public:

    int FarFieldOperatorKernel()
    {
        std::string tag = ClassName()+"::FarFieldOperatorKernel";
        
        ptic(tag);
        
        if( kappa.Dimension(0) != wave_chunk_count )
        {
            eprint(tag + ": kappa_.Dimension(0) != wave_chunk_count.");
        }
        
        if( c.Dimension(0) != wave_chunk_count )
        {
            eprint(tag + ": c_.Dimension(0) != wave_chunk_count.");
        }

        int m = meas_count;
        int n = simplex_count;

        // Check for maximal size of work group.
        
        std::size_t max_work_group_size;
        ret = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, nullptr);

        if (block_size > static_cast<Int>(max_work_group_size))
        {
            SetBlockSize(static_cast<Int>(max_work_group_size));
        }
        
        std::string source = CreateSourceString(
#include "FarFieldOperatorKernel.cl"
            ,block_size,wave_chunk_size
        );
        
        if constexpr( print_kernel_codeQ )
        {
            logprint("");
            logprint("");
            logprint("FarFieldOperatorKernel.cl");
            logprint("");
            logprint(source);
            logprint("");
            logprint("");
        }
            
        const char * source_str = source.c_str();
        std::size_t source_size = source.size();

        // Create the rest of the memory buffers on the device for each vector
        ptic(tag + ": clCreateBuffer");
        cl_mem d_kappa      = clCreateBuffer(context, CL_MEM_READ_ONLY, wave_chunk_count * sizeof(Real),        nullptr, &ret);
        cl_mem d_coeff      = clCreateBuffer(context, CL_MEM_READ_ONLY, 4 * wave_chunk_count * sizeof(Complex), nullptr, &ret);
        cl_mem d_n          = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Int),                            nullptr, &ret);
        cl_mem d_m          = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Int),                            nullptr, &ret);
        cl_mem d_wave_count = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(Int),                            nullptr, &ret);
        ptoc(tag + ": clCreateBuffer");

        // write to buffers
        ptic(tag + ": clEnqueueWriteBuffer");
        clEnqueueWriteBuffer(command_queue, d_kappa,            CL_FALSE, 0, wave_chunk_count * sizeof(Real),        kappa.data(),          0, nullptr, nullptr);
        clEnqueueWriteBuffer(command_queue, d_coeff,            CL_FALSE, 0, 4 * wave_chunk_count * sizeof(Complex), c.data(),              0, nullptr, nullptr);
        clEnqueueWriteBuffer(command_queue, d_n,                CL_FALSE, 0, sizeof(Int),                            &n,                    0, nullptr, nullptr);
        clEnqueueWriteBuffer(command_queue, d_m,                CL_FALSE, 0, sizeof(Int),                            &m,                    0, nullptr, nullptr);
        clEnqueueWriteBuffer(command_queue, d_wave_count,       CL_FALSE, 0, sizeof(Int),                            &wave_count,           0, nullptr, nullptr);

        clEnqueueWriteBuffer(command_queue, B_buf, CL_FALSE, 0, rows_rounded * wave_count * sizeof(Complex), B_ptr, 0, nullptr, nullptr);
        ptoc(tag + ": clEnqueueWriteBuffer");

        // Create a program from the kernel source
        cl_program program = clCreateProgramWithSource(context, 1,
                (const char **)&source_str, (const size_t *)&source_size, &ret);
        
        // Build the program
        ret = clBuildProgram(program, 1, &device_id, clBuildOpts, nullptr, nullptr);
        if (ret != 0)
        {
            cl_check_ret( tag, "clCreateProgramWithSource" );
            
            std::size_t result_size = 0;
            ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &result_size);
            std::string result (result_size,' ');
            ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, result_size, &result[0], nullptr);
            print(result);
        }
        
        // Create the OpenCL kernel
        cl_kernel kernel = clCreateKernel(program, "FarFieldOperatorKernel", &ret);
        cl_check_ret( tag, "clCreateKernel" );
        
        // Set the arguments of the kernel
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mid_points);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&normals);
        ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&meas_directions);
        ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&B_buf);
        ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&C_buf);
        ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&d_kappa);
        ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&d_coeff);
        ret = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&d_n);
        ret = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&d_m);
        ret = clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *)&d_wave_count);
        
        clFinish(command_queue);
        
        // Execute the OpenCL kernel on the list
        
        std::size_t local_item_size  = block_size;
        std::size_t global_item_size = local_item_size * ((m - 1) / block_size + 1);
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &global_item_size, &local_item_size, 0, nullptr, nullptr);
                
        // Read the memory buffer C on the device to the local variable C
        ret = clEnqueueReadBuffer(command_queue, C_buf, CL_TRUE, 0, wave_count * m * sizeof(Complex), C_ptr, 0, nullptr, nullptr);

        // Clean up
        ret = clFinish(command_queue);
        ret = clFlush(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);
        ret = clReleaseMemObject(d_kappa);
        ret = clReleaseMemObject(d_coeff);
        ret = clReleaseMemObject(d_n);
        ret = clReleaseMemObject(d_m);
        ret = clReleaseMemObject(d_wave_count);
        
        ptoc(tag);
        
        return 0;
    }
