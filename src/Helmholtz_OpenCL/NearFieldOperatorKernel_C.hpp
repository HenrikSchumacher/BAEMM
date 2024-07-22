public:

    int NearFieldOperatorKernel_C(
        cptr<Real> evaluation_points_ptr,
        const Int  evaluation_count_,
        const WaveNumberContainer_T  & kappa_,
        const CoefficientContainer_T & c_
    )
    {
        std::string tag = ClassName() + "::NearFieldOperatorKernel_C";
        
        ptic(tag);
        
        // Allocate local host pointers for the device buffers to use.
        // Henrik: I am not sure whether this is necessary.
        Tensor1<Real,Int> Kappa ( wave_chunk_count );
        kappa_.Write(Kappa.data());
        
        Tensor2<Complex,Int> Coeff ( wave_chunk_count, 4 );
        c_.Write(Coeff.data());

        Int n = simplex_count;
        Int evaluation_count = evaluation_count_;

        //Check for maximal size of work group.
        std::size_t max_work_group_size;
        ret = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);

        if (block_size > max_work_group_size)
        {
            SetBlockSize(static_cast<Int>(max_work_group_size));
        }

        std::string source = CreateSourceString(
    #include "NearFieldOperatorKernel_C.cl"
            ,block_size,wave_chunk_size
        );
            
        const char *      source_str  = source.c_str();
        const std::size_t source_size = source.size();
    
        // Create the rest of the memory buffers on the device for each vector.
        cl_mem d_kappa            = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,     wave_chunk_count * sizeof(Real),    Kappa.data(),      &ret);
        cl_mem d_coeff            = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 4 * wave_chunk_count * sizeof(Complex), Coeff.data(),      &ret);
        cl_mem d_n                = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int),                            &n,                &ret);
        cl_mem d_wave_count       = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int),                            &wave_count,       &ret);
        cl_mem d_evaluation_count = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int),                            &evaluation_count, &ret);
        cl_mem evaluation_points  = clCreateBuffer(context, CL_MEM_READ_ONLY,                       4 * evaluation_count * sizeof(Real),    NULL,              &ret);

        // Write potential to buffers.
        clEnqueueWriteBuffer(command_queue, B_buf, CL_FALSE, 0, rows_rounded * wave_count * sizeof(Complex), B_ptr, 0, NULL, NULL);

        clEnqueueWriteBuffer(command_queue, evaluation_points, CL_FALSE, 0, 4 * evaluation_count * sizeof(Real), evaluation_points_ptr, 0, NULL, NULL);

        // Create a program from the kernel source
        cl_program program = clCreateProgramWithSource(context, 1, &source_str, &source_size, &ret);

        // Build the program
        ret = clBuildProgram(program, 1, &device_id, clBuildOpts, NULL, NULL);
        if (ret != 0)
        {
            cl_check_ret( tag, "clCreateProgramWithSource" );
            
            char result[16384];
            size_t size;
            ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(result), &result, &size);
            
            print(result);
        }
                
        // Create the OpenCL kernel
        cl_kernel kernel = clCreateKernel(program, "NearFieldOperatorKernel_C", &ret);
        cl_check_ret( tag, "clCreateKernel" );
        
        // Set the arguments of the kernel
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&mid_points);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&normals);
        ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&B_buf);
        ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&evaluation_points);
        ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void *)&C_buf);
        ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void *)&d_kappa);
        ret = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&d_coeff);
        ret = clSetKernelArg(kernel, 7, sizeof(cl_mem), (void *)&d_n);
        ret = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&d_wave_count);
        ret = clSetKernelArg(kernel, 9, sizeof(cl_mem), (void *)&d_evaluation_count);

        ret = clFinish(command_queue);
        
        // Execute the OpenCL kernel on the list
        size_t local_item_size = block_size;
        size_t global_item_size = local_item_size * ((evaluation_count_ - 1)/block_size + 1);
        
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

        // Read the memory buffer C on the device to the local variable C
        ret = clEnqueueReadBuffer(command_queue, C_buf, CL_TRUE, 0, wave_count * evaluation_count * sizeof(Complex), C_ptr, 0, NULL, NULL);

        // Clean up
        ret = clFinish(command_queue);
        ret = clFlush(command_queue);
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);
        ret = clReleaseMemObject(evaluation_points);
        ret = clReleaseMemObject(d_kappa);
        ret = clReleaseMemObject(d_coeff);
        ret = clReleaseMemObject(d_n);
        ret = clReleaseMemObject(d_wave_count);
        ret = clReleaseMemObject(d_evaluation_count);

        ptoc(tag);
        
        return 0;
    }
