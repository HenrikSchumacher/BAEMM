public:
        int FarFieldOperatorKernel_C(
                const WaveNumberContainer_T  & kappa_,
                const CoefficientContainer_T & c_       
                ) 
        {
        float* kappa = (float*)malloc(wave_chunk_count * 4 * sizeof(float));
        memcpy(kappa, kappa_.data(), 4 * sizeof(float));
        Complex* coeff = (Complex*)malloc(wave_chunk_count * 4 * sizeof(Complex));
        memcpy(coeff, c_.data(), wave_chunk_count * 4 * sizeof(Complex));

        int n = simplex_count;
        int m = meas_count;
        size_t size4 = 4 * n * sizeof(float);
        size_t size3 = 3 * m * sizeof(float);

        size_t max_work_group_size; //check for maximal size of work group
        ret = clGetDeviceInfo(
                        device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), 
                        &max_work_group_size, NULL);

        if (block_size > max_work_group_size)
        {
                SetBlockSize(max_work_group_size);
        }

        // Load the kernel source code into the array source_str
        std::FILE *fp;
        const char* const_source_str;
        char *source_str, *source_str_temp;
        size_t source_size;

        fp = std::fopen("../HerglotzWaveKernel_C.cl", "r");
        if (!fp) {
                fprintf(stderr, "Failed to load kernel.\n");
                exit(1);
        }
        source_str_temp = (char*)malloc(MAX_SOURCE_SIZE);
        source_size = std::fread( source_str_temp, 1, MAX_SOURCE_SIZE, fp);
        std::fclose( fp );
        const_source_str = source_str_temp;
        source_str = manipulate_string(const_source_str,coeff,block_size,wave_chunk_size,source_size);

        // Create the rest of the memory buffers on the device for each vector 
        cl_mem d_kappa = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                wave_chunk_count * sizeof(float), kappa, &ret);
        cl_mem d_coeff = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                4 * wave_chunk_count * sizeof(Complex), coeff, &ret);
        cl_mem d_n = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                sizeof(int), &n, &ret);
        cl_mem d_m = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                sizeof(int), &m, &ret);
        cl_mem d_wave_count = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                sizeof(int), &wave_count, &ret);

        // write to buffers
        clEnqueueWriteBuffer(command_queue, B_buf, CL_FALSE, 0,
                                rows_rounded * wave_count * sizeof(Complex), B_ptr, 0, NULL, NULL);

        // Create a program from the kernel source
        cl_program program = clCreateProgramWithSource(context, 1, 
                (const char **)&source_str, (const size_t *)&source_size, &ret);

        // Build the program
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        char result[16384];
        size_t size;
        ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(result), &result, &size);
        printf("%s\n", result);
                
        // Create the OpenCL kernel
        cl_kernel kernel = clCreateKernel(program, "HerglotzWaveKernel_C", &ret);
        
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
        size_t local_item_size = block_size;
        size_t global_item_size = rows_rounded;

        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, 
                &global_item_size, &local_item_size, 
                0, NULL, NULL);
        // Read the memory buffer C on the device to the local variable C
        ret = clEnqueueReadBuffer(command_queue, C_buf, CL_TRUE, 0, 
                wave_count * n * sizeof(Complex), C_ptr, 0, NULL, NULL);

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

        free(source_str);
        free(source_str_temp);
        free(coeff);
        return 0;
        }