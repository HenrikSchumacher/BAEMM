public:

        int FarFieldOperatorKernel_C(
            const WaveNumberContainer_T  & kappa_,
            const CoefficientContainer_T & c_
        )
        {
            std::string tag = ClassName()+"::FarFieldOperatorKernel_C";
            
            ptic(tag);
            
            if( kappa_.Dimension(0) != wave_chunk_count )
            {
                eprint(tag + ": kappa_.Dimension(0) != wave_chunk_count.");
            }
            
            if( c_.Dimension(0) != wave_chunk_count )
            {
                eprint(tag + ": c_.Dimension(0) != wave_chunk_count.");
            }
            
            // Allocate local host pointers for the device buffers to use.
            // Henrik: I am not sure whether this is necessary.
            Tensor1<Real,Int> Kappa ( wave_chunk_count );
            kappa_.Write(Kappa.data());
            
            Tensor2<Complex,Int> Coeff ( wave_chunk_count, 4 );
            c_.Write(Coeff.data());

            int n = simplex_count;
            int m = meas_count;

            // Check for maximal size of work group.
            
            std::size_t max_work_group_size;
            ret = clGetDeviceInfo( device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);

            if (block_size > max_work_group_size)
            {
                SetBlockSize(static_cast<Int>(max_work_group_size));
            }
            
            std::string source = CreateSourceString(
#include "FarFieldOperatorKernel_C.cl"
                ,block_size,wave_chunk_size
            );
                
            const char * source_str = source.c_str();
            std::size_t source_size = source.size();

            // Create the rest of the memory buffers on the device for each vector
            cl_mem d_kappa      = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,     wave_chunk_count * sizeof(Real),    Kappa.data(), &ret);
            cl_mem d_coeff      = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, 4 * wave_chunk_count * sizeof(Complex), Coeff.data(), &ret);
            cl_mem d_n          = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int),                            &n,           &ret);
            cl_mem d_m          = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int),                            &m,           &ret);
            cl_mem d_wave_count = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, sizeof(int),                            &wave_count,  &ret);

            // write to buffers
            clEnqueueWriteBuffer(command_queue, B_buf, CL_FALSE, 0, rows_rounded * wave_count * sizeof(Complex), B_ptr, 0, NULL, NULL);


            // Create a program from the kernel source
            cl_program program = clCreateProgramWithSource(context, 1,
                    (const char **)&source_str, (const size_t *)&source_size, &ret);
            
            // Build the program
            ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
            if (ret != 0)
            {
                    char result[16384];
                    size_t size;
                    ret = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(result), &result, &size);
                    printf("%s\n", result);
            }
            // Create the OpenCL kernel
            cl_kernel kernel = clCreateKernel(program, "FarFieldOperatorKernel_C", &ret);

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
            ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
                    
            // Read the memory buffer C on the device to the local variable C
            ret = clEnqueueReadBuffer(command_queue, C_buf, CL_TRUE, 0, wave_count * m * sizeof(Complex), C_ptr, 0, NULL, NULL);

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
