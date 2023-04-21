public:
        struct kernel_list{
                cl_mem d_kappa;
                cl_mem d_coeff;
                cl_mem d_n;
                cl_mem d_wave_count;
        };

        // the idea of this host code is to keep the compiled program and uploaded buffers as far as possible to accelerate
        // the use in a linear solver where a variability on sizes and coefficients is unneeded in the iterations
        int BoundaryOperatorKernel_C(
                ) 
        {
                // write to buffers
                clEnqueueWriteBuffer(command_queue, B_buf, CL_FALSE, 0,
                                        rows_rounded * wave_count * sizeof(Complex), B_ptr, 0, NULL, NULL);
                clFinish(command_queue);

                // Execute the OpenCL kernel on the list
                size_t local_item_size = block_size;
                size_t global_item_size = rows_rounded;
                ret = clEnqueueNDRangeKernel(command_queue, global_kernel, 1, NULL, 
                        &global_item_size, &local_item_size, 
                        0, NULL, NULL);
                // Read the memory buffer C on the device to the local variable C
                ret = clEnqueueReadBuffer(command_queue, C_buf, CL_TRUE, 0, 
                        wave_count * simplex_count * sizeof(Complex), C_ptr, 0, NULL, NULL);
                        
                clFinish(command_queue);
                return 0;
        }

        template<typename R_ext, typename C_ext, typename I_ext>
        kernel_list LoadKernel(
                const R_ext * kappa_,
                const C_ext * c_,
                const I_ext wave_count_,
                const I_ext wave_chunk_size_  
                ) 
        {
            ASSERT_INT(I_ext);
            ASSERT_REAL(R_ext);
            ASSERT_COMPLEX(C_ext);
        
            LoadParameters(kappa_, c_, wave_count_, wave_chunk_size_);

            RequireBuffers( wave_count );


            Real* Kappa = (Real*)malloc(wave_chunk_count * 4 * sizeof(Real));
            memcpy(Kappa, kappa.data(), 4 * sizeof(Real));
            Complex* Coeff = (Complex*)malloc(wave_chunk_count * 4 * sizeof(Complex));
            memcpy(Coeff, c.data(), wave_chunk_count * 4 * sizeof(Complex));

            int n = simplex_count;

            size_t max_work_group_size; //check for maximal size of work group
            ret = clGetDeviceInfo(
                            device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), 
                            &max_work_group_size, NULL);

            if (block_size > max_work_group_size)
            {
                    SetBlockSize(max_work_group_size);
            }
            
            // Load the kernel source code into the array source_str
            char *source_str;
            size_t source_size;

            source_str = manipulate_string(
#include "BoundaryOperatorKernel_C.cl"                
                    ,Coeff,block_size,wave_chunk_size,source_size);

            // Create the rest of the memory buffers on the device for each vector 
            cl_mem d_kappa = clCreateBuffer(context, CL_MEM_READ_ONLY,
                    wave_chunk_count * sizeof(Real), NULL, &ret);
            cl_mem d_coeff = clCreateBuffer(context, CL_MEM_READ_ONLY,
                    4 * wave_chunk_count * sizeof(Complex), NULL, &ret);
            cl_mem d_n = clCreateBuffer(context, CL_MEM_READ_ONLY,
                    sizeof(Int), NULL, &ret);
            cl_mem d_wave_count = clCreateBuffer(context, CL_MEM_READ_ONLY,
                    sizeof(Int), NULL, &ret);
        
            clEnqueueWriteBuffer(command_queue, d_kappa, CL_FALSE, 0,
                                        wave_chunk_count * sizeof(Real), Kappa, 0, NULL, NULL);
            clEnqueueWriteBuffer(command_queue, d_coeff, CL_FALSE, 0,
                                        4 * wave_chunk_count * sizeof(Complex), Coeff, 0, NULL, NULL);
            clEnqueueWriteBuffer(command_queue, d_n, CL_FALSE, 0,
                                        sizeof(Int), &n, 0, NULL, NULL);
            clEnqueueWriteBuffer(command_queue, d_wave_count, CL_FALSE, 0,
                                        sizeof(Int), &wave_count, 0, NULL, NULL);

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
                    printf("%s", result);
            }
                    
            // Create the OpenCL kernel
            global_kernel = clCreateKernel(program, "BoundaryOperatorKernel_C", &ret);
            
            // Set the arguments of the kernel
            ret = clSetKernelArg(global_kernel, 0, sizeof(cl_mem), (void *)&mid_points);
            ret = clSetKernelArg(global_kernel, 1, sizeof(cl_mem), (void *)&normals);
            ret = clSetKernelArg(global_kernel, 2, sizeof(cl_mem), (void *)&B_buf);
            ret = clSetKernelArg(global_kernel, 3, sizeof(cl_mem), (void *)&C_buf);
            ret = clSetKernelArg(global_kernel, 4, sizeof(cl_mem), (void *)&d_kappa);
            ret = clSetKernelArg(global_kernel, 5, sizeof(cl_mem), (void *)&d_coeff);
            ret = clSetKernelArg(global_kernel, 6, sizeof(cl_mem), (void *)&d_n);
            ret = clSetKernelArg(global_kernel, 7, sizeof(cl_mem), (void *)&d_wave_count);
            clFinish(command_queue);
            
            kernel_list list;
            list.d_kappa = d_kappa;
            list.d_coeff = d_coeff;
            list.d_n = d_n;
            list.d_wave_count = d_wave_count;
            
            // clean up
            ret = clReleaseProgram(program);
            free(source_str);
            free(Kappa);
            free(Coeff);

            return list;
        }

        void DestroyKernel(
                kernel_list* list
        )
        {               
                // Clean up
                ret = clFinish(command_queue);
                ret = clFlush(command_queue);
                ret = clReleaseKernel(global_kernel);
                ret = clReleaseMemObject(list->d_kappa);
                ret = clReleaseMemObject(list->d_coeff);
                ret = clReleaseMemObject(list->d_n);
                ret = clReleaseMemObject(list->d_wave_count);
        }
        