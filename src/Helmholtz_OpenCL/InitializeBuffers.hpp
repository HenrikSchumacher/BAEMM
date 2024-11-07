public:

    template<typename R_ext>
    void InitializeBuffers(Int simplex_count, cptr<R_ext> meas_directions_)
    {
        const Int size      =     simplex_count * sizeof(Real);
        const Int size4     = 4 * simplex_count * sizeof(Real);
        const Int msize4    = 4 * meas_count * sizeof(Real);

        // Allocate pinned memory in Host buffer
        mid_points_pin      = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size4,     nullptr, &ret);
        normals_pin         = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, size4,     nullptr, &ret);
        meas_directions_pin = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, msize4,    nullptr, &ret);

        // Map properties to pinned pointers
        mid_points_ptr      = (Real*)clEnqueueMapBuffer(command_queue, mid_points_pin,      CL_TRUE, CL_MAP_WRITE, 0, size4,   0, nullptr, nullptr, nullptr);
        normals_ptr         = (Real*)clEnqueueMapBuffer(command_queue, normals_pin,         CL_TRUE, CL_MAP_WRITE, 0, size4,   0, nullptr, nullptr, nullptr);
        meas_directions_ptr = (Real*)clEnqueueMapBuffer(command_queue, meas_directions_pin, CL_TRUE, CL_MAP_WRITE, 0, msize4,  0, nullptr, nullptr, nullptr);
        areas_ptr           = (Real*)malloc(size);
        single_diag_ptr     = (Real*)malloc(size);
        tri_coords_ptr      = (Real*)malloc(3 * size4);
                                            
        // copy measurement directions
        
        ParallelDo(
            [=]( const Int i )
            {
                meas_directions_ptr[4*i+0] = static_cast<Real>(meas_directions_[3*i+0]);
                meas_directions_ptr[4*i+1] = static_cast<Real>(meas_directions_[3*i+1]);
                meas_directions_ptr[4*i+2] = static_cast<Real>(meas_directions_[3*i+2]);
                meas_directions_ptr[4*i+3] = zero;
            },
            meas_count, CPU_thread_count
        );

        // Allocate memory in device buffer

        mid_points      = clCreateBuffer(context, CL_MEM_READ_ONLY, size4,     nullptr, &ret);
        normals         = clCreateBuffer(context, CL_MEM_READ_ONLY, size4,     nullptr, &ret);
        meas_directions = clCreateBuffer(context, CL_MEM_READ_ONLY, msize4,    nullptr, &ret);
    }
