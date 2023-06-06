public:

    template<typename I_ext, typename R_ext>
    void InitializeBuffers(I_ext simplex_count_, const R_ext* meas_directions_)
    {
        Int simplex_count = (Int)simplex_count_;
        const Int size     =     simplex_count * sizeof(Real);
        const Int size4    = 4 * simplex_count * sizeof(Real);
        const Int msize4    = 4 * meas_count * sizeof(Real);

        // Allocate pinned memory in Host buffer
        mid_points_pin = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        size4, NULL, &ret);

        normals_pin = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        size4, NULL, &ret);

        areas_pin = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        size, NULL, &ret);

        single_diag_pin = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        size, NULL, &ret);

        tri_coords_pin = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        3 * size4, NULL, &ret);

        meas_directions_pin = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        msize4, NULL, &ret);

        // Map properties to pinned pointers
        mid_points_ptr      = (Real*)clEnqueueMapBuffer(command_queue,
                                            mid_points_pin, CL_TRUE,
                                            CL_MAP_WRITE, 0, size4, 0,
                                            NULL, NULL, NULL);
        normals_ptr         = (Real*)clEnqueueMapBuffer(command_queue,
                                            normals_pin, CL_TRUE,
                                            CL_MAP_WRITE, 0, size4, 0,
                                            NULL, NULL, NULL);
                                             
        areas_ptr           = (Real*)clEnqueueMapBuffer(command_queue,
                                            areas_pin, CL_TRUE,
                                            CL_MAP_WRITE, 0, size, 0,
                                            NULL, NULL, NULL);
        single_diag_ptr     = (Real*)clEnqueueMapBuffer(command_queue,
                                            single_diag_pin, CL_TRUE,
                                            CL_MAP_WRITE, 0, size, 0,
                                            NULL, NULL, NULL);
        tri_coords_ptr      = (Real*)clEnqueueMapBuffer(command_queue,
                                            tri_coords_pin, CL_TRUE,
                                            CL_MAP_WRITE, 0, 3*size4, 0,
                                            NULL, NULL, NULL);
        meas_directions_ptr = (Real*)clEnqueueMapBuffer(command_queue,
                                            meas_directions_pin, CL_TRUE,
                                            CL_MAP_WRITE, 0, msize4, 0,
                                            NULL, NULL, NULL);
                                            
        // copy measurement directions
        #pragma omp parallel for num_threads( OMP_thread_count ) schedule( static )
        for( Int i = 0; i < meas_count; ++i )
        {
            meas_directions_ptr[4*i+0] = static_cast<Real>(meas_directions_[3*i+0]);
            meas_directions_ptr[4*i+1] = static_cast<Real>(meas_directions_[3*i+1]);
            meas_directions_ptr[4*i+2] = static_cast<Real>(meas_directions_[3*i+2]);
            meas_directions_ptr[4*i+3] = zero;
        }

        // Allocate memory in device buffer
        mid_points = clCreateBuffer(context, CL_MEM_READ_ONLY,
        size4, NULL, &ret);

        normals = clCreateBuffer(context, CL_MEM_READ_ONLY,
        size4, NULL, &ret);

        // *areas = clCreateBuffer(*context, CL_MEM_READ_ONLY,
        // size, NULL, &ret);

        // *single_diag = clCreateBuffer(*context, CL_MEM_READ_ONLY,
        // size, NULL, &ret);

        // *tri_coords = clCreateBuffer(*context, CL_MEM_READ_ONLY,
        // 3 * size4, NULL, &ret);

        meas_directions = clCreateBuffer(context, CL_MEM_READ_ONLY,
        msize4, NULL, &ret);
    }