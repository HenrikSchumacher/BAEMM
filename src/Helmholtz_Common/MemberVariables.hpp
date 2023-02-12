private:

    const Int OMP_thread_count = 1;

    const Int vertex_count;
    const Int simplex_count;

    Tensor2<Real,Int> vertex_coords;
    Tensor2<Int ,Int> triangles;

    Real    c   [4][2] = {{}};
    Complex c_C [4]    = {{}};

    Real * restrict areas_ptr      = nullptr;
    Real * restrict mid_points_ptr = nullptr;
    Real * restrict normals_ptr    = nullptr;

    Complex * restrict B_ptr = nullptr;
    Complex * restrict C_ptr = nullptr;

    Sparse_T AvOp;
    Sparse_T AvOpTransp;
    Sparse_T Mass;

    bool B_loaded        = false;
    bool C_loaded        = false;

    bool mass            = false;
    bool Re_single_layer = false;
    bool Im_single_layer = false;
    bool Re_double_layer = false;
    bool Im_double_layer = false;
    bool Re_adjdbl_layer = false;
    bool Im_adjdbl_layer = false;

    Int wave_chunk_size    =  1;
    Int wave_chunk_count   =  0;
    Int wave_count         =  0;
    Int ldB                =  0;
    Int ldC                =  0;

    Int block_size         = 64;
    Int block_count        =  0;
    Int rows_rounded       =  0;

