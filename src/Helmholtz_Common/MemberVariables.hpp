private:

    const Int CPU_thread_count = 1;

    const Int vertex_count;
    const Int simplex_count;
    const Int meas_count = 0;
    
    Tensor2<Real,Int> vertex_coords;
    Tensor2<Int ,Int> triangles;
    Tensor1<Real,Int> areas_lumped_inv;

    WaveNumberContainer_T kappa;
    WaveNumberContainer_T kappa3;
    
    CoefficientContainer_T c;
    CoefficientContainer_T c3;

    Real * restrict areas_ptr           = nullptr;
    Real * restrict mid_points_ptr      = nullptr;
    Real * restrict normals_ptr         = nullptr;
    Real * restrict single_diag_ptr     = nullptr;
    Real * restrict tri_coords_ptr      = nullptr;
    Real * restrict meas_directions_ptr = nullptr;

    Complex * restrict B_ptr = nullptr;
    Complex * restrict C_ptr = nullptr;

    Sparse_T AvOp;
    Sparse_T AvOpTransp;

    Sparse_T CurlOp;
    Sparse_T CurlOpTransp;

    Sparse_T MassOp;

    bool B_loaded        = false;
    bool C_loaded        = false;

    bool Re_mass_matrix  = false;
    bool Im_mass_matrix  = false;
    bool Re_single_layer = false;
    bool Im_single_layer = false;
    bool Re_double_layer = false;
    bool Im_double_layer = false;
    bool Re_adjdbl_layer = false;
    bool Im_adjdbl_layer = false;

    bool use_diagonal = true;

    Int wave_chunk_size    = 16;
    Int wave_chunk_count   =  0;
    Int wave_count         =  0;
    Int ldB                =  0;
    Int ldC                =  0;

    Int block_size         = 64;
    Int block_count        =  0;
    Int rows_rounded       =  0;


public:

//    Real cg_tol             = ?;
//    Real gmres_tol          = ?;
    Int  gmres_max_iter     =  30;
    Int  gmres_max_restarts = 10;
