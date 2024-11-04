public:

    /**
     * Applies the boundary POTENTIAL operators to the input B_in (representing a Ppiecewise linear continuous function on the mesh), i.e. 
     * Computes C_out = alpha * A * B_in + beta * C_out, and evaluates the result on a set of points (evaluation_points_).
     *
     * where B_in and C_out out are matrices of size vertex_count x wave_count_ and
     * represent the vertex values of  wave_count_ piecewise-linear functions.
    * The operator A is a linear combination of several operators, depending on kappa:
     *
     * A = coeff_(-,1) * SingleLayerOperator
     *     + coeff(-,2) * DoubleLayerOperator
     * 
     * The canonical choices would be alpha = 1 and beta = 0.
     * 
     * @tparam I_ext: External integer type.
     * @tparam R_ext: External Real type.
     * @tparam C_ext: External Complex type.
     * @param B_in: Input array of size meas_count*wave_count_ - Herglotz wave kernel.
     * @param ldB_in: Leading dimension of input. Usually wave_count_. 
     * @param C_out: Output array.
     * @param ldC_out: Leading dimension of output. Usually wave_count_. 
     * @param kappa_list: An (wave_count_/wave_chunk_size_) x 1 complex array representing the wavenumbers.
     * @param coeff_list: An (wave_count_/wave_chunk_size_) x 4 complex array representing the used combination of operators (by the second and third columns).
     * @param evaluation_points_: An evaluation_count_ x 3 real array for parsing the evaluation points.
     */
    template<typename R_ext, typename C_ext, typename I_ext>
    void ApplyNearFieldOperators_PL(
        const C_ext alpha, cptr<C_ext> B_in,  const I_ext ldB_in,
        const C_ext beta,  mptr<C_ext> C_out, const I_ext ldC_out,
        const R_ext * kappa_list,
        const C_ext * coeff_list,
        const I_ext wave_count_,
        const I_ext wave_chunk_size_,
        const R_ext* evaluation_points_, 
        const I_ext evaluation_count_
    )
    {
        //  The same as above, but with several wave numbers kappa_list and several coefficients.

        CheckInteger<I_ext>();
        CheckScalars<R_ext,C_ext>;
        
        LoadParameters( kappa_list, coeff_list, wave_count_, wave_chunk_size_ );
        
        ApplyNearFieldOperators_PL( 
            alpha, B_in, ldB_in,
            beta , C_out, ldC_out,
            evaluation_points_, evaluation_count_ );
    }


    /** @brief Applies the boundary potential operators. Assumes that Assumes that 'LoadParameters' has been called before.
     */ 
    template<typename R_ext, typename C_ext, typename I_ext>
    void ApplyNearFieldOperators_PL(
        const C_ext alpha, cptr<C_ext> B_in,  const I_ext ldB_in_,
        const C_ext beta,  mptr<C_ext> C_out, const I_ext ldC_out_,
        const R_ext * evaluation_points_, const I_ext evaluation_count_
    )
    {
        // The same as above, but assumes that
        CheckInteger<I_ext>();
        CheckComplex<C_ext>();

        ptic(ClassName()+"::ApplyNearFieldOperators_PL");
        
        if( wave_chunk_count < 1 )
        {
            ptoc(ClassName()+"::ApplyNearFieldOperators_PL");
            return;
        }
        
    
        const Int ldB_in  = int_cast<Int>(ldB_in_ );
//        const Int ldC_out = int_cast<Int>(ldC_out_);
        const Int evaluation_count = int_cast<Int>(evaluation_count_);

        RequireBuffersNearField( wave_count, evaluation_count_ );
        
        if( Re_single_layer || Im_single_layer ||
            Re_double_layer || Im_double_layer
        )
        {
            // use averaging operator to get from PL to PC boundary functions
            AvOp.Dot(
                Scalar::One <Complex>, B_in,  ldB_in,
                Scalar::Zero<Complex>, B_ptr, ldB,
                ldB
            );

            ModifiedB();
            C_loaded = true;

            // initialize evaluation point buffers
            Real * & evaluation_points_ptr;
            InitializeEvaluationPointBuffer(evaluation_count, evaluation_points_ptr, evaluation_points_);
 
            // Apply integral operators.
            NearFieldOperatorKernel( evaluation_points_, evaluation_count, kappa, c );
            
            UnmapEvaluationPointBuffer(evaluation_points_ptr);

            combine_matrices(
                alpha, C_ptr, wave_count,
                beta,  C_out, ldC,
                evaluation_count, wave_count, CPU_thread_count                                                                                     
            );
        }
        
        ptoc(ClassName()+"::ApplyNearFieldOperators_PL");
    }
