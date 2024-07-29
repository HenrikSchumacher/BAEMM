public:

    template<typename R_ext, typename C_ext, typename I_ext>
    void ApplyNearFieldOperators_PL(
        const C_ext alpha, cptr<C_ext> B_in,  const I_ext ldB_in,
        const C_ext beta,  mptr<C_ext> C_out, const I_ext ldC_out,
        const R_ext kappa_,
        const C_ext coeff_0,
        const C_ext coeff_1,
        const C_ext coeff_2,
        const C_ext coeff_3,
        const I_ext wave_count_,
        const I_ext wave_chunk_size_,
        cptr<R_ext> evaluation_points_, 
        const I_ext evaluation_count_
    )
    {
        // Computes
        //
        //     C_out = alpha * A * B_in + beta * C_out,
        //
        // where B_in and C_out out are matrices of size vertex_count x wave_count_ and
        // represent the vertex values of  wave_count_ piecewise-linear functions.
        // The operator A is a linear combination of several operators, depending on kappa:
        //
        // A =   coeff_0 * MassMatrix
        //     + coeff_1 * SingleLayerOp[kappa]
        //     + coeff_2 * DoubleLayerOp[kappa]
        //     + coeff_3 * AdjDblLayerOp[kappa]
        //
        //
        
        CheckInteger<I_ext>();
        CheckScalars<R_ext,C_ext>();
        
        LoadCoefficients(kappa_,coeff_0,coeff_1,coeff_2,coeff_3,wave_count_,wave_chunk_size_);
        
        ApplyNearFieldOperators_PL( alpha, B_in, ldB_in, beta, C_out, ldC_out,
                                    evaluation_points_, evaluation_count_ );
    }

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


    // Applies the boundary operators in the WEAK FORM to the input pointer
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

            // TODO: This file is in the Helmholtz_Common directory.
            // TODO: No OpenCL code should show up here.
            
            //initialize evaluation point buffers

            // TODO: This is an allocation. Where is the corresponding deallocation?
            // TODO: In principle, this can be allocated several times, no?
            // TODO: So is there a memory leak?
            cl_mem evaluation_points_pin = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, 4 * evaluation_count * sizeof(Real), nullptr, &ret);

            Real * evaluation_points_ptr = (Real*)clEnqueueMapBuffer(command_queue, evaluation_points_pin, CL_TRUE, CL_MAP_WRITE, 0, 4 * evaluation_count * sizeof(Real), 0, nullptr, nullptr, nullptr);

            ParallelDo(
                [=]( const Int i )
                {
                    evaluation_points_ptr[4*i+0] = static_cast<Real>(evaluation_points_[3*i+0]);
                    evaluation_points_ptr[4*i+1] = static_cast<Real>(evaluation_points_[3*i+1]);
                    evaluation_points_ptr[4*i+2] = static_cast<Real>(evaluation_points_[3*i+2]);
                    evaluation_points_ptr[4*i+3] = zero;
                },
                evaluation_count, CPU_thread_count
            );
                    
            // Apply off-diagonal part of integral operators.
            NearFieldOperatorKernel( evaluation_points_ptr, evaluation_count, kappa, c );

            
            combine_matrices(
                alpha, C_ptr, wave_count,
                beta,  C_out, ldC,
                evaluation_count, wave_count, CPU_thread_count                                                                                     
            );
        }
        
        ptoc(ClassName()+"::ApplyNearFieldOperators_PL");
    }
