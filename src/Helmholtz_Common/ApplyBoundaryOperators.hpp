public:
    /**
     * Applies the WEAK boundary operators SL, DL and DL' to some input piecewise linear, continuous function on the surface, 
     * i.e. computes C_out = alpha * A * B_in + beta * C_out,
     * where B_in and C_out out are matrices of size vertex_count x wave_count_ and
     * represent the vertex values of  wave_count_ piecewise-linear functions.
     * The operator A is a linear combination of several operators, depending on kappa:
     *
     * A = coeff_list(.,0) * MassMatrix
     *     + coeff_list(.,1) * SingleLayerOperator
     *     + coeff_list(.,2) * DoubleLayerOperator
     *     + coeff_list(.,3) * AdjointDoubleLayerOperator
     * 
    * @tparam WC Number of right hand sides for the used GMRES- and CG-algorithms, shall either be =0 or =wave_chunk_count_ * wave_chunk_size_.
    * @tparam I_ext External integer type.
    * @tparam R_ext External Real type.
    * @tparam C_ext External Complex type.
     * @param B_in Input array of size vertex_count * wave_count_.
     * @param ldB_in Leading dimension of input. Usually wave_count_. 
     * @param C_out Output array of size vertex_count * wave_count_.
     * @param ldC_out Leading dimension of output. Usually wave_count_. 
     * @param kappa_list An (wave_count_/wave_chunk_size_) x 1 Complex array representing the wavenumbers.
     * @param coeff_list An (wave_count_/wave_chunk_size_) x 4 Complex array representing the used combination of boundary operators.
     */
    template<Int WC = VarSize, typename R_ext, typename C_ext, typename I_ext>
    void ApplyBoundaryOperators_PL(
        const C_ext alpha, cptr<C_ext> B_in,  const I_ext ldB_in,
        const C_ext beta,  mptr<C_ext> C_out, const I_ext ldC_out,
        cptr<R_ext> kappa_list,
        cptr<C_ext> coeff_list,
        const I_ext wave_count_,
        const I_ext wave_chunk_size_
    )
    {
        CheckInteger<I_ext>();
        CheckScalars<R_ext,C_ext>();
        
        LoadBoundaryOperators_PL(kappa_list, coeff_list, wave_count_, wave_chunk_size_);
        
        ApplyBoundaryOperators_PL<WC>( alpha, B_in, ldB_in, beta, C_out, ldC_out );
        
        UnloadBoundaryOperators_PL();
    }

    /**
     * Loads boundary operators. Can be used ist they are called frequently with the same coeffeicients to reduce the just-in-time compilations of the OpenCL-code.
     */
    template<typename R_ext, typename C_ext, typename I_ext>
    void LoadBoundaryOperators_PL(
        cptr<R_ext> kappa_list,
        cptr<C_ext> coeff_list,
        const I_ext wave_count_,
        const I_ext wave_chunk_size_
    )
    {
        LoadParameters(kappa_list, coeff_list, wave_count_, wave_chunk_size_);
        
        // Every Helmholtz_* class has to implement this.
        
        LoadBoundaryOperatorKernel_PL();
    }

    /**
     * Unloads boundary operators (cleans up OpenCL-allocations for new use of the boundary operators).
     */
    void UnloadBoundaryOperators_PL()
    {
        // Every Helmholtz_* class has to implement this.
        
        UnloadBoundaryOperatorKernel_PL();
    }

    /** Applies the boundary operators in the WEAK FORM to the input pointer.
    * Assumes that `LoadBoundaryOperators_PL` has been called before.
    */
    template<Int WC = VarSize, typename C_ext, typename I_ext>
    void ApplyBoundaryOperators_PL(
        const C_ext alpha, cptr<C_ext> B_in,  const I_ext ldB_in_,
        const C_ext beta,  mptr<C_ext> C_out, const I_ext ldC_out_
    )
    {
        CheckInteger<I_ext>();
        CheckComplex<C_ext>();
        
        std::string tag = ClassName()+"::ApplyBoundaryOperators_PL" +
            + "<" + ToString(VarSize)
            + "," + TypeName<C_ext>
            + "," + TypeName<I_ext>
            + ">";
        
        ptic(tag);
        
        if( wave_chunk_count < 1 )
        {
            ptoc(tag);
            return;
        }
        
        
        if( (WC > VarSize) && (WC != wave_count) )
        {
            eprint( tag + "WC != wave_count. Doing nothing." );
            ptoc(tag);
            return;
        }
    
        const Int ldB_in  = int_cast<Int>(ldB_in_ );
        const Int ldC_out = int_cast<Int>(ldC_out_);
        
        Scalar::Complex<C_ext> addTo = Scalar::Zero<C_ext>;

        RequireBuffers( wave_count );
        
        if( Re_single_layer || Im_single_layer ||
            Re_double_layer || Im_double_layer ||
            Re_adjdbl_layer || Im_adjdbl_layer
        )
        {
            // use averaging operator to get from PL to PC boundary functions
            AvOp.Dot<WC>(
                Scalar::One <Complex>, B_in,  ldB_in,
                Scalar::Zero<Complex>, B_ptr, ldB,
                wave_count
            );

            ModifiedB();
            C_loaded = true;

            // Apply off-diagonal part of integral operators.
            BoundaryOperatorKernel();

            // Apply diagonal of single layer operator.
            ApplySingleLayerDiagonal( kappa, c );
            
            // use transpose averaging operator to get from PC to PL boundary functions
            AvOpTransp.Dot<WC>(
                alpha, C_ptr, ldC,
                beta,  C_out, ldC_out,
                wave_count
            );
            
            addTo = Scalar::One<C_ext>;
        }
        else
        {
            addTo = beta * Scalar::One<C_ext>;
        }
        
        // Apply mass matrix.
        if( Re_mass_matrix || Im_mass_matrix )
        {
            for( Int chunk = 0; chunk < wave_chunk_count - 1; ++chunk )
            {
                const Scalar::Complex<C_ext> factor
                        = alpha * static_cast<C_ext>(c[chunk][0]);
                
                MassOp.Dot<VarSize>(
                    factor, &B_in [wave_chunk_size * chunk], ldB_in,
                    addTo,  &C_out[wave_chunk_size * chunk], ldC_out,
                    wave_chunk_size
                );
            }
            {
                const Int chunk = wave_chunk_count - 1;
                const Scalar::Complex<C_ext> factor
                        = alpha * static_cast<C_ext>(c[chunk][0]);
                
                MassOp.Dot<VarSize>(
                    factor, &B_in [wave_chunk_size * chunk], ldB_in,
                    addTo,  &C_out[wave_chunk_size * chunk], ldC_out,
                    wave_count - wave_chunk_size*chunk
                );
            }
            
        }
        
        ptoc(tag);
    }
