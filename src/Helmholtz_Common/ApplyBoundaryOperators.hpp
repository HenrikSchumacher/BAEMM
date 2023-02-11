public:
    
    template<typename R_ext, typename C_ext>
    void ApplyBoundaryOperators_PL(
        const C_ext alpha, ptr<C_ext> B_in,  const Int ldB_in,
        const C_ext beta,  mut<C_ext> C_out, const Int ldC_out,
        const std::vector<R_ext>  & kappa,
        const std::array<C_ext,4> & coeff,
        const Int wave_count_
    )
    {
        static_assert( Scalar::IsReal<R_ext>, "Template parameter R_ext has to be a real floating point type." );
        
        static_assert( Scalar::IsComplex<C_ext>, "Template parameter C_ext has to be a complex floating point type." );
        
        ptic(ClassName()+"::ApplyBoundaryOperators_PL");
        tic(ClassName()+"::ApplyBoundaryOperators_PL");
        // TODO: Aim is to implement the following:
        //
        // Computes
        //
        //     C_out = alpha * A * B_in + beta * C_out,
        //
        // where B_in and C_out out are matrices of size vertex_count x wave_count_ and
        // represent the vertex values of  wave_count_ piecewise-linear functions.
        // The operator A is a linear combination of several operators:
        //
        // A =   coeff[0] * MassMatrix
        //     + coeff[1] * SingleLayerOp
        //     + coeff[2] * DoubleLayerOp
        //     + coeff[3] * AdjDblLayerOp
        
        // TODO: Explain how kappa is distributed over this data!
        
        if( kappa.size() != wave_count_ / wave_chunk_size )
        {
            eprint(ClassName()+"::ApplyBoundaryOperators_PL: kappa.size() != wave_count_ / wave_chuck_size.");

            return;
        }

        
        // TODO: Adjust coeffients!

        LoadCoefficients(coeff);
        
        Scalar::Complex<C_ext> addTo = Scalar::Zero<C_ext>;

        if( Re_single_layer || Im_single_layer || Re_double_layer || Im_double_layer || Re_adjdbl_layer || Im_adjdbl_layer )
        {
            RequireBuffers( wave_count_, wave_chunk_size );
            
            AvOp.Dot(
                Scalar::One<Complex>,  B_in,  ldB_in,
                Scalar::Zero<Complex>, B_ptr, ldB,
                wave_count
            );
            B_loaded = true;
            ModifiedB();
            
            // Convert kappa input to intenal type.
            WaveNumberContainer_T kappa_ ( kappa.size() );
            copy_buffer(kappa.data(), kappa_.data(), kappa.size() );
            
            BoundaryOperatorKernel_C( kappa_ );
            
            // TODO: Apply diagonal part of single layer boundary operator.
            
            // TODO: Is there some diagonal part of double layer and adjdbl boundary operator?
            
            AvOpTransp.Dot(
                alpha, C_ptr, ldC,
                beta,  C_out, ldC_out,
                wave_count
            );
            C_loaded = true;
            
            addTo = Scalar::One<C_ext>;
        }
        if( mass )
        {
            const Scalar::Complex<C_ext> factor = alpha * Scalar::Complex<C_ext>(c[0][0],c[0][1]);
            
            Mass.Dot(
                factor, B_in,  ldB_in,
                addTo,  C_out, ldC_out,
                wave_count
            );
        }
        
        toc(ClassName()+"::ApplyBoundaryOperators_PL");
        ptoc(ClassName()+"::ApplyBoundaryOperators_PL");
    }
    
    // Overload for just one wave number
    template<typename R_ext, typename C_ext>
    void ApplyBoundaryOperators_PL(
        const C_ext alpha, ptr<C_ext> B_in,  const Int ldB_in,
        const C_ext beta,  mut<C_ext> C_out, const Int ldC_out,
        const R_ext kappa,
        const std::array<C_ext,4> & coeff,
        const Int wave_count_
    )
    {
        std::vector<Real> kappa_list (
              DivideRoundUp(wave_count_, wave_chunk_size),
              static_cast<Real>(kappa)
        );
        
        ApplyBoundaryOperators_PL(
            alpha, B_in,  ldB_in,
            beta,  C_out, ldC_out,
            kappa_list, coeff, wave_count_
        );
    }
