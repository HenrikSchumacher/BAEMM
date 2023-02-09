public:
    
    template<typename T_in, typename T_out>
    void ApplyBoundaryOperators_PL(
        const Complex alpha, ptr<T_in>  B_in,  const Int ldB_in,
        const Complex beta,  mut<T_out> C_out, const Int ldC_out,
        const std::vector<Real>      & kappa,
        const std::array <Complex,4> & coeff,
        const Int wave_count_
    )
    {
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
        
        Complex addTo = Scalar::Zero<Complex>;

        if( single_layer || double_layer || adjdbl_layer )
        {
            RequireBuffers( wave_count_, wave_chunk_size );
            
            AvOp.Dot(
                Scalar::One <Complex>, B_in,  ldB_in,
                Scalar::Zero<Complex>, B_ptr, ldB,
                wave_count
            );
            B_loaded = true;
            ModifiedB();
            
            BoundaryOperatorKernel_C( kappa );
            
            // TODO: Apply diagonal part of single layer boundary operator.
            
            // TODO: Is there some diagonal part of double layer and adjdbl boundary operator?
            
            AvOpTransp.Dot(
                alpha, C_ptr, ldC,
                beta,  C_out, ldC_out,
                wave_count
            );
            C_loaded = true;
            
            addTo = Scalar::One<Complex>;
        }
        if( mass )
        {
            Mass.Dot(
                alpha * Complex(c[0][0],c[0][1]), B_in, ldB_in, addTo, C_out, ldC_out, wave_count
            );
        }
        
        toc(ClassName()+"::ApplyBoundaryOperators_PL");
        ptoc(ClassName()+"::ApplyBoundaryOperators_PL");
    }
    
    // Overload for just one wave number
    template<typename T_in, typename T_out>
    void ApplyBoundaryOperators_PL(
        const Complex alpha, ptr<T_in>  B_in,  const Int ldB_in,
        const Complex beta,  mut<T_out> C_out, const Int ldC_out,
        const Real kappa,
        const std::array <Complex,4> & coeff,
        const Int wave_count_
    )
    {
        std::vector<Real> kappa_list ( DivideRoundUp(wave_count_, wave_chunk_size), kappa  );
        
        ApplyBoundaryOperators_PL(
            alpha, B_in,  ldB_in,
            beta,  C_out, ldC_out,
            kappa_list, coeff, wave_count_
        );
    }
