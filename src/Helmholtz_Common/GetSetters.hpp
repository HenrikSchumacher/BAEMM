public:
    
    Int VertexCount() const
    {
        return vertex_count;
    }
    
    Int SimplexCount() const
    {
        return simplex_count;
    }
    
    const Sparse_T & MassMatrix() const
    {
        return Mass;
    }

    ptr<Real> Areas() const
    {
        return areas_ptr;
    }


public:

    template<typename C>
    void LoadCoefficients( const std::array<C,4> & coeff )
    {
        // We have to process the coefficients anyways.
        // Hence we can already multiply by one_over_four_pi so that the kernels don't have to do that each time. (The performance gain is not measureable, though.)
        
        SetMassMatrixCoefficient ( static_cast<Complex>(coeff[0]) );
        SetSingleLayerCoefficient( static_cast<Complex>(coeff[1]) );
        SetDoubleLayerCoefficient( static_cast<Complex>(coeff[2]) );
        SetAdjDblLayerCoefficient( static_cast<Complex>(coeff[3]) );
    }


    Complex GetMassMatrixCoefficient() const
    {
        return Complex( c[0][0], c[0][1] );
    }

    void SetMassMatrixCoefficient( const Complex & z )
    {
        c_C[0]  = z;
        c[0][0] = real(z);
        c[0][1] = imag(z);
        mass = (c[0][0] != zero) || (c[0][1] != zero);
    }


    Complex GetSingleLayerCoefficient() const
    {
        return c_C[1] * four_pi;
    }

    void SetSingleLayerCoefficient( const Complex & z )
    {
        c_C[1]  = z * one_over_four_pi;
        c[1][0] = real(c_C[1]);
        c[1][1] = imag(c_C[1]);
        single_layer = (c[1][0] != zero) || (c[1][1] != zero);
    }


    Complex GetDoubleLayerCoefficient() const
    {
        return c_C[2] * four_pi;
    }

    void SetDoubleLayerCoefficient( const Complex & z )
    {
        c_C[2]  = z * one_over_four_pi;
        c[2][0] = real(c_C[2]);
        c[2][1] = imag(c_C[2]);
        double_layer = (c[2][0] != zero) || (c[2][1] != zero);
    }

    Complex GetAdjDblLayerCoefficient() const
    {
        return c_C[3] * four_pi;
    }


    void SetAdjDblLayerCoefficient( const Complex & z )
    {
        c_C[3]  = z * one_over_four_pi;
        c[3][0] = real(c_C[3]);
        c[3][1] = imag(c_C[3]);
        adjdbl_layer = (c[3][0] != zero) || (c[3][1] != zero);
    }



    Int GetWaveChunkSize() const
    {
        return wave_chunk_size;
    }

    void SetWaveChunkSize( const Int wave_chunk_size_ )
    {
        RequireBuffers( wave_count, wave_chunk_size_ );
    }
