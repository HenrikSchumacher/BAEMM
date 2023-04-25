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

    Int GetWaveCount() const
    {
        return wave_count;
    }

    void SetWaveCount( const Int wave_count_ )
    {
        wave_count = wave_count_;
        B_loaded = false;
        C_loaded = false;
    }

    Int GetWaveChunkSize() const
    {
        return wave_chunk_size;
    }

    void SetWaveChunkSize( const Int wave_chunk_size_ )
    {
        wave_chunk_size = wave_chunk_size_;
        B_loaded = false;
        C_loaded = false;
    }

    Int GetWaveChunkCount( const Int wave_count_ ) const
    {
        return CeilDivide(wave_count_, wave_chunk_size);
    }

    Int GetBlockSize() const
    {
        return block_size;
    }

    Int GetMeasCount() const
    {
        return meas_count;
    }

    const WaveNumberContainer_T& GetWaveNumbers() const
    {
        return kappa;
    }

    const CoefficientContainer_T& GetCoefficients() const
    {
        return c;
    }


    void SetBlockSize( const Int block_size_ )
    {
        block_size    = block_size_;
        block_count   = CeilDivide( simplex_count, block_size);
        rows_rounded  = block_count * block_size;

        B_loaded = false;
        C_loaded = false;
    }

    ptr<Real> SingleLayerDiagonal() const
    {
        return single_diag_ptr;
    }

    ptr<Real> TriangleAreas() const
    {
        return areas_ptr;
    }

    ptr<Real> TriangleNormals() const
    {
        return normals_ptr;
    }

    ptr<Real> TriangleMidpoints() const
    {
        return mid_points_ptr;
    }

    void UseDiagonal( const bool use_it )
    {
        use_diagonal = use_it;
    }

