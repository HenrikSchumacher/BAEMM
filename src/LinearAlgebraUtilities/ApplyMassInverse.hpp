public:

    template<size_t solver_count, typename I, typename T, typename R>
    void ApplyMassInverse(const T* B_in, T* C_out, const I ld, const R cg_tol)
    {
        const I   n = VertexCount();

        ConjugateGradient<solver_count,T,size_t> cg(n,200,CPU_thread_count);

        auto id = [&]( const T * x, T *y )
        {
            memcpy(y,x, ld * n * sizeof(T));
        };

        auto mass = [&]( const T * x, T *y )
        {
            Mass.Dot(
                Tools::Scalar::One<T>, x, ld,
                Tools::Scalar::Zero<T>, y, ld,
                ld
            );
        };

        zerofy_buffer(C_out, static_cast<size_t>(ld * n), CPU_thread_count);
        
        (void)cg(mass,id,B_in,ld,C_out,ld,cg_tol);
    }
