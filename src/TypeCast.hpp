public:

    template<typename S, typename T, typename I>
    void type_cast(T* C, const S* B, I length, I OMP_thread_count)
    {
        #pragma omp parallel for num_threads( OMP_thread_count ) schedule( static )
        for(I i = 0; i < length; i++)
        {
            std::cout << B[i] << std::endl;
            C[i] = static_cast<T>(B[i]);
            std::cout << C[i] << std::endl;
        }
    }