public:

    // calculates the Hadamard product of two matrices and saves it in the second entry
    // further one can choose to sum over the leading dimension (over the columns)
    template<typename T, typename I>
    void HadamardProduct(ptr<T> A, ptr<T> B , mut<T> C, I rows, I columns, bool ld_sum)
    {   
        I i,j;
        if(!ld_sum)
        {
            #pragma omp parallel for num_threads( OMP_thread_count ) schedule( static ) private(i) private(j)
            for( i = 0; i < rows; ++i )
            {
                LOOP_UNROLL_FULL
                for( j = 0; j < columns; ++j )
                {
                    C[i * columns + j] = A[i * columns + j] * B[i * columns + j];
                }
            }
        }
        else
        {
            #pragma omp parallel for num_threads( OMP_thread_count ) schedule( static ) private(i) private(j)
            for( i = 0; i < rows; ++i )
            {
                C[i] = 0;
                LOOP_UNROLL_FULL
                for( j = 0; j < columns; ++j )
                {
                    C[i] += A[i * columns + j] * B[i * columns + j];
                }
            }
        }
    }