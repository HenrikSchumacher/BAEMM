public:

    // calculates the Hadamard product of two matrices and saves it in the second entry
    // further one can choose to sum over the leading dimension (over the columns)
    template<typename T, typename I>
    void HadamardProduct(ptr<T> A, ptr<T> B , mut<T> C, I rows, I columns, bool ld_sum)
    {   
        I i,j;
        if(!ld_sum)
        {   
            // //CheckThis
            //     zip_buffers(
            //     Zippers::Times<T,T,T>(), A, B, C , rows * columns, CPU_thread_count
            // );
        }
        else
        {
            //CheckThis
            ParallelDo(
                [=]( const I i )
                {
                    C[i] = dot_buffers( &A[i * columns], &B[i * columns], columns );
                },
                int_cast<I>(rows),
                int_cast<I>(CPU_thread_count)
            );
        }
    }
