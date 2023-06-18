public:

    template<typename S, typename T, typename I>
    void type_cast(T* C, const S* B, I length, I CPU_thread_count)
    {
        copy_buffer(B, C, length, CPU_thread_count);
    }
