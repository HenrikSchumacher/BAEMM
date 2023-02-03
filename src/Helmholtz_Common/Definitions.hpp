public:
    
//        using Int        = uint32_t;
    using LInt       = std::size_t;
    using Int        = int;
    using Real       = float;
    using Complex    = std::complex<Real>;
    
    using UInt       = uint32_t;
    
    using Sparse_T = Sparse::MatrixCSR<Complex,Int,LInt>;
    
    using NS::StringEncoding::UTF8StringEncoding;
    
    static constexpr Real zero  = 0;
    static constexpr Real one   = 1;
    static constexpr Real two   = 2;
    static constexpr Real three = 3;
    
    static constexpr Real half     = one / two;
    static constexpr Real third    = one / three;
    static constexpr Real sixth    = half * third;
    static constexpr Real twelveth = half * sixth;
    
    static constexpr Real pi      = M_PI;
    static constexpr Real two_pi  = two * pi;
    static constexpr Real four_pi = two * two_pi;
    
//        static constexpr Real one_over_two_pi  = one / two_pi;
    static constexpr Real one_over_four_pi = one / four_pi;
