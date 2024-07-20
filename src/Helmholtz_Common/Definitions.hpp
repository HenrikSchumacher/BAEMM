public:
    
    using Int        = int;
    using LInt       = std::size_t;
    using Real       = float;
    using Complex    = std::complex<Real>;    
    using UInt       = uint32_t;

    static constexpr bool use_lumped_mass_as_precQ = use_lumped_mass_as_precQ_;
    static constexpr bool use_mass_choleskyQ       = use_mass_choleskyQ_;
    
    using Sparse_T   = Sparse::MatrixCSR<Real,Int,LInt>;
    using Cholesky_T = Sparse::CholeskyDecomposition<Real,Int,LInt>;

    using WaveNumberContainer_T  = Tensor1<Real   ,Int>;
    using CoefficientContainer_T = Tensor2<Complex,Int>;
    
    static constexpr Real zero  = 0;
    static constexpr Real one   = 1;
    static constexpr Real two   = 2;
    static constexpr Real three = 3;
    
    static constexpr Real half     = one / two;
    static constexpr Real third    = one / three;
    static constexpr Real sixth    = half * third;
    static constexpr Real twelveth = half * sixth;
    
    static constexpr Real pi      = Scalar::Pi<Real>;
    static constexpr Real two_pi  = two * pi;
    static constexpr Real four_pi = two * two_pi;
    
//        static constexpr Real one_over_two_pi  = one / two_pi;
    static constexpr Real one_over_four_pi = one / four_pi;

    enum class WaveType : unsigned char
    {
        Plane  = 201,
        Radial = 202
    };


