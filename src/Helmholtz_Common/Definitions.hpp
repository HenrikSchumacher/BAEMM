public:
    
    using Int        = Int32;
    using LInt       = Size_T;
    using Real       = Real32;
    using Complex    = std::complex<Real32>;
    using UInt       = UInt32;
    
    using Sparse_T   = Sparse::MatrixCSR<Real,Int,LInt>;

    using WaveNumberContainer_T  = Tensor1<Real   ,Int>;
    using CoefficientContainer_T = Tensor2<Complex,Int>;
    
    static constexpr Real zero  = 0;
    static constexpr Real one   = 1;
    static constexpr Real two   = 2;
    static constexpr Real three = 3;
    
    static constexpr Real half     = Inv<Real>(two);
    static constexpr Real third    = Inv<Real>(three);
    static constexpr Real sixth    = half * third;
    static constexpr Real twelveth = half * sixth;
    
    static constexpr Real pi      = Scalar::Pi<Real>;
    static constexpr Real two_pi  = two * pi;
    static constexpr Real four_pi = two * two_pi;
    
    static constexpr Real one_over_four_pi = Inv<Real>(four_pi);

    static constexpr bool lumped_mass_as_prec_for_intopsQ = false;
//    static constexpr bool lumped_mass_as_prec_for_intopsQ = true;
    

