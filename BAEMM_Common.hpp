namespace BAEMM
{
    using namespace Tools;
    using namespace Tensors;
    using namespace Repulsor;
    
    enum class WaveType : unsigned char
    {
        Plane  = 201,
        Radial = 202
    };
    
    
    template<typename I>
    inline void CheckInteger()
    {
        static_assert( std::is_integral_v<I>, "" );
    }
    
    template<typename R>
    inline void CheckReal()
    {
        static_assert( Scalar::FloatQ<R>, "" );
        static_assert( Scalar::RealQ<R>, "" );
    }
    
    template<typename C>
    inline void CheckComplex()
    {
        static_assert( Scalar::FloatQ<C>, "" );
        static_assert( Scalar::ComplexQ<C>, "" );
    }
    
    template<typename R, typename C>
    inline void CheckScalars()
    {
        CheckReal<R>();
        CheckComplex<C>();
        static_assert( std::is_same_v<Scalar::Real<C>,R>, "" );
    }

}
