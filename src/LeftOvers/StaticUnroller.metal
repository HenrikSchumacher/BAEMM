template <uint Begin, uint End, uint Step, class Action>
struct StaticUnroller
{
    static void Call( thread Action & f )
    {
        f(Begin);
        StaticUnroller<Begin+Step, End, Step, Action>::Call( f );
    }
};

template <uint End, uint Step, class Action>
struct StaticUnroller<End, End, Step, Action>
{
    static void Call( thread Action & f ) { }
};
