public:

    template<typename R_ext, typename C_ext, typename I_ext>
    void ApplyOperators_PL(
        const C_ext alpha, const B_ext * B_in,  const I_ext ldB_in,
        const C_ext beta,        C_ext * C_out, const I_ext ldC_out,
        const I_ext cols
    )
    {
        // Computes
        //
        //     C_out = alpha * A * B_in + beta * C_out,
        //
        // where A is the averaging operator and
        // where B_in and C_out out are matrices of size vertex_count x cols and
        // simplex_count x cols, respectively.
     
        AvOp.Dot(
            Scalar::One<Complex>,  B_in,  ldB_in,
            Scalar::Zero<Complex>, C_out, ldC_out,
            cols
        );

    }
