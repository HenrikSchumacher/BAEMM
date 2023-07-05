# BÄMM!
BÄMM! - a brute force approach for the boundary element method

Operator application:
In the following we use the following typename aliases:
    C_ext: user defined complex type
    R_ext: user defined real type
    I_ext: user defined int type

Boundary operators:
    You, for single uses of the operator, applicate the boundary operators normally on a PL-function by calling

    ApplyBoundaryOperators_PL(
            const C_ext alpha, ptr<C_ext> B_in,  const I_ext ldB_in,
            const C_ext beta,  mut<C_ext> C_out, const I_ext ldC_out,
            const R_ext * kappa_list,
            const C_ext * coeff_list,
            const I_ext wave_count_,
            const I_ext wave_chunk_size_
        )

    Where kappa_list has length 
    wave_chunk_count = wave_count_/wave_chunk_size
    and coeff list is a 2D-array of size wave_chunk_count x 4 (where the coulumns represent the factors of M,SL,DL,adjDL). M denotes the PL mass matrix.
    This function loads the latter parameters into global scope, compiles the kernel, applies the resulting operator to B_in and copies the result into C_out.
    Note that this is the weak form, hence for the strong form one has to apply M^(-1) to the result.

    (Only OpenCL)
    For multiple uses of the SAME operator (as for instance in a linear solver) there is another way to apply the kernel:

    kernel_list LoadKernel(
                    const R_ext * kappa_,
                    const C_ext * c_,
                    const I_ext wave_count_,
                    const I_ext wave_chunk_size_  
                    ) 

    Loads the kernel, compiles, uploads the constant buffers and returns a struct with the buffer information.

    void ApplyBoundaryOperators_PL(
            const I_ext ld_in_,
            const C_ext alpha, ptr<C_ext> B_in,
            const C_ext beta,  mut<C_ext> C_out
        )
    Is then the reduces application of the boundary operators to this fixed environment.
    IMPORTANT: after being finished with the kernel one also has to "destroy it to release the device buffers by calling:

    void DestroyKernel(
                    kernel_list* list
            )


Boundary to Farfield map:

    The application works quite exactly as for the boundary operators:

    ApplyFarFieldOperators_PL(
                const C_ext alpha, ptr<C_ext> B_in,  const I_ext ldB_in,
                const C_ext beta,  mut<C_ext> C_out, const I_ext ldC_out,
                const R_ext * kappa_list,
                const C_ext * coeff_list,
                const I_ext wave_count_,
                const I_ext wave_chunk_size_
            )

    Note that as there are only single- and double layer far field operators, the coefficients in the "mass" and the "adjoint DL" rows of coeff_list will be ignored.

Wave function assembly:

    For the (CPU) assembly of the standard incident waves with entries (c[1] + i*k*c[2]*<d,n>)*e^(i*k*<d,x>) one uses

    template<typename R_ext, typename C_ext, typename I_ext>
        void CreateIncidentWave_PL(
            const C_ext alpha, ptr<R_ext> incident_directions,  const I_ext inc_count,
            const C_ext beta,  mut<C_ext> C_out, const I_ext ldC_out,
            const R_ext * kappa_list,
            const C_ext * coeff_list,
            const I_ext wave_count_,
            const I_ext wave_chunk_size_
        )

    where inc_count = wave chunk_count is necessary and excess wave vectors will be ignored.

    To assemble a Herglotz wave function with kernel B_in via the GPU one needs to call

    template<typename R_ext, typename C_ext, typename I_ext>
        void CreateHerglotzWave_PL(
            const C_ext alpha, ptr<C_ext> B_in,  const I_ext ldB_in,
            const C_ext beta,  mut<C_ext> C_out, const I_ext ldC_out,
            const R_ext * kappa_list,
            const C_ext * coeff_list,
            const I_ext wave_count_,
            const I_ext wave_chunk_size_
        )

    coeff[:][0] and coeff[:][3] will again be ignored for all wave assemblies.

Far Field Operators:
The far field operator, its directional derivative and the $L^2$-adjoint of its directional derivative are implemented in FarField.hpp. The routine GaussNewton calculates (M + DF*DF)^{-1} for M to be a scaled metric operator.
Also there are executables in the "main" folder for the direct application from "ouside". They read their data from data.txt (which contains data as specified in WriteFiles), simplices.bin, meas_direction.bin and coords.bin and the differential operators read their input from B.bin. The result is again written to B.bin. GaussNewtonStep calculates (regpar*M + DF*DF)^{-1}(-1)(DF*(res) + regpar * DE) for the Tangent-Point-Energy and a proper metric.