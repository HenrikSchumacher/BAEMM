public:

    void BoundaryOperatorKernel_C(
        const WaveNumberContainer_T  & kappa_,
        const CoefficientContainer_T & c_
    )
    {
        if( !B_loaded )
        {
            wprint(ClassName()+"::BoundaryOperatorKernel_C: No values loaded into B. doing nothing.");
        }
        
        switch( wave_chunk_size )
        {
            case 1:
            {
                boundaryOperatorKernel_C<4,2,1>(kappa_,c_);
                break;
            }
            case 2:
            {
                boundaryOperatorKernel_C<4,2,2>(kappa_,c_);
                break;
            }
            case 4:
            {
                boundaryOperatorKernel_C<4,2,4>(kappa_,c_);
                break;
            }
            case 8:
            {
                boundaryOperatorKernel_C<4,2,8>(kappa_,c_);
                break;
            }
            case 16:
            {
                boundaryOperatorKernel_C<4,2,16>(kappa_,c_);
                break;
            }
            case 32:
            {
                boundaryOperatorKernel_C<4,2,32>(kappa_,c_);
                break;
            }
            case 64:
            {
                boundaryOperatorKernel_C<4,2,64>(kappa_,c_);
                break;
            }
            default:
            {
                eprint(ClassName()+"::BoundaryOperatorKernel_C: wave_chunk_size must be a power of 2 that is smaller or equal to 64.");
                break;
            }
        }
    }

    template<
        Int i_blk_size, Int j_blk_size, Int wave_chunk_size
    >
    void boundaryOperatorKernel_C(
        const WaveNumberContainer_T  & kappa_,
        const CoefficientContainer_T & c_
    )
    {
        ptic(ClassName()+"::BoundaryOperatorKernel_C<"+ToString(i_blk_size)+","+ToString(j_blk_size)+">");
        
//        constexpr Int i_chunk_size = i_blk_size * wave_count;

        if( (simplex_count/i_blk_size) * i_blk_size != simplex_count )
        {
            wprint(ClassName()+"::BoundaryOperatorKernel_C: Loop peeling not applied.");
        }
        
        const int k_chunk_size  = wave_chunk_size;
        const int k_chunk_count = kappa_.Size();
        const int k_ld          = kappa_.Size() * k_chunk_size;  // Leading dim of B and C.

        JobPointers<Int> job_ptr(simplex_count/i_blk_size, OMP_thread_count);
        
        for( int k_chunk = 0; k_chunk < k_chunk_count; ++k_chunk )
        {
            #pragma omp parallel for num_threads( OMP_thread_count)
            for( Int thread = 0; thread < OMP_thread_count; ++thread )
            {
                Tiny::Matrix<i_blk_size,4,Real,Int> x;
                Tiny::Matrix<i_blk_size,4,Real,Int> nu;
                Tiny::Matrix<j_blk_size,4,Real,Int> y;
                Tiny::Matrix<j_blk_size,4,Real,Int> mu;
                
                Tiny::Vector<4,Real,Int> z;
                
//                Tiny::Matrix<j_blk_size,k_chunk_size,Complex,Int> B_blk;
                Tiny::Matrix<i_blk_size,k_chunk_size,Complex,Int> C_blk;
                
                
                Tiny::Matrix<i_blk_size,j_blk_size,Complex,Int> A;
                
                const Int i_blk_begin = job_ptr[thread  ];
                const Int i_blk_end   = job_ptr[thread+1];
                
                const Int j_blk_begin = 0;
                const Int j_blk_end   = simplex_count/j_blk_size;
                
                for( Int i_blk = i_blk_begin; i_blk < i_blk_end; ++i_blk )
                {
                    const Int i_base = i_blk_size * i_blk;

                    C_blk.SetZero();
                    
                    x.Read( mid_points.data(i_base) );
                    nu.Read( normals.data(i_base) );
                    
                    
                    for( Int j_blk = j_blk_begin; j_blk < j_blk_end; ++j_blk )
                    {
                        const Int j_base = j_blk_size * j_blk;
                        
                        y.Read( mid_points.data(j_base) );
                        mu.Read( normals.data(j_base) );
                        
                        
                        LOOP_UNROLL_FULL
                        for( Int i = 0; i < i_blk_size; ++i )
                        {
                            const Int i_global = i_base + i;
                            
                            LOOP_UNROLL_FULL
                            for( Int j = 0; j < j_blk_size; ++j )
                            {
                                const Int j_global = j_base + j;
                                
                                const Real delta_ij = static_cast<Real>(
                                    (i_global==j_global)
                                    &&
                                    (i_global<simplex_count)
                                    &&
                                    (j_global<simplex_count)
                                );
                                
                                z[0] = y[j][0] - x[i][0];
                                z[1] = y[j][1] - x[i][1];
                                z[2] = y[j][2] - x[i][2];
                                
                                const Real r = std::sqrt( z[0] * z[0] + z[1] * z[1] + z[2] * z[2] );
                            
                                
                                const Real r_inv = (one-delta_ij)/(r + delta_ij);
                                
                                const Real dot_z_nu = z[0]*nu[i][0] + z[1]*nu[i][1] + z[2]*nu[i][2];
                                const Real dot_z_mu = z[0]*mu[j][0] + z[1]*mu[j][1] + z[2]*mu[j][2];
                                
                                
                                const Real r3_inv = r_inv * r_inv * r_inv;
                                
                                const Complex I_kappa_r {
                                    Scalar::Zero<Real>,
                                    kappa_[k_chunk] * r
                                };
                                
                                const Complex exp_I_kappa_r ( std::exp(I_kappa_r) );
                                
                                const Complex factor { (I_kappa_r - one) * r3_inv };
                                
                                A[i][j] = exp_I_kappa_r * (
                                    c_[k_chunk][1] * r_inv
                                    +
                                    factor * (
                                        c_[k_chunk][2] * dot_z_mu
                                        -
                                        c_[k_chunk][3] * dot_z_nu
                                    )
                                );

                            } // for( Int j = 0; j < j_blk_size; ++j )
                            
                        } // for( Int i = 0; i < i_blk_size; ++i )
                        
                        ptr<Complex> B_block = &B_ptr[k_ld * j_base + k_chunk_size * k_chunk];
                        
                        LOOP_UNROLL_FULL
                        for( Int k = 0; k < k_chunk_size; ++k )
                        {
                            LOOP_UNROLL_FULL
                            for( Int i = 0; i < i_blk_size; ++i )
                            {
                                LOOP_UNROLL_FULL
                                for( Int j = 0; j < j_blk_size; ++j )
                                {
                                    C_blk[i][k] += A[i][j] * B_block[k_ld * j + k];
                                }
                            }
                        }
                        
                        
                    } // for( Int j_blk = j_blk_begin; j_blk < j_blk_end; ++j_blk )

                    
                    LOOP_UNROLL_FULL
                    for( Int i = 0; i < i_blk_size; ++i )
                    {
                        copy_buffer<k_chunk_size>(
                            &C_blk[i][0],
                            &C_ptr[k_ld * (i_base+i) + k_chunk_size * k_chunk]
                        );
                    }
                    
                } // for( Int i_blk = i_blk_begin; i_blk < i_blk_end; ++i_blk )
                
            } // for( Int thread = 0; thread < OMP_thread_count; ++thread )
        }
        
        ptoc(ClassName()+"::BoundaryOperatorKernel_C<"+ToString(i_blk_size)+","+ToString(j_blk_size)+">");
    }
