public:

    void DoubleLayer(
        const WaveNumberContainer_T  & kappa_,
        const CoefficientContainer_T & c_
    )
    {
        
        ptic(ClassName()+"::DoubleLayer");
        
        if( !B_loaded )
        {
            wprint(ClassName()+"::DoubleLayer: No values loaded into B. doing nothing.");
        }
        
        JobPointers<Int> job_ptr(simplex_count, OMP_thread_count);

        #pragma omp parallel for num_threads( OMP_thread_count)
        for( Int thread = 0; thread < OMP_thread_count; ++thread )
        {
            const Int i_begin = job_ptr[thread  ];
            const Int i_end   = job_ptr[thread+1];
            
            Tensor1<Complex,Int> C_i_buf (wave_count);
            mut<Complex> C_i = C_i_buf.data();
            
            for( Int i = i_begin; i < i_end; ++i )
            {
                const Tiny::Vector<3,Real,Int> y {
                    mid_points_ptr[4*i+0],
                    mid_points_ptr[4*i+1],
                    mid_points_ptr[4*i+2]
                };

                zerofy_buffer( C_i, wave_count );
                
                for( Int j = 0; j < simplex_count; ++j )
                {
                    // wlog: y = 0;
                    
                    // The vertices of the triangle.
                    const Tiny::Vector<3,Real,Int> x_0 {
                        tri_coords_ptr[12*j+4*0+0] - y[0],
                        tri_coords_ptr[12*j+4*0+1] - y[1],
                        tri_coords_ptr[12*j+4*0+2] - y[2],
                    };
                    const Tiny::Vector<3,Real,Int> x_1 {
                        tri_coords_ptr[12*j+4*1+0] - y[0],
                        tri_coords_ptr[12*j+4*1+1] - y[1],
                        tri_coords_ptr[12*j+4*1+2] - y[2],
                    };
                    const Tiny::Vector<3,Real,Int> x_2 {
                        tri_coords_ptr[12*j+4*2+0] - y[0],
                        tri_coords_ptr[12*j+4*2+1] - y[1],
                        tri_coords_ptr[12*j+4*2+2] - y[2],
                    };
                    
                    // The edge vectors.
                    const Tiny::Vector<3,Real,Int> v_0 { x_2[0] - x_1[0], x_2[1] - x_1[1], x_2[2] - x_1[2] };
                    const Tiny::Vector<3,Real,Int> v_1 { x_0[0] - x_2[0], x_0[1] - x_2[1], x_0[2] - x_2[2] };
                    const Tiny::Vector<3,Real,Int> v_2 { x_1[0] - x_0[0], x_1[1] - x_0[1], x_1[2] - x_0[2] };
                    
                    // The triangle normal.
                    Tiny::Vector<3,Real,Int> nu {
                        normals_ptr[4*j+0],
                        normals_ptr[4*j+1],
                        normals_ptr[4*j+2]
                    };

                    const Real area = areas_ptr[j];
                    
                    // The midpoint.
                    const Tiny::Vector<3,Real,Int> m {
                        mid_points_ptr[4*j+0],
                        mid_points_ptr[4*j+1],
                        mid_points_ptr[4*j+2]
                    };

                    // h = signed height of y over the triangle.
                    const Real h = - Dot( m, nu );


                    const Real factors [3] {
                        Dot(x_1,v_0) / Dot(v_0,v_0),
                        Dot(x_2,v_1) / Dot(v_1,v_1),
                        Dot(x_0,v_2) / Dot(v_2,v_2)
                    };
                    
                    // Orthogonal projection of Y onto triangle's edge opposing x_0.
                    const Tiny::Vector<3,Real,Int> Z_0 {
                        x_0[0] - factors[0] * v_0[0],
                        x_0[1] - factors[0] * v_0[1],
                        x_0[2] - factors[0] * v_0[2]
                    };
                    // Orthogonal projection of Y onto triangle's edge opposing x_1.
                    const Tiny::Vector<3,Real,Int> Z_1 {
                        x_1[0] - factors[1] * v_1[0],
                        x_1[1] - factors[1] * v_1[1],
                        x_1[2] - factors[1] * v_1[2]
                    };
                    // Orthogonal projection of Y onto triangle's edge opposing x_2.
                    const Tiny::Vector<3,Real,Int> Z_2 {
                        x_2[0] - factors[2] * v_2[0],
                        x_2[1] - factors[2] * v_2[1],
                        x_2[2] - factors[2] * v_2[2]
                    };

                    // w_i = nu x Z_i = (F-Z_i) x nu = the height vector of the triangle turned by 90 degree.
                    const Tiny::Vector<3,Real,Int> w_0 {
                        nu[1] * Z_0[2] - nu[2] * Z_0[1],
                        nu[2] * Z_0[0] - nu[0] * Z_0[2],
                        nu[0] * Z_0[1] - nu[1] * Z_0[0]
                    };
                    const Tiny::Vector<3,Real,Int> w_1 {
                        nu[1] * Z_1[2] - nu[2] * Z_1[1],
                        nu[2] * Z_1[0] - nu[0] * Z_1[2],
                        nu[0] * Z_1[1] - nu[1] * Z_1[0]
                    };
                    const Tiny::Vector<3,Real,Int> w_2 {
                        nu[1] * Z_2[2] - nu[2] * Z_2[1],
                        nu[2] * Z_2[0] - nu[0] * Z_2[2],
                        nu[0] * Z_2[1] - nu[1] * Z_2[0]
                    };

                    // These are the heights of the triangle x_0, x_1, x_2 w.r.t. footpoint.
                    const Real h_0 = w_0.Norm();
                    const Real h_1 = w_1.Norm();
                    const Real h_2 = w_2.Norm();
                    
                    const Real h_0_inv = Scalar::Inv<Real>(h_0);
                    const Real h_1_inv = Scalar::Inv<Real>(h_1);
                    const Real h_2_inv = Scalar::Inv<Real>(h_2);

                    const Real a_0 = Dot(w_0,x_1) * h_0_inv;
                    const Real a_1 = Dot(w_1,x_2) * h_1_inv;
                    const Real a_2 = Dot(w_2,x_0) * h_2_inv;
                    
                    const Real b_0 = Dot(w_0,x_2) * h_0_inv;
                    const Real b_1 = Dot(w_1,x_0) * h_1_inv;
                    const Real b_2 = Dot(w_2,x_1) * h_2_inv;
                    
                    
                    // Inverse distances from the x_i to y.
                    const Real R_0_inv = Scalar::Inv<Real>( std::sqrt(Dot(x_0,x_0)) );
                    const Real R_1_inv = Scalar::Inv<Real>( std::sqrt(Dot(x_1,x_1)) );
                    const Real R_2_inv = Scalar::Inv<Real>( std::sqrt(Dot(x_2,x_2)) );
                    
                    const Real a_2_over_R_0 = a_2 * R_0_inv;
                    const Real b_2_over_R_1 = b_2 * R_1_inv;
                    
                    const Real a_0_over_R_1 = a_0 * R_1_inv;
                    const Real b_0_over_R_2 = b_0 * R_2_inv;
                    
                    const Real a_1_over_R_2 = a_1 * R_2_inv;
                    const Real b_1_over_R_0 = b_1 * R_0_inv;
                    
                    const Real c_over_h_0 = h * h_0_inv;
                    const Real c_over_h_1 = h * h_1_inv;
                    const Real c_over_h_2 = h * h_2_inv;
                    
                    
                    const Real factor_2 = Scalar::Half<Real> * h * kappa[0] * kappa[0];

                    const Real factor_1 = h * factor_2 - Scalar::One<Real>;
                    
                    const Real anglesum = MyMath::Equal3( b_0 >= a_0, b_1 >= a_1, b_2 >= a_2 )
                            ? ( (h>=Scalar::Zero<Real>) ? - Scalar::TwoPi<Real> : Scalar::TwoPi<Real> )
                            : Scalar::Zero<Real>;
                    
                    const Real double_singular_part = factor_1 * (
                        anglesum
                        + std::atan( c_over_h_2 * b_2_over_R_1 )
                        - std::atan( c_over_h_2 * a_2_over_R_0 )
                        + std::atan( c_over_h_0 * b_0_over_R_2 )
                        - std::atan( c_over_h_0 * a_0_over_R_1 )
                        + std::atan( c_over_h_1 * b_1_over_R_0 )
                        - std::atan( c_over_h_1 * a_1_over_R_2 )
                    )
                    +
                    factor_2 * (
                        h_2 * ( std::atanh( b_2_over_R_1 ) - std::atanh( a_2_over_R_0 ) )
                        +
                        h_0 * ( std::atanh( b_0_over_R_2 ) - std::atanh( a_0_over_R_1 ) )
                        +
                        h_1 * ( std::atanh( b_1_over_R_0 ) - std::atanh( a_1_over_R_2 ) )
                    )
                    ;
                    
                    const Real delta = static_cast<Real>(i==j);
                    
                    const Real r_2 = Dot(m,m);
                    const Real r   = std::sqrt(r_2);
                    const Real r_3 = r_2 * r;
                    
                    
                    const Complex I_kappa_r { 0, r * kappa[0] };
                    
                    const Complex double_regular_part = -(
                        Scalar::Half<Real> * r_2 * kappa[0]
                        +
                        std::exp( I_kappa_r ) * (I_kappa_r - Scalar::One<Real>)
                     ) * (h / (r_3 + delta) * (Scalar::One<Real>-delta));
                    
                    const Complex A_ij
                    = c_[0][2] * ( double_singular_part/area + double_regular_part );

                    ptr<Complex> B_j = &B_ptr[ldB*j];
                    
                    for( Int k = 0; k < wave_count; ++k )
                    {
                        C_i[k] += A_ij * B_j[k];
                    }
                    
                } // for( Int j = 0; j < simplex_count; ++j )
                
                copy_buffer( C_i, &C_ptr[ldC*i], wave_count );
                
            } // for( Int i = i_begin; i < i_end; ++i )
            
        } // for( Int thread = 0; thread < OMP_thread_count; ++thread )
        
        ptoc(ClassName()+"::DoubleLayer");

    }
            
