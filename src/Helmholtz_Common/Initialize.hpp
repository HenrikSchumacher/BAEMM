public:
    
    void Initialize()
    {
        ptic(ClassName()+"::Initialize");
        
        SetBlockSize(64);
        
        // For assembling AvOp.
        Tensor1<LInt,    Int> Av_outer ( simplex_count + 1, 0 );
        Tensor1<Int,    LInt> Av_inner ( 3 * simplex_count );
        Tensor1<Real,   LInt> Av_vals  ( 3 * simplex_count );
        Av_outer[0] = 0;
        
        // For assembling CurlOp.
        Tensor1<LInt,    Int> Curl_outer ( 3 * simplex_count + 1, 0 );
        Tensor1<Int,    LInt> Curl_inner ( 9 * simplex_count );
        Tensor1<Real,   LInt> Curl_vals  ( 9 * simplex_count );
        Curl_outer[0] = 0;

        // For assembling MassMatrix.
        Tensor3<Int,    LInt> i_list ( simplex_count, 3, 3 );
        Tensor3<Int,    LInt> j_list ( simplex_count, 3, 3 );
        Tensor3<Real,   LInt> a_list ( simplex_count, 3, 3 );
        
        // We pad 3-vector with an additional float so that we can use float3 in the metal kernels. (float3 has size 4 * 4 Byte to preserve alignement.)

        //CheckThis
        ParallelDo(
            [&,this]( const Int i )
            {
               Tiny::Vector<3,Real,Int> x_0;
               Tiny::Vector<3,Real,Int> x_1;
               Tiny::Vector<3,Real,Int> x_2;
               
               Tiny::Vector<3,Real,Int> df_0;
               Tiny::Vector<3,Real,Int> df_1;
               Tiny::Vector<3,Real,Int> a_nu;

               Tiny::Vector<3,Real,Int> u_0;
               Tiny::Vector<3,Real,Int> u_1;
               Tiny::Vector<3,Real,Int> u_2;
               Tiny::Vector<3,Real,Int> c;
               
               Tiny::Matrix<2,3,Real,Int> df_dagger;
               Tiny::Matrix<2,2,Real,Int> g;
               Tiny::Matrix<2,2,Real,Int> g_inv;
               Tiny::Matrix<3,3,Real,Int> grad_f;
               
               Int i_0 = triangles(i,0);
               Int i_1 = triangles(i,1);
               Int i_2 = triangles(i,2);

               tri_coords_ptr[12*i+ 0] = x_0[0] = vertex_coords(i_0,0);
               tri_coords_ptr[12*i+ 1] = x_0[1] = vertex_coords(i_0,1);
               tri_coords_ptr[12*i+ 2] = x_0[2] = vertex_coords(i_0,2);
               tri_coords_ptr[12*i+ 3] = Scalar::Zero<Real>;
               tri_coords_ptr[12*i+ 4] = x_1[0] = vertex_coords(i_1,0);
               tri_coords_ptr[12*i+ 5] = x_1[1] = vertex_coords(i_1,1);
               tri_coords_ptr[12*i+ 6] = x_1[2] = vertex_coords(i_1,2);
               tri_coords_ptr[12*i+ 7] = Scalar::Zero<Real>;
               tri_coords_ptr[12*i+ 8] = x_2[0] = vertex_coords(i_2,0);
               tri_coords_ptr[12*i+ 9] = x_2[1] = vertex_coords(i_2,1);
               tri_coords_ptr[12*i+10] = x_2[2] = vertex_coords(i_2,2);
               tri_coords_ptr[12*i+11] = Scalar::Zero<Real>;
               
               c[0] = third * (x_0[0] + x_1[0] + x_2[0]);
               c[1] = third * (x_0[1] + x_1[1] + x_2[1]);
               c[2] = third * (x_0[2] + x_1[2] + x_2[2]);
               
               // Compute the vectors pointing from the triangle center to the corners.
               // Will be used for diagonal of single layer operator.
               u_0 = x_0; u_0 -= c;
               u_1 = x_1; u_1 -= c;
               u_2 = x_2; u_2 -= c;
               
               mid_points_ptr[4*i+0] = c[0];
               mid_points_ptr[4*i+1] = c[1];
               mid_points_ptr[4*i+2] = c[2];
               mid_points_ptr[4*i+3] = zero;

               df_0 = x_1; df_0 -= x_0;
               df_1 = x_2; df_1 -= x_0;
               
               
               // The triangle normal and area.
               
               // Area weighted normals
               a_nu[0] =  Scalar::Half<Real> * ( df_0[1] * df_1[2] - df_0[2] * df_1[1] );
               a_nu[1] =  Scalar::Half<Real> * ( df_0[2] * df_1[0] - df_0[0] * df_1[2] );
               a_nu[2] =  Scalar::Half<Real> * ( df_0[0] * df_1[1] - df_0[1] * df_1[0] );
               
               const Real a = std::sqrt( a_nu[0]*a_nu[0] + a_nu[1]*a_nu[1] + a_nu[2]*a_nu[2] );
               
               areas_ptr[i] = a;

               const Real a_inv = Scalar::One<Real> / a;
               
               normals_ptr[4*i+0] = a_inv * a_nu[0];
               normals_ptr[4*i+1] = a_inv * a_nu[1];
               normals_ptr[4*i+2] = a_inv * a_nu[2];
               normals_ptr[4*i+3] = zero;

               
               // Assemble averaging operator
               {
                   const Real a_over_3  = a * third;
                   
                   Av_outer[i+1]   = 3 * (i+1);
                   Av_inner[3*i+0] = i_0;
                   Av_inner[3*i+1] = i_1;
                   Av_inner[3*i+2] = i_2;
                   Av_vals [3*i+0] = a_over_3;
                   Av_vals [3*i+1] = a_over_3;
                   Av_vals [3*i+2] = a_over_3;
               }
               
               // Assemble curl operator
               {
                   Curl_outer[3*(i+1)+0] = (3 * (i+1) + 0) * 3;
                   Curl_outer[3*(i+1)+1] = (3 * (i+1) + 1) * 3;
                   Curl_outer[3*(i+1)+2] = (3 * (i+1) + 2) * 3;

                   Curl_inner[(i * 3 + 0) * 3 + 0 ] = i_0;
                   Curl_inner[(i * 3 + 0) * 3 + 1 ] = i_1;
                   Curl_inner[(i * 3 + 0) * 3 + 2 ] = i_2;
                   Curl_inner[(i * 3 + 1) * 3 + 0 ] = i_0;
                   Curl_inner[(i * 3 + 1) * 3 + 1 ] = i_1;
                   Curl_inner[(i * 3 + 1) * 3 + 2 ] = i_2;
                   Curl_inner[(i * 3 + 2) * 3 + 0 ] = i_0;
                   Curl_inner[(i * 3 + 2) * 3 + 1 ] = i_1;
                   Curl_inner[(i * 3 + 2) * 3 + 2 ] = i_2;
                   
                   g[0][0] = Dot(df_0,df_0);
                   g[0][1] = Dot(df_0,df_1);
                   g[1][0] = Dot(df_1,df_0);
                   g[1][1] = Dot(df_1,df_1);
                   
                   Real det = g[0][0] * g[1][1] - g[0][1] * g[0][1];
                   
                   Real det_inv = Scalar::One<Real> / det;
                   
                   g_inv[0][0] =  g[1][1] * det_inv;
                   g_inv[0][1] = -g[0][1] * det_inv;
                   g_inv[1][1] =  g[0][0] * det_inv;
                   
                   //  df_dagger = g^{-1} * df^T (2 x 3 matrix)
                   df_dagger[0][0] = g_inv[0][0] * df_0[0] + g_inv[0][1] * df_1[0];
                   df_dagger[0][1] = g_inv[0][0] * df_0[1] + g_inv[0][1] * df_1[1];
                   df_dagger[0][2] = g_inv[0][0] * df_0[2] + g_inv[0][1] * df_1[2];
                   df_dagger[1][0] = g_inv[0][1] * df_0[0] + g_inv[1][1] * df_1[0];
                   df_dagger[1][1] = g_inv[0][1] * df_0[1] + g_inv[1][1] * df_1[1];
                   df_dagger[1][2] = g_inv[0][1] * df_0[2] + g_inv[1][1] * df_1[2];
                   
                   // The local map from the three vertex values to the gradient per triangle.
                   // I.e., grad_f is _not_ the gradient of f,
                   // but rather the gradient _operator_ belonging to f.
                   
                   grad_f[0][0] = - df_dagger[0][0] - df_dagger[1][0];
                   grad_f[0][1] =   df_dagger[0][0];
                   grad_f[0][2] =   df_dagger[1][0];
                   grad_f[1][0] = - df_dagger[0][1] - df_dagger[1][1];
                   grad_f[1][1] =   df_dagger[0][1];
                   grad_f[1][2] =   df_dagger[1][1];
                   grad_f[2][0] = - df_dagger[0][2] - df_dagger[1][2];
                   grad_f[2][1] =   df_dagger[0][2];
                   grad_f[2][2] =   df_dagger[1][2];
                   
                   mptr<Real> Curl_f = Curl_vals.data(9 * i);
                   
                   // Curl_f = nu x grad_f times area of triangle
                   
                   Curl_f[0] = a_nu[1] * grad_f[2][0] - a_nu[2] * grad_f[1][0];
                   Curl_f[1] = a_nu[1] * grad_f[2][1] - a_nu[2] * grad_f[1][1];
                   Curl_f[2] = a_nu[1] * grad_f[2][2] - a_nu[2] * grad_f[1][2];
                   
                   Curl_f[3] = a_nu[2] * grad_f[0][0] - a_nu[0] * grad_f[2][0];
                   Curl_f[4] = a_nu[2] * grad_f[0][1] - a_nu[0] * grad_f[2][1];
                   Curl_f[5] = a_nu[2] * grad_f[0][2] - a_nu[0] * grad_f[2][2];
                   
                   Curl_f[6] = a_nu[0] * grad_f[1][0] - a_nu[1] * grad_f[0][0];
                   Curl_f[7] = a_nu[0] * grad_f[1][1] - a_nu[1] * grad_f[0][1];
                   Curl_f[8] = a_nu[0] * grad_f[1][2] - a_nu[1] * grad_f[0][2];
               }
               
               // Assemble mass matrix
               {
                   i_list(i,0,0) = i_0;
                   i_list(i,0,1) = i_0;
                   i_list(i,0,2) = i_0;
                   i_list(i,1,0) = i_1;
                   i_list(i,1,1) = i_1;
                   i_list(i,1,2) = i_1;
                   i_list(i,2,0) = i_2;
                   i_list(i,2,1) = i_2;
                   i_list(i,2,2) = i_2;
                   
                   j_list(i,0,0) = i_0;
                   j_list(i,0,1) = i_1;
                   j_list(i,0,2) = i_2;
                   j_list(i,1,0) = i_0;
                   j_list(i,1,1) = i_1;
                   j_list(i,1,2) = i_2;
                   j_list(i,2,0) = i_0;
                   j_list(i,2,1) = i_1;
                   j_list(i,2,2) = i_2;
               
                   const Real a_over_6  = a * sixth;
                   const Real a_over_12 = a * twelveth;
                   
                   a_list(i,0,0) = a_over_6;
                   a_list(i,0,1) = a_over_12;
                   a_list(i,0,2) = a_over_12;
                   a_list(i,1,0) = a_over_12;
                   a_list(i,1,1) = a_over_6;
                   a_list(i,1,2) = a_over_12;
                   a_list(i,2,0) = a_over_12;
                   a_list(i,2,1) = a_over_12;
                   a_list(i,2,2) = a_over_6;
               }
               
               
               // Compute diagonal of single layer boundary operator.
               
               const Real u0u0 = Dot(u_0,u_0);
               const Real u1u1 = Dot(u_1,u_1);
               const Real u2u2 = Dot(u_2,u_2);
               
               const Real u0u1 = Dot(u_0,u_1);
               const Real u1u2 = Dot(u_1,u_2);
               const Real u2u0 = Dot(u_2,u_0);

               const Real L0 = std::sqrt(u0u0);
               const Real L1 = std::sqrt(u1u1);
               const Real L2 = std::sqrt(u2u2);
               
               const Real L0_inv = Scalar::One<Real>/L0;
               const Real L1_inv = Scalar::One<Real>/L1;
               const Real L2_inv = Scalar::One<Real>/L2;
               
               Real sum = Scalar::Zero<Real>;
               
               {
                   const Real a = u1u1 - u1u2;
                   const Real b = u2u2 - u1u2;
                   
                   const Real vnorminv = Scalar::One<Real> / std::sqrt( std::abs(a + b) );
                   const Real SinAlpha = L1_inv * vnorminv * a;
                   const Real SinBeta  = L2_inv * vnorminv * b;
                
                   sum += L1 * std::sqrt( Scalar::One<Real> - SinAlpha * SinAlpha ) * ( std::atanh(SinAlpha) + std::atanh(SinBeta) );
               }
               
               {
                   const Real a = u2u2 - u2u0;
                   const Real b = u0u0 - u2u0;
                   
                   const Real vnorminv = Scalar::One<Real> / std::sqrt( std::abs(a + b) );
                   const Real SinAlpha = L2_inv * vnorminv * a;
                   const Real SinBeta  = L0_inv * vnorminv * b;
                   
                   sum += L2 * std::sqrt( Scalar::One<Real> - SinAlpha * SinAlpha ) * ( std::atanh(SinAlpha) + std::atanh(SinBeta) );
               }
               
               {
                   const Real a = u0u0 - u0u1;
                   const Real b = u1u1 - u0u1;
                   
                   const Real vnorminv = Scalar::One<Real> / std::sqrt( std::abs(a + b) );
                   const Real SinAlpha = L0_inv * vnorminv * a;
                   const Real SinBeta  = L1_inv * vnorminv * b;
                   
                   sum += L0 * std::sqrt( Scalar::One<Real> - SinAlpha * SinAlpha ) * ( std::atanh(SinAlpha) + std::atanh(SinBeta) );
               }
               
               single_diag_ptr[i] = sum / areas_ptr[i];
            },
            simplex_count, CPU_thread_count
        );
        
        AvOp = Sparse_T(
            std::move(Av_outer), std::move(Av_inner), std::move(Av_vals),
            simplex_count, vertex_count,
            CPU_thread_count
        );

        AvOp.SortInner();

        AvOpTransp = AvOp.Transpose();
        
        CurlOp = Sparse_T(
            std::move(Curl_outer), std::move(Curl_inner), std::move(Curl_vals),
            3 * simplex_count, vertex_count,
            CPU_thread_count
        );
        
        CurlOp.SortInner();
        
        CurlOpTransp = CurlOp.Transpose();
        
        Mass = Sparse_T(
            a_list.Size(),
            i_list.data(), j_list.data(), a_list.data(),
            vertex_count, vertex_count,
            CPU_thread_count, true, false
        );
        
        ptoc(ClassName()+"::Initialize");
    }
