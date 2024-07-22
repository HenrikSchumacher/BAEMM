R"(
__constant float2 zero = (float2)(0.0f,0.0f);
__constant float  one  = 1.0f;

__kernel void BoundaryOperatorKernel_C(
        const __global   float4 * mid_points     ,
        const __global   float4 * normals        ,
        const __global   float2 * B_global       ,
              __global   float2 * C_global       ,
              __constant float  * kappa_buf      ,
              __constant float2 * coeff          ,
              __constant int    * N              ,
              __constant int    * wave_count
)  
{
    const int n             = *N;
    const int k_chunk_count = (*wave_count) / k_chunk_size;
    const int k_ld          = (*wave_count);
    
    const int block_count = (n + block_size - 1)/block_size;

    const int i = get_global_id(0);

    __private float3 x_i;
    __private float3 nu_i;
    
    __private float2 A_i [block_size];

    __local float2 B [block_size][k_chunk_size];

    __local float3 y  [block_size];
    __local float3 mu [block_size];
    
    if( i < n )
    {
        x_i  = mid_points[i].xyz;
        
        if( Re_adjdbl_layer || Im_adjdbl_layer )
        {
            nu_i = normals[i].xyz;
        }
    }
    else
    {
        x_i.x = 1.f;
        x_i.y = 1.f;
        x_i.z = 1.f;
        
        if( Re_adjdbl_layer || Im_adjdbl_layer )
        {
            nu_i.x = 0.f;
            nu_i.y = 0.f;
            nu_i.z = 1.f;
        }
    }
    
    for( int k_chunk = 0; k_chunk < k_chunk_count; ++k_chunk )
    {   
        const __private float  kappa = kappa_buf[k_chunk];
        const __private float2 c [4] = {
            coeff[4*k_chunk + 0],
            coeff[4*k_chunk + 1],
            coeff[4*k_chunk + 2],
            coeff[4*k_chunk + 3]
        };

        __private float2 C_i [k_chunk_size] = {zero};
        
        for( int j_block = 0; j_block < block_count; ++j_block )
        {
            {
                const int j_loc  = get_local_id(0);
                const int j      = block_size * j_block + j_loc;
                
                if( j < n )
                {
                    y[j_loc] = mid_points[j].xyz;
                    
                    if( Re_double_layer || Im_double_layer )
                    {
                        mu[j_loc] = normals[j].xyz;
                    }
                }
                else
                {                    
                    y [j_loc].x = 0.f;
                    y [j_loc].y = 0.f;
                    y [j_loc].z = 0.f;
                    
                    if( Re_double_layer || Im_double_layer )
                    {
                        mu[j_loc].x = 0.f;
                        mu[j_loc].y = 0.f;
                        mu[j_loc].z = 1.f;
                    }
                }
                
                barrier(CLK_LOCAL_MEM_FENCE);
                
                for( int j_loc = 0; j_loc < block_size; ++j_loc )
                {
                    const int j = block_size * j_block + j_loc;

                    const float delta = (float)( (i == j) || (i >= n) || (j >= n) );

                    const float3 z = y[j_loc] - x_i;

                    const float r = sqrt( z.x * z.x + z.y * z.y + z.z * z.z );
                    
                    const float r_inv = (one - delta) / (r + delta);

                    const float KappaR = kappa * r;
                    
                    float CosKappaR;
                    float SinKappaR = sincos( KappaR, &CosKappaR );
                    
                    A_i[j_loc] = zero;

                    if( Re_single_layer && Im_single_layer )
                    {
                        A_i[j_loc].x += (c[1].x * CosKappaR - c[1].y * SinKappaR) * r_inv;
                        A_i[j_loc].y += (c[1].x * SinKappaR + c[1].y * CosKappaR) * r_inv;
                    }
                    else if (Re_single_layer )
                    {
                        A_i[j_loc].x += (c[1].x * CosKappaR) * r_inv;
                        A_i[j_loc].y += (c[1].x * SinKappaR) * r_inv;
                    }
                    else if (Im_single_layer )
                    {
                        A_i[j_loc].x += (- c[1].y * SinKappaR) * r_inv;
                        A_i[j_loc].y += (+ c[1].y * CosKappaR) * r_inv;
                    }


                    if( Re_double_layer || Im_double_layer
                        ||
                        Re_adjdbl_layer || Im_adjdbl_layer
                    )
                    {
                        const float r3_inv = r_inv * r_inv * r_inv;
                        const float KappaRCosKappaR = KappaR * CosKappaR;
                        const float KappaRSinKappaR = KappaR * SinKappaR;

                        const float a_0 = -(KappaRSinKappaR + CosKappaR) * r3_inv;
                        const float a_1 =  (KappaRCosKappaR - SinKappaR) * r3_inv;


                        if( Re_double_layer || Im_double_layer )
                        {
                            const float factor = (
                                  z.x * mu[j_loc].x
                                + z.y * mu[j_loc].y
                                + z.z * mu[j_loc].z
                            );

                            const float b_0 = factor * a_0;
                            const float b_1 = factor * a_1;

                            if( Re_double_layer )
                            {
                                A_i[j_loc].x += b_0 * c[2].x;
                                A_i[j_loc].y += b_1 * c[2].x;
                            }
                            if( Im_double_layer )
                            {
                                A_i[j_loc].x -= b_1 * c[2].y;
                                A_i[j_loc].y += b_0 * c[2].y;
                            }
                        }

                        if( Re_adjdbl_layer || Im_adjdbl_layer )
                        {
                            const float factor = - (
                                  z.x * nu_i.x
                                + z.y * nu_i.y
                                + z.z * nu_i.z
                            );

                            const float b_0 = factor * a_0;
                            const float b_1 = factor * a_1;

                            if( Re_adjdbl_layer )
                            {
                                A_i[j_loc].x += b_0 * c[3].x;
                                A_i[j_loc].y += b_1 * c[3].x;
                            }
                            if( Im_adjdbl_layer )
                            {
                                A_i[j_loc].x -= b_1 * c[3].y;
                                A_i[j_loc].y += b_0 * c[3].y;
                            }
                        }
                    }                    
                }
            }
            
            {
                const int j_loc  = get_local_id(0);
                const int j      = block_size * j_block + j_loc;

                __global const float2 * B_j_blk = &B_global[k_ld * j + k_chunk_size * k_chunk];

                #pragma unroll
                for( int k = 0; k < k_chunk_size; ++k )
                {
                    B[j_loc][k] = B_j_blk[k];
                }
            }
                        
            barrier(CLK_LOCAL_MEM_FENCE);

            for( int j_loc = 0; j_loc < block_size; ++j_loc )
            {
                #pragma unroll
                for( int k = 0; k < k_chunk_size; ++k )
                {
                    C_i[k].x +=    A_i[j_loc].x * B[j_loc][k].x
                                 - A_i[j_loc].y * B[j_loc][k].y;

                    C_i[k].y +=    A_i[j_loc].x * B[j_loc][k].y
                                 + A_i[j_loc].y * B[j_loc][k].x;
                }
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);            
        }

        {
            __global float2 * C_i_blk = &C_global[k_ld * i + k_chunk_size * k_chunk];

            #pragma unroll
            for( int k = 0; k < k_chunk_size; ++k )
            {
                C_i_blk[k] = C_i[k];
            }
        } 
    }
}
)"
