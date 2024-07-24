R"(
__constant float2 zero = (float2)(0.0f,0.0f);
__constant float  one  = 1.0f;

__kernel void HerglotzWaveKernel(
        const __global   float4 * mid_points     ,
        const __global   float4 * normals        ,
        const __global   float4 * meas_directions,
        const __global   float2 * B_global       ,
              __global   float2 * C_global       ,
              __constant float  * kappa_buf      ,
              __constant float2 * coeff          ,
              __constant int    * N              ,
              __constant int    * M              ,
              __constant int    * wave_count
)
{
    const int n             = *N;
    const int m             = *M;
    const int k_chunk_count = (*wave_count) / k_chunk_size;
    const int k_ld          = (*wave_count);
    
    const int block_count = (m + block_size - 1)/block_size;

    const int i = get_global_id(0);

    __private float3 x_i;
    __private float3 nu_i;
    
    __private float2 A_i [block_size];

    __local float2 B [block_size][k_chunk_size];

    __local float3 y [block_size];
    
    if( i < n )
    {
        x_i  = mid_points[i].xyz;

        if( Re_double_layer || Im_double_layer )
        {
            nu_i = normals[i].xyz;
        }
    }
    else
    {
        x_i.x = 1.f;
        x_i.y = 1.f;
        x_i.z = 1.f;

        if( Re_double_layer || Im_double_layer )
        {
            nu_i.x = 0.f;
            nu_i.y = 0.f;
            nu_i.z = 1.f;
        }
    }
    
    for( int k_chunk = 0; k_chunk < k_chunk_count; ++k_chunk )
    {   
        const __private float  kappa = kappa_buf[k_chunk];
        const __private float2 c [2] = {coeff[4*k_chunk + 1], coeff[4*k_chunk + 2]};

        __private float2 C_i [k_chunk_size] = {zero};
        
        for( int j_block = 0; j_block < block_count; ++j_block )
        {
            {
                const int j_loc  = get_local_id(0);
                const int j      = block_size * j_block + j_loc;
                
                if( j < m )
                {
                    y[j_loc] = meas_directions[j].xyz;
                }
                else
                {                    
                    y[j_loc].x = 0.f;
                    y[j_loc].y = 0.f;
                    y[j_loc].z = 0.f;
                }
                
                barrier(CLK_LOCAL_MEM_FENCE);
                
                for( int j_loc = 0; j_loc < block_size; ++j_loc )
                {
                    const int j = block_size * j_block + j_loc;

                    const float delta = (float)((i < n) || (j < m) );

                    const float KappaR = kappa * (y[j_loc].x * x_i.x + y[j_loc].y * x_i.y + y[j_loc].z * x_i.z);
                    
                    float CosKappaR;
                    float SinKappaR = sincos( KappaR, &CosKappaR );
                    
                    A_i[j_loc] = zero;

                    if( Re_single_layer && Im_single_layer )
                    {
                        A_i[j_loc].x += delta*(c[0].x * CosKappaR + c[0].y * SinKappaR);
                        A_i[j_loc].y += delta*(-c[0].x * SinKappaR + c[0].y * CosKappaR);
                    }
                    else if (Re_single_layer )
                    {
                        A_i[j_loc].x += delta*( c[0].x * CosKappaR);
                        A_i[j_loc].y -= delta*( c[0].x * SinKappaR);
                    }
                    else if (Im_single_layer )
                    {
                        A_i[j_loc].x += delta*( c[0].y * SinKappaR);
                        A_i[j_loc].y += delta*( c[0].y * CosKappaR);
                    }

                    if( Re_double_layer || Im_double_layer)
                    {
                        const float dKappaR = delta * kappa * (nu_i.x * y[j_loc].x + nu_i.y * y[j_loc].y + nu_i.z * y[j_loc].z);
                        const float dKappaRCosKappaR = dKappaR * CosKappaR;
                        const float dKappaRSinKappaR = dKappaR * SinKappaR;

                        if( Re_double_layer )
                        {
                            A_i[j_loc].x -= dKappaRSinKappaR * c[1].x;
                            A_i[j_loc].y -= dKappaRCosKappaR * c[1].x;
                        }
                        if( Im_double_layer )
                        {
                            A_i[j_loc].x += dKappaRCosKappaR * c[1].y;
                            A_i[j_loc].y -= dKappaRSinKappaR * c[1].y;
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
                    B[j_loc][k].x = B_j_blk[k].x;
                    B[j_loc][k].y = -B_j_blk[k].y;
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
