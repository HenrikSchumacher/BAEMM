R"(
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c Helmholtz.metal -o Helmholtz.air && xcrun -sdk macosx metallib Helmholtz.air -o Helmholtz.metallib

// FIXME: Comment-out the following line for run-time compilation:

// FIXME: We use "block_size" and "wave_chunk_size" as "template parameters" for jit-compilation.
// FIXME: Comment-in the following two lines for run-time compilation:
//constant constexpr int  block_size      = 64;
//constant constexpr int  wave_chunk_size = 16;
//constant constexpr bool Re_single_layer    = true;
//constant constexpr bool Im_single_layer    = true;
//constant constexpr bool Re_double_layer    = true;
//constant constexpr bool Im_double_layer    = true;
//constant constexpr bool Re_adjdbl_layer    = true;
//constant constexpr bool Im_adjdbl_layer    = true;

#include <metal_stdlib>

using namespace metal;

using cmplx = float2;

using cmplx4 = float4x2;

using Row_T = array<float, 2 * wave_chunk_size>;

constant constexpr float zero    = static_cast<float>(0);
constant constexpr float one     = static_cast<float>(1);
//constant constexpr float two     = static_cast<float>(2);
//constant     constexpr float one_half = one / two;

//constant constexpr float pi      = 3.141592653589793;
//constant constexpr float two_pi  = two * pi;
//constant constexpr float four_pi = two * two_pi;

//constant     constexpr float one_over_two_pi  = one / two_pi;
//constant constexpr float one_over_four_pi = one / four_pi;

[[max_total_threads_per_threadgroup(block_size)]]
[[kernel]] void BoundaryOperatorKernel_C(
    const constant float3 * const mid_points    [[buffer(0)]], // triangle midpoints
    const constant float3 * const normals       [[buffer(1)]], // triangle normals
    const constant cmplx  * const B_global      [[buffer(2)]], // buffer for right hand sides
          device   cmplx  * const C_global      [[buffer(3)]], // buffer for results C = A * B
    const constant float  * const kappa_buf     [[buffer(4)]], // vector of wave numbers
    const constant cmplx4 *       c_buf         [[buffer(5)]], // coefficients for the ops
    const constant int    &       n             [[buffer(6)]], // number of triangles
    const constant int    &       wave_count    [[buffer(7)]], // number of right hand sides
                                   
    const uint thread_position_in_threadgroup   [[thread_position_in_threadgroup]],
    const uint thread_position_in_grid          [[thread_position_in_grid]],
    const uint threads_per_threadgroup          [[threads_per_threadgroup]],
    const uint threadgroup_position_in_grid     [[threadgroup_position_in_grid]]
)
{
    
    // Important assumptions:
    // - wave_count and wave_chunk_size are divisible by vec_size
    // - n is divisible by block_size
    // - threads_per_threadgroup == block_size
    // - kappa has at least k_chunk_count = wave_count / k_chunk_size entries
    // - To work correctly, kappa has to be constant in each chunk of waves.
    
    constexpr int k_chunk_size  = wave_chunk_size;
    const     int k_chunk_count = wave_count / k_chunk_size;
    const     int k_ld          = wave_count;  // Leading dim of B and C.
    
    // number of blocks
    const int block_count = (n + block_size - 1) / block_size;
    
//    const int i_loc   = thread_position_in_threadgroup;
    const int i = thread_position_in_grid;
//    const int i_block = threadgroup_position_in_grid;

    // each thread in the threadgroup gets one target point x and normal nu assigned.
    thread float3 x_i;
    thread float3 nu_i;
    
    thread cmplx A_i [block_size]; // stores a row chunk of A_i for j in threadgroup

    
    // The rows of the input matrix belonging to the threadgroup.
    threadgroup cmplx B [block_size][k_chunk_size];
    
    // Each thread loads the x-data for itself only once.
    if( i < n )
    {
        x_i  = mid_points[i];
        
        if( Re_adjdbl_layer || Im_adjdbl_layer )
        {
            nu_i = normals[i];
        }
    }
    else
    {
        // Load just anything.
        x_i[0] = 1.f;
        x_i[1] = 1.f;
        x_i[2] = 1.f;
        
        if( Re_adjdbl_layer || Im_adjdbl_layer )
        {
            nu_i[0] = 0.f;
            nu_i[1] = 0.f;
            nu_i[2] = 1.f;
        }
    }

    
    
    // Typically, k_chunk_count is much smaller than block_count.
    // Since loading and writing to device memory is very expensive, it might be a better to make
    // the loop over the k_chunks the outer loop and pay the (small) price of recomputing A multiple times.
    for( int k_chunk = 0; k_chunk < k_chunk_count; ++k_chunk )
    {   
        const thread float  kappa = kappa_buf[k_chunk];
        const thread cmplx4 c     = c_buf[k_chunk];
        
        // Each thread maintains one row of the output matrix.
        thread cmplx C_i [k_chunk_size] = {zero};
        
        for( int j_block = 0; j_block < block_count; ++j_block )
        {
            // Compute the block of A corresponding to
            //  [ block_size * i_block,...,block_size * (i_block+1)[
            //  x
            //  [ block_size * j_block,...,block_size * (j_block+1)[
            {
                // Each thread lets its x interact with a bunch of ys loaded into threadgroup.
                threadgroup float3 y  [block_size];
                threadgroup float3 mu [block_size];
                
                const int j_loc  = thread_position_in_threadgroup;
                const int j      = block_size * j_block + j_loc;
                
                // Each thread in the threadgroup loads 1 entry of y and mu.
                if( j < n )
                {
                    y[j_loc] = mid_points[j];
                    
                    if( Re_double_layer || Im_double_layer )
                    {
                        mu[j_loc] = normals[j];
                    }
                }
                else
                {
                    // Load just anything not equal to x_i.
                    
                    y [j_loc][0] = 0.f;
                    y [j_loc][1] = 0.f;
                    y [j_loc][2] = 0.f;
                    
                    if( Re_adjdbl_layer || Im_adjdbl_layer )
                    {
                        mu[j_loc][0] = 0.f;
                        mu[j_loc][1] = 0.f;
                        mu[j_loc][2] = 1.f;
                    }
                }
                
                // need synchronization after loading data
                threadgroup_barrier(mem_flags::mem_threadgroup);
                
                for( int j_loc = 0; j_loc < block_size; ++j_loc )
                {
                    const int j = block_size * j_block + j_loc;
                    
                    // We ignore the results if i and j coincide or if one of i or j are invalid.
                    const float delta = static_cast<float>( (i == j) || (i >= n) || (j >= n) );
                    
                    const float3 z = y[j_loc] - x_i;
                    
                    const float r = sqrt( z[0] * z[0] + z[1] * z[1] + z[2] * z[2] );
                    
                    // Since each operator has at least one factor of r_inv, this will set to 0 all entries in the block A that lie on the main diagonal or outside the actual n x n matrix.
                    const float r_inv = (one - delta) / (r + delta);
                                        
                    const float KappaR = kappa * r;

                    float CosKappaR;
                    float SinKappaR = sincos( KappaR, CosKappaR );
                    
                    A_i[j_loc][0] = 0.f;
                    A_i[j_loc][1] = 0.f;

                    if( Re_single_layer && Im_single_layer )
                    {
                        A_i[j_loc][0] += (c[1][0] * CosKappaR - c[1][1] * SinKappaR) * r_inv;
                        A_i[j_loc][1] += (c[1][0] * SinKappaR + c[1][1] * CosKappaR) * r_inv ;
                    }
                    else if (Re_single_layer )
                    {
                        A_i[j_loc][0] += (c[1][0] * CosKappaR) * r_inv;
                        A_i[j_loc][1] += (c[1][0] * SinKappaR) * r_inv;
                    }
                    else if (Im_single_layer )
                    {
                        A_i[j_loc][0] += (- c[1][1] * SinKappaR) * r_inv;
                        A_i[j_loc][1] += (+ c[1][1] * CosKappaR) * r_inv;
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
                            // We have to add
                            // c[2] * exp(I * kappa * r) * (I * kappa * r - 1) ( z.mu_j ) / r^3
                            // to A_i.

                            const float factor = (
                                  z[0] * mu[j_loc][0]
                                + z[1] * mu[j_loc][1]
                                + z[2] * mu[j_loc][2]
                            );

                            const float b_0 = factor * a_0;
                            const float b_1 = factor * a_1;

                            if( Re_double_layer )
                            {
                                A_i[j_loc][0] += b_0 * c[2][0];
                                A_i[j_loc][1] += b_1 * c[2][0];
                            }
                            if( Im_double_layer )
                            {
                                A_i[j_loc][0] -= b_1 * c[2][1];
                                A_i[j_loc][1] += b_0 * c[2][1];
                            }
                        }

                        if( Re_adjdbl_layer || Im_adjdbl_layer )
                        {
                            // We have to add
                            // c[3] * exp(I * kappa * r) * (I * kappa * r - 1) ( - z.nu_i ) / r^3
                            // to A_i.

                            const float factor = - (
                                  z[0] * nu_i[0]
                                + z[1] * nu_i[1]
                                + z[2] * nu_i[2]
                            );

                            const float b_0 = factor * a_0;
                            const float b_1 = factor * a_1;

                            if( Re_adjdbl_layer )
                            {
                                A_i[j_loc][0] += b_0 * c[3][0];
                                A_i[j_loc][1] += b_1 * c[3][0];
                            }
                            if( Im_adjdbl_layer )
                            {
                                A_i[j_loc][0] -= b_1 * c[3][1];
                                A_i[j_loc][1] += b_0 * c[3][1];
                            }
                        }
                    }
                    
                } // for( int j_loc = 0; j_loc < block_size; ++j_loc )
            }
            
            // Load the block of B corresponding to
            //  [ block_size * j_block,...,block_size * (j_block+1)[
            //  x
            //  [ k_chunk_size * k_chunk,...,k_chunk_size * (k_chunk+1)[
                
            // TODO: Improve reading of data.
            // Each thread in threadgroup loads 1 row of B.
            {
                const int j_loc  = thread_position_in_threadgroup;
                const int j      = block_size * j_block + j_loc;

                const constant cmplx * B_j_blk = &B_global[k_ld * j + k_chunk_size * k_chunk];

                for( int k = 0; k < k_chunk_size; ++k )
                {
                    B[j_loc][k] = B_j_blk[k];
                }
            }
            
            
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Do the actual matrix-matrix multiplication.
            for( int j_loc = 0; j_loc < block_size; ++j_loc )
            {
                for( int k = 0; k < k_chunk_size; ++k )
                {
                    C_i[k][0] +=   A_i[j_loc][0] * B[j_loc][k][0]
                                 - A_i[j_loc][1] * B[j_loc][k][1];

                    C_i[k][1] +=   A_i[j_loc][0] * B[j_loc][k][1]
                                 + A_i[j_loc][1] * B[j_loc][k][0];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            
        } // for( int j_block = 0; j_block < block_count; ++j_block )
        
        // Write the block of C corresponding to
        //  [ block_size * i_block,...,block_size * (i_block+1)[
        //  x
        //  [ k_chunk_size * k_chunk,...,k_chunk_size * (k_chunk+1)[
        //
        // TODO: Improve writing of data.
        // Each thread takes care of its own row.
        {
            device cmplx * C_i_blk = &C_global[k_ld * i + k_chunk_size * k_chunk];

            for( int k = 0; k < k_chunk_size; ++k )
            {
                C_i_blk[k] = C_i[k];
            }
        }
    }
    
} // BoundaryOperatorKernel_C

// FIXME: Comment-out the following line for run-time compilation:
)"
