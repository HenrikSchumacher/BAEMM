
// FIXME: For non-jit compilation (e.g. for debugging)
// FIXME: we have run the following command in the terminal:
// FIXME: xcrun -sdk macosx metal -c GEMM2.metal -o GEMM2.air && xcrun -sdk macosx metallib GEMM2.air -o GEMM2.metallib

// FIXME: Comment-out the following line for run-time compilation:
R"(
// FIXME: Comment-in the following lines for run-time compilation:
//constant constexpr uint BLOCK_SIZE = 16;

#include <metal_stdlib>


#define OFFSET(row, col, ld) ((row)*(ld)+(col))

#define FETCH_FLOAT4(pointer) (reinterpret_cast<thread float4*>(&(pointer))[0])
#define FETCH_FLOAT4_DEV(pointer) (reinterpret_cast<device float4*>(&(pointer))[0])
#define FETCH_FLOAT4_SHARED(pointer) (reinterpret_cast<threadgroup float4*>(&(pointer))[0])
#define FETCH_FLOAT4_CONST(pointer) (reinterpret_cast<constant float4*>(&(pointer))[0])

using namespace metal;

//constant constexpr float zero    = static_cast<float>(0);
//constant constexpr float one     = static_cast<float>(1);
//constant constexpr float two     = static_cast<float>(2);
//constant     constexpr float one_half = one / two;

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
struct Matrix
{
    uint width;
    uint height;
    uint stride;
    device float * elements;
    
//    Matrix() = default;
//    ~Matrix() = default;
//
//    Matrix( uint width_, uint height_, uint stride_, device float * elements_ )
//    :   width    (width_   )
//    ,   height   (height_  )
//    ,   stride   (stride_  )
//    ,   elements (elements_)
//    {}
//
//    Matrix( uint width_, uint height_, uint stride_, device float * elements_ )
//    :   width    (width_   )
//    ,   height   (height_  )
//    ,   stride   (stride_  )
//    ,   elements (elements_)
//    {}
};

// Get a matrix element
float GetElement(const Matrix A, uint row, uint col)
{
    return A.elements[row * A.stride + col];
}
// Set a matrix element
void SetElement(Matrix A, uint row, uint col, float value)
{
    A.elements[row * A.stride + col] = value;
}
// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 Matrix GetSubMatrix(Matrix A, uint row, uint col)
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

[[kernel]] void GEMM_RM_NVidea(
    const constant uint  & M            [[buffer(0)]],
    const constant uint  & N            [[buffer(1)]],
    const constant uint  & K            [[buffer(2)]],
    const constant float & alpha        [[buffer(3)]],
          device   float * A_           [[buffer(4)]],
          device   float * B_           [[buffer(5)]],
    const constant float & beta         [[buffer(6)]],
          device   float * C_           [[buffer(7)]],

    const uint2 l_id                      [[thread_position_in_threadgroup]],
    const uint2   id                      [[thread_position_in_grid]],
    const uint2 g_id                      [[threadgroup_position_in_grid]],
    const uint2 g_size                    [[threadgroups_per_grid]],
    const uint2 threads_per_threadgroup   [[threads_per_threadgroup]]
)
{
    Matrix A;
    A.width    = M;
    A.height   = K;
    A.stride   = K;
    A.elements = A_;
    
    Matrix B;
    B.width    = K;
    B.height   = N;
    B.stride   = N;
    B.elements = B_;
    
    Matrix C;
    C.width    = M;
    C.height   = N;
    C.stride   = N;
    C.elements = C_;
    
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
    // Block row and column
        uint blockRow = g_id.y;
        uint blockCol = g_id.x;
        // Each thread block computes one sub-matrix Csub of C
        Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
        // Each thread computes one element of Csub
        // by accumulating results into Cvalue
        float Cvalue = 0;
        // Thread row and column within Csub
        uint row = l_id.y;
        uint col = l_id.x;
        // Loop over all the sub-matrices of A and B that are
        // required to compute Csub
        // Multiply each pair of sub-matrices together
        // and accumulate the results
        for( uint m = 0; m < (A.width / BLOCK_SIZE); ++m )
        {
            // Get sub-matrix Asub of A
            Matrix Asub = GetSubMatrix(A, blockRow, m);
            // Get sub-matrix Bsub of B
            Matrix Bsub = GetSubMatrix(B, m, blockCol);
            // Shared memory used to store Asub and Bsub respectively
            threadgroup float As[BLOCK_SIZE][BLOCK_SIZE];
            threadgroup float Bs[BLOCK_SIZE][BLOCK_SIZE];
            // Load Asub and Bsub from device memory to shared memory
            // Each thread loads one element of each sub-matrix
            As[row][col] = GetElement(Asub, row, col);
            Bs[row][col] = GetElement(Bsub, row, col);
            // Synchronize to make sure the sub-matrices are loaded
            // before starting the computation
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            // Multiply Asub and Bsub together
            for( uint e = 0; e < BLOCK_SIZE; ++e )
            {
                Cvalue += As[row][e] * Bs[e][col];
            }
            // Synchronize to make sure that the preceding
            // computation is done before loading two new
            // sub-matrices of A and B in the next iteration
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        // Write Csub to device memory
        // Each thread writes one element
        SetElement(Csub, row, col, Cvalue);
}

// FIXME: Comment-in the following lines for run-time compilation:
)"



