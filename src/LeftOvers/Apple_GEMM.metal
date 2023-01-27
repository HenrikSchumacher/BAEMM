#include <metal_stdlib>
using namespace metal;

struct MetalMatrixDim
{
    ushort m;
    ushort k;
    ushort n;
    ushort pbytes;
    ushort 32;
} ;


kernel void MatrixMultiply(
   device const float *       A    [[ buffer(0) ]],
   device const float *       B    [[ buffer(1) ]],
   device       float *       C    [[ buffer(2) ]],
   constant MetalMatrixDim &  dims [[ buffer(3) ]],
   ushort2                    gid  [[ thread_position_in_grid ]]
)
{
    const ushort m = dims.m;
    const ushort k = dims.k;
    const ushort n = dims.n;

    ushort pbytes = dims.pbytes;
    ushort qbytes = dims.qbytes; // I have a hunch that qbytes == 32.

//    ushort2 gidIn = ushort2( gid.x << 3, gid.y << 3 );
    ushort2 gidIn = ushort2( 8 * gid.x, 8 * gid.y );
    
    const uint i_base = 8 * gid.x;
    const uint j_base = 8 * gid.y;

    if( gidIn.x >= m || gidIn.y >= k )
    {
        return;
    }

    const device float4 * a_vec = (const device float4*)(A + i_base );
    const device float4 * b_vec = (const device float4*)(B + j_base );
    
    device float4* c = static_cast<device float4*>( &C[8 * i_base + j_base] );

//    const device float4* Bend = (const device float4*)((const device char*)B + 32*n);
    // It's about 8 rows of B.
    const device float4* Bend = static_cast<device float4*>(&B[8 * n]);
    
    
    float4 s [8][2] = {{0.0f}};

    
    do
    {
        float4 a[0] = a_vec[0]; float4 a[1] = a_vec[1];
        float4 b[0] = b_vec[0]; float4 b[1] = b_vec[1];
        
        s[0][0] += (a[0].x * b[0]); s[0][1] += (a[0].x * b[1]);
        s[1][0] += (a[0].y * b[0]); s[1][1] += (a[0].y * b[1]);
        s[2][0] += (a[0].z * b[0]); s[2][1] += (a[0].z * b[1]);
        s[3][0] += (a[0].w * b[0]); s[3][1] += (a[0].w * b[1]);

        s[4][0] += (a[1].x * b[0]); s[4][1] += (a[1].x * b[1]);
        s[5][0] += (a[1].y * b[0]); s[5][1] += (a[1].y * b[1]);
        s[6][0] += (a[1].z * b[0]); s[6][1] += (a[1].z * b[1]);
        s[7][0] += (a[1].w * b[0]); s[7][1] += (a[1].w * b[1]);

        a_vec = (device float4*)((device char*)a_vec + pbytes);
        b_vec = (device float4*)((device char*)b_vec + 32);

    } while(b_vec < Bend);

    
    // c seems to be an 8 x 8 block.
    c[0] = s[0][0];  c[1] = s[0][1];  c = (device float4*)( (device char*)c + 32 );
    c[0] = s[1][0];  c[1] = s[1][1];  c = (device float4*)( (device char*)c + 32 );
    c[0] = s[2][0];  c[1] = s[2][1];  c = (device float4*)( (device char*)c + 32 );
    c[0] = s[3][0];  c[1] = s[3][1];  c = (device float4*)( (device char*)c + 32 );
    c[0] = s[4][0];  c[1] = s[4][1];  c = (device float4*)( (device char*)c + 32 );
    c[0] = s[5][0];  c[1] = s[5][1];  c = (device float4*)( (device char*)c + 32 );
    c[0] = s[6][0];  c[1] = s[6][1];  c = (device float4*)( (device char*)c + 32 );
    c[0] = s[7][0];  c[1] = s[7][1];
}


//typedef struct
//{
//    ushort m, k, n, pbytes, qbytes;
//} MetalMatrixDim;
//
//
//kernel void MatrixMultiply(const device float*       A    [[ buffer(0) ]],
//                           const device float*       B    [[ buffer(1) ]],
//                           device float*             C    [[ buffer(2) ]],
//                           constant MetalMatrixDim&  dims [[ buffer(3) ]],
//                           ushort2                   gid  [[ thread_position_in_grid ]])
//{
//    ushort m = dims.m;
//    ushort k = dims.k;
//    ushort n = dims.n;
//
//    ushort pbytes = dims.pbytes;
//    ushort qbytes = dims.qbytes;
//
//    ushort2 gidIn = ushort2(gid.x << 3, gid.y << 3);
//
//    if (gidIn.x >= m || gidIn.y >= k) return;
//
//    const device float4* a = (const device float4*)(A + gidIn.x);
//    const device float4* b = (const device float4*)(B + gidIn.y);
//
//    C = (device float*)((device char*)C + gidIn.x*qbytes);
//
//    device float4* c = (device float4*)(C + gidIn.y);
//
//    const device float4* Bend = (const device float4*)((const device char*)B + qbytes*n);
//
//    float4 s0  = 0.0f, s1  = 0.0f, s2  = 0.0f, s3  = 0.0f;
//    float4 s4  = 0.0f, s5  = 0.0f, s6  = 0.0f, s7  = 0.0f;
//    float4 s8  = 0.0f, s9  = 0.0f, s10 = 0.0f, s11 = 0.0f;
//    float4 s12 = 0.0f, s13 = 0.0f, s14 = 0.0f, s15 = 0.0f;
//
//    do
//    {
//        float4 aCurr0 = a[0];
//        float4 aCurr1 = a[1];
//        float4 bCurr0 = b[0];
//        float4 bCurr1 = b[1];
//
//        s0   += (aCurr0.x * bCurr0);
//        s2   += (aCurr0.y * bCurr0);
//        s4   += (aCurr0.z * bCurr0);
//        s6   += (aCurr0.w * bCurr0);
//
//        s1   += (aCurr0.x * bCurr1);
//        s3   += (aCurr0.y * bCurr1);
//        s5   += (aCurr0.z * bCurr1);
//        s7   += (aCurr0.w * bCurr1);
//
//        s8   += (aCurr1.x * bCurr0);
//        s10  += (aCurr1.y * bCurr0);
//        s12  += (aCurr1.z * bCurr0);
//        s14  += (aCurr1.w * bCurr0);
//
//        s9   += (aCurr1.x * bCurr1);
//        s11  += (aCurr1.y * bCurr1);
//        s13  += (aCurr1.z * bCurr1);
//        s15  += (aCurr1.w * bCurr1);
//
//        a = (device float4*)((device char*)a + pbytes);
//        b = (device float4*)((device char*)b + qbytes);
//
//    } while(b < Bend);
//
//    c[0] = s0;  c[1] = s1;  c = (device float4*)((device char*)c + qbytes);
//    c[0] = s2;  c[1] = s3;  c = (device float4*)((device char*)c + qbytes);
//    c[0] = s4;  c[1] = s5;  c = (device float4*)((device char*)c + qbytes);
//    c[0] = s6;  c[1] = s7;  c = (device float4*)((device char*)c + qbytes);
//    c[0] = s8;  c[1] = s9;  c = (device float4*)((device char*)c + qbytes);
//    c[0] = s10; c[1] = s11; c = (device float4*)((device char*)c + qbytes);
//    c[0] = s12; c[1] = s13; c = (device float4*)((device char*)c + qbytes);
//    c[0] = s14; c[1] = s15;
//}
