#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// #ifdef __APPLE__
// #include <OpenCL/opencl.h>
// #else
// #include <C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include/CL/cl.h>
// #endif

#define  MAX_SOURCE_SIZE (0x100000)

char* manipulate_string(const char* str, Complex* coeff, Int& block_size, Int& k_chunk_size, size_t& str_length)
{
    char* result;
    result = (char*)malloc(MAX_SOURCE_SIZE);
    strcat(result , "#define  block_size ");
    sprintf(result , "%s %d" , result , block_size);
    strcat(result,"\n");
    strcat(result , "#define  k_chunk_size ");
    sprintf(result , "%s %d" , result, k_chunk_size);
    strcat(result,"\n");
    
    if (coeff[1].real() == 0.0f)
    {
        strcat(result,"__constant bool Re_single_layer = false;\n");
    }
    else
    {
        strcat(result,"__constant bool Re_single_layer = true;\n");
    }
    if (coeff[1].imag() == 0.0f)
    {
        strcat(result,"__constant bool Im_single_layer = false;\n");
    }
    else
    {
        strcat(result,"__constant bool Im_single_layer = true;\n");
    }


    if (coeff[2].real() == 0.0f)
    {
        strcat(result,"__constant bool Re_double_layer = false;\n");
    }
    else
    {
        strcat(result,"__constant bool Re_double_layer = true;\n");
    }
    if (coeff[2].imag() == 0.0f)
    {
        strcat(result,"__constant bool Im_double_layer = false;\n");
    }
    else
    {
        strcat(result,"__constant bool Im_double_layer = true;\n");
    }


    if (coeff[3].real() == 0.0f)
    {
        strcat(result,"__constant bool Re_adjdbl_layer = false;\n");
    }
    else
    {
        strcat(result,"__constant bool Re_adjdbl_layer = true;\n");
    }
    if (coeff[3].imag() == 0.0f)
    {
        strcat(result,"__constant bool Im_adjdbl_layer = false;\n");
    }
    else
    {
        strcat(result,"__constant bool Im_adjdbl_layer = true;\n");
    }
    str_length += strlen(result);
    strcat(result,str);

    return result;
}   