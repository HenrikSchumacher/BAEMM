#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// #ifdef __APPLE__
// #include <OpenCL/opencl.h>
// #else
// #include <C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/include/CL/cl.h>
// #endif

#define  MAX_SOURCE_SIZE (0x100000)

char* manipulate_string(const char* str, Int& block_size, Int& k_chunk_size, size_t& str_length)
{
    // first we add definitions of block_size and k_chunk_size as macros to the kernel string
    char* result = (char*)malloc(MAX_SOURCE_SIZE);
    strcpy(result , "#define  block_size ");
    sprintf(result , "%s %d" , result , block_size);
    strcat(result,"\n");
    strcat(result , "#define  k_chunk_size ");
    sprintf(result , "%s %d" , result, k_chunk_size);
    strcat(result,"\n");
    
    // now we add booleans for the coefficients being !=0, important as we need global booleans for the threads to interpret them right simultaneously
    if (!Re_single_layer)
    {
        strcat(result,"__constant bool Re_single_layer = false;\n");
    }
    else
    {
        strcat(result,"__constant bool Re_single_layer = true;\n");
    }
    if (!Im_single_layer)
    {
        strcat(result,"__constant bool Im_single_layer = false;\n");
    }
    else
    {
        strcat(result,"__constant bool Im_single_layer = true;\n");
    }


    if (!Re_double_layer)
    {
        strcat(result,"__constant bool Re_double_layer = false;\n");
    }
    else
    {
        strcat(result,"__constant bool Re_double_layer = true;\n");
    }
    if (!Im_double_layer)
    {
        strcat(result,"__constant bool Im_double_layer = false;\n");
    }
    else
    {
        strcat(result,"__constant bool Im_double_layer = true;\n");
    }


    if (!Re_adjdbl_layer)
    {
        strcat(result,"__constant bool Re_adjdbl_layer = false;\n");
    }
    else
    {
        strcat(result,"__constant bool Re_adjdbl_layer = true;\n");
    }
    if (!Im_adjdbl_layer)
    {
        strcat(result,"__constant bool Im_adjdbl_layer = false;\n");
    }
    else
    {
        strcat(result,"__constant bool Im_adjdbl_layer = true;\n");
    }

    // update string length and add the original source string
    strcat(result,str);
    str_length = strlen(result);

    return result;
}   