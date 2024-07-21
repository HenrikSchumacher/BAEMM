#pragma once

#include <CL/cl.h>

//#define TOOLS_ENABLE_PROFILER

//#include <complex>
//#include <cblas.h>
//#include <lapack.h>

#ifdef __APPLE__
/// Use these while on a mac. Don't forget to issue the compiler flag `-framework Accelerate`.
///
    #include "submodules/Repulsor/submodules/Tensors/Accelerate.hpp"
#else
/// This should work for OpenBLAS.
    #include "submodules/Repulsor/submodules/Tensors/OpenBLAS.hpp"
#endif

#include "submodules/Repulsor/Repulsor.hpp"

#include "submodules/Repulsor/submodules/Tensors/Sparse.hpp"
#include "submodules/Repulsor/submodules/Tensors/src/Sparse/ApproximateMinimumDegree.hpp"
#include "submodules/Repulsor/submodules/Tensors/GMRES.hpp"
#include "submodules/Repulsor/submodules/Tensors/ConjugateGradient.hpp"

namespace BAEMM
{
    using namespace Tools;
    using namespace Tensors;
    using namespace Repulsor;
    using cmplx  = cl_float2;
    using float4 = cl_float4;
    //https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/index.html#//apple_ref/doc/uid/TP40016642-CH27-SW1
    
    template<
        bool use_lumped_mass_as_precQ_ = true,
        bool use_mass_choleskyQ_ = false
    >
    class Helmholtz_OpenCL
    {
#include "src/Helmholtz_Common/Definitions.hpp"
            
#include "src/Helmholtz_Common/MemberVariables.hpp"
        // OpenCL utilities
        cl_int ret; // return value of the OpenCL commands for bug identification
        
        cl_device_id device_id = NULL;
        cl_context context;
        cl_kernel global_kernel;    // globally saved OpenCL Kernel. Only used in the kernel for the solver mode

        cl_command_queue command_queue;
        
        // OpenCL Device buffers
        cl_mem areas;           // buffer for the areas of the simplices
        cl_mem mid_points;      // buffer for the midpoints of the simplices
        cl_mem normals;         // buffer for the simplex-normals
        cl_mem single_diag;     // diagonal of the SL Operator
        cl_mem tri_coords;
        cl_mem meas_directions; // measurement directkons (m x 3 tensor)
        
        // buffers for in- and output
        cl_mem B_buf = NULL;    
        cl_mem C_buf = NULL;

        LInt B_size = 0;
        LInt C_size = 0;

        // pin host memory (bigger bandwith with pinned memory)
        cl_mem areas_pin;
        cl_mem mid_points_pin;
        cl_mem normals_pin;
        cl_mem single_diag_pin;
        cl_mem tri_coords_pin;
        cl_mem meas_directions_pin;
        
        cl_mem B_buf_pin = NULL;
        cl_mem C_buf_pin = NULL;
        
    public:
    
        template<typename ExtReal,typename ExtInt>
        Helmholtz_OpenCL(
            cptr<ExtReal> vertex_coords_, ExtInt vertex_count_,
            cptr<ExtInt>  triangles_    , ExtInt simplex_count_,
            cptr<ExtReal> meas_directions_, ExtInt meas_count_,
            ExtInt CPU_thread_count_
        )
        :   CPU_thread_count ( int_cast<Int>(CPU_thread_count_)     )
        ,   vertex_count     ( int_cast<Int>(vertex_count_)         )
        ,   simplex_count    ( int_cast<Int>(simplex_count_)        )
        ,   vertex_coords    ( vertex_coords_, vertex_count,  3     )
        ,   triangles        ( triangles_,     simplex_count, 3     )
        ,   meas_count       ( int_cast<Int>(meas_count_)           )
        ,   areas_lumped_inv ( vertex_count, Scalar::Zero<Real> )
        {
            
            // The profile should be reset by a user, not by the class Helmholtz_OpenCL.
            // Mind: One might want to profile more than one class.
            
//            std::filesystem::path path { std::filesystem::current_path() };
//            std::string path_string{ path.string() };
//            Profiler::Clear( path_string );
            
//            tic(ClassName());

             // Get platform and device information            
            cl_platform_id platform_id;  
            cl_uint ret_num_devices;
            cl_uint ret_num_platforms;

            ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
            ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 1, 
                                    &device_id, &ret_num_devices);
            
            if (ret_num_devices == 0)
            {
                eprint(ClassName()+": No OpenCL GPU device available.");
            }

            context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

            // Apple hardware does not support this OpenCL 2.0 feature.
//            command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);

            command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
            
            
            // initialize the Opencl buffers and host pointers
            InitializeBuffers(simplex_count,meas_directions_);
            Initialize();     
            
            clEnqueueWriteBuffer(command_queue, mid_points, CL_FALSE, 0,
                                4 * simplex_count * sizeof(Real), mid_points_ptr, 0, NULL, NULL);
            clEnqueueWriteBuffer(command_queue, normals, CL_FALSE, 0,
                                4 * simplex_count * sizeof(Real), normals_ptr, 0, NULL, NULL);   
            clEnqueueWriteBuffer(command_queue, meas_directions, CL_FALSE, 0,
                                4 * meas_count * sizeof(Real), meas_directions_ptr, 0, NULL, NULL);    
        }
        
        template<typename ExtReal,typename ExtInt>
        Helmholtz_OpenCL(
            cptr<ExtReal> vertex_coords_, ExtInt vertex_count_,
            cptr<ExtInt>  triangles_    , ExtInt simplex_count_,
            cptr<ExtReal> meas_directions_, ExtInt meas_count_,
            ExtInt device_num,
            ExtInt CPU_thread_count_
        )
        :   CPU_thread_count ( int_cast<Int>(CPU_thread_count_)     )
        ,   vertex_count     ( int_cast<Int>(vertex_count_)         )
        ,   simplex_count    ( int_cast<Int>(simplex_count_)        )
        ,   vertex_coords    ( vertex_coords_, vertex_count,  3     )
        ,   triangles        ( triangles_,     simplex_count, 3     )
        ,   meas_count       ( int_cast<Int>(meas_count_)           )
        ,   areas_lumped_inv ( vertex_count, Scalar::Zero<Real> )
        {
            
            // The profile should be reset by a user, not by the class Helmholtz_OpenCL.
            // Mind: One might want to profile more than one class.
            
//            std::filesystem::path path { std::filesystem::current_path() };
//            std::string path_string{ path.string() };
//            Profiler::Clear( path_string );
            
//            tic(ClassName());

             // Get platform and device information
            cl_platform_id platform_id;  
            cl_uint ret_num_platforms;
            
            ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
            
            dump(ret_num_platforms);
            
            cl_device_id device_id_list[8];
            cl_uint ret_num_devices;
            
            ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 8,
                                  device_id_list, &ret_num_devices);

            dump(ret_num_devices);
            
            if (ret_num_devices == 0)
            {
                eprint(ClassName()+": No OpenCL GPU device available.");
            }
            else if (device_num < ret_num_devices)           
            {           
                device_id = device_id_list[device_num];
            }
            else
            {
                device_id = device_id_list[0];
            }
            
            dump(device_id);

            context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

            // Apple hardware does not support the following OpenCL 2.0 feature.
//            command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);

            // Instead we can use this (deprecated) feature:
            command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
            
            // initialize the Opencl buffers and host pointers
            InitializeBuffers(simplex_count,meas_directions_);
            Initialize();     
            
            clEnqueueWriteBuffer(command_queue, mid_points, CL_FALSE, 0,
                                4 * simplex_count * sizeof(Real), mid_points_ptr, 0, NULL, NULL);
            clEnqueueWriteBuffer(command_queue, normals, CL_FALSE, 0,
                                4 * simplex_count * sizeof(Real), normals_ptr, 0, NULL, NULL);   
            clEnqueueWriteBuffer(command_queue, meas_directions, CL_FALSE, 0,
                                4 * meas_count * sizeof(Real), meas_directions_ptr, 0, NULL, NULL);    
        }
        
        ~Helmholtz_OpenCL()
        {
            //clean up
            ret = clEnqueueUnmapMemObject(command_queue,mid_points_pin,(void*)mid_points_ptr,0,NULL,NULL);
            ret = clEnqueueUnmapMemObject(command_queue,normals_pin,(void*)normals_ptr,0,NULL,NULL);
            ret = clEnqueueUnmapMemObject(command_queue,areas_pin,(void*)areas_ptr,0,NULL,NULL);
            ret = clEnqueueUnmapMemObject(command_queue,single_diag_pin,(void*)normals_ptr,0,NULL,NULL);
            ret = clEnqueueUnmapMemObject(command_queue,tri_coords_pin,(void*)normals_ptr,0,NULL,NULL);
            ret = clEnqueueUnmapMemObject(command_queue,meas_directions_pin,(void*)meas_directions_ptr,0,NULL,NULL);
            ret = clEnqueueUnmapMemObject(command_queue,B_buf_pin,(void*)B_ptr,0,NULL,NULL);
            ret = clEnqueueUnmapMemObject(command_queue,C_buf_pin,(void*)C_ptr,0,NULL,NULL);
            ret = clFlush(command_queue);
            ret = clFinish(command_queue);

            ret = clReleaseMemObject(mid_points_pin);
            ret = clReleaseMemObject(normals_pin);
            ret = clReleaseMemObject(areas_pin);
            ret = clReleaseMemObject(single_diag_pin);
            ret = clReleaseMemObject(tri_coords_pin);
            ret = clReleaseMemObject(meas_directions_pin);
            ret = clReleaseMemObject(B_buf_pin);
            ret = clReleaseMemObject(C_buf_pin);

            ret = clReleaseMemObject(mid_points);
            ret = clReleaseMemObject(normals);
            ret = clReleaseMemObject(meas_directions);
            ret = clReleaseMemObject(B_buf);
            ret = clReleaseMemObject(C_buf);
            ret = clReleaseCommandQueue(command_queue);
            ret = clReleaseContext(context);
        }
#include "src/TypeCast.hpp"

#include "src/Helmholtz_OpenCL/InitializeBuffers.hpp"    

#include "src/Helmholtz_Common/Initialize.hpp"

#include "src/Helmholtz_Common/GetSetters.hpp"
        
#include "src/Helmholtz_Common/LoadParameters.hpp"
        
#include "src/Helmholtz_Common/LoadParameters3.hpp"
        
#include "src/Helmholtz_Common/InputOutput.hpp"
        
#include "src/Helmholtz_OpenCL/StringManipulation.hpp"

#include "src/Helmholtz_OpenCL/RequireBuffers.hpp"

//#include "src/Helmholtz_Common/ApplyOperators.hpp"
        
#include "src/Helmholtz_Common/ApplyMassInverse.hpp"
        
#include "src/Helmholtz_Common/ApplyBoundaryOperators.hpp"

#include "src/Helmholtz_Common/ApplyFarFieldOperators.hpp"

#include "src/Helmholtz_Common/ApplyNearFieldOperators.hpp"
        
#include "src/Helmholtz_Common/ApplySingleLayerDiagonal.hpp"

#include "src/Helmholtz_Common/CreateIncidentWave.hpp"

#include "src/Helmholtz_Common/CreateHerglotzWave.hpp"
        
#include "src/Helmholtz_OpenCL/BoundaryOperatorKernel_C.hpp"

#include "src/Helmholtz_OpenCL/BoundaryOperatorKernel_C_SolverMode.hpp"

#include "src/Helmholtz_OpenCL/FarFieldOperatorKernel_C.hpp"

#include "src/Helmholtz_OpenCL/NearFieldOperatorKernel_C.hpp"

#include "src/WaveFunctions_OpenCL/IncidentWaveKernel_Plane_C.hpp"

#include "src/WaveFunctions_OpenCL/IncidentWaveKernel_Radial_C.hpp"

#include "src/WaveFunctions_OpenCL/HerglotzWaveKernel_C.hpp"

#include "src/LinearAlgebraUtilities/HadamardProduct.hpp"

#include "src/Helmholtz_Common/FarField.hpp"


//#include "src/Helmholtz_Metal/BoundaryOperatorKernel_ReIm.hpp
    
    public:
        
        const float * VertexCoordinates() const
        {
            return vertex_coords.data();
        }

        const float * TriangleCoordinates() const
        {
            return tri_coords_ptr;
        }
        
    public:
        
        std::string ClassName() const
        {
            return std::string("Helmholtz_OpenCL")
            + "<" + ToString(use_lumped_mass_as_precQ)
            + "," + ToString(use_mass_choleskyQ) 
            + ">";
        }
    };
        
} // namespace BAEMM
