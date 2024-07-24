#pragma once

#include <CL/cl.h>

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

#include "BAEMM_Common.hpp"

namespace BAEMM
{
    using cmplx  = cl_float2;
    using float4 = cl_float4;
    
    class Helmholtz_OpenCL
    {
#include "src/Helmholtz_Common/Definitions.hpp"
            
#include "src/Helmholtz_Common/MemberVariables.hpp"
        // OpenCL utilities
        cl_int ret; // return value of the OpenCL commands for bug identification
        
        cl_device_id device_id = nullptr;
        cl_context context;
        
        cl_kernel bdr_kernel;    // Globally saved OpenCL Kernel for the boundary operator.

        
        cl_command_queue command_queue;
        
        // OpenCL Device buffers
        cl_mem areas = nullptr;           // buffer for the areas of the simplices
        cl_mem mid_points = nullptr;      // buffer for the midpoints of the simplices
        cl_mem normals = nullptr;         // buffer for the simplex-normals
        cl_mem single_diag = nullptr;     // diagonal of the SL Operator
        cl_mem tri_coords = nullptr;
        cl_mem meas_directions = nullptr; // measurement directkons (m x 3 tensor)
        
        // buffers for in- and output
        cl_mem B_buf = nullptr;    
        cl_mem C_buf = nullptr;

        LInt B_size = 0;
        LInt C_size = 0;

        // pin host memory (bigger bandwith with pinned memory)
        cl_mem areas_pin = nullptr;
        cl_mem mid_points_pin = nullptr;
        cl_mem normals_pin = nullptr;
        cl_mem single_diag_pin = nullptr;
        cl_mem tri_coords_pin = nullptr;
        cl_mem meas_directions_pin = nullptr;
        
        cl_mem B_buf_pin = nullptr;
        cl_mem C_buf_pin = nullptr;
        
        cl_mem d_kappa      = nullptr;
        cl_mem d_coeff      = nullptr;
        cl_mem d_n          = nullptr;
        cl_mem d_wave_count = nullptr;
        

//        const char * clBuildOpts = nullptr;
        const char * clBuildOpts = "-cl-fast-relaxed-math";
//        const char * clBuildOpts = "-cl-finite-math-only -cl-mad-enable";
        
    protected:
        
        void cl_check_ret(
            const std::string & tag, const std::string & cl_name
        ) const
        {
            if( ret != 0 )
            {
                eprint( tag + ": Call to " + cl_name + " failed. Error code = " + ToString(ret) + ".");
            }
        }
        
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

             // Get platform and device information            
            cl_platform_id platform_id;  
            cl_uint ret_num_devices;
            cl_uint ret_num_platforms;

            ret = clGetPlatformIDs(1,&platform_id,&ret_num_platforms);
            ret = clGetDeviceIDs(platform_id,CL_DEVICE_TYPE_GPU,1,&device_id,&ret_num_devices);
            
            if (ret_num_devices == 0)
            {
                eprint(ClassName()+": No OpenCL GPU device available.");
            }

            context = clCreateContext(nullptr,1,&device_id,nullptr,nullptr,&ret);

#ifdef __APPLE__
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
            // Apple froze OpenCL support at version 1.2. Thus, the OpenCL 2.0
            // feature clCreateCommandQueueWithProperties is not supported.
            // Instead we can use this (deprecated) feature:
            command_queue = clCreateCommandQueue(context,device_id,0,&ret);
    #pragma clang diagnostic pop
#else
            command_queue = clCreateCommandQueueWithProperties(context,device_id,0,&ret);
#endif
            
            // initialize the Opencl buffers and host pointers
            InitializeBuffers(simplex_count,meas_directions_);
            Initialize();     
            
            clEnqueueWriteBuffer(command_queue,mid_points,     CL_FALSE,0,4 * simplex_count * sizeof(Real),mid_points_ptr,     0,nullptr,nullptr);
            clEnqueueWriteBuffer(command_queue,normals,        CL_FALSE,0,4 * simplex_count * sizeof(Real),normals_ptr,        0,nullptr,nullptr);
            clEnqueueWriteBuffer(command_queue,meas_directions,CL_FALSE,0,4 * meas_count    * sizeof(Real),meas_directions_ptr,0,nullptr,nullptr);
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
            std::string tag = ClassName()+"(...)";
            
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
            
            cl_device_id device_id_list[8];
            cl_uint      ret_num_devices;
            
            ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, 8, device_id_list, &ret_num_devices);
//            
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
            
            logprint(DeviceInfo());
            
            context = clCreateContext( nullptr, 1, &device_id, nullptr, nullptr, &ret);

#ifdef __APPLE__
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
            // Apple froze OpenCL support at version 1.2. Thus, the OpenCL 2.0
            // feature clCreateCommandQueueWithProperties is not supported.
            // Instead we can use this (deprecated) feature:
            command_queue = clCreateCommandQueue(context,device_id,0,&ret);
    #pragma clang diagnostic pop
#else
            command_queue = clCreateCommandQueueWithProperties(context,device_id,0,&ret);
#endif
            
            // Initialize the OpenCL buffers and host pointers.
            InitializeBuffers(simplex_count,meas_directions_);
            Initialize();     
            
            clEnqueueWriteBuffer(command_queue,mid_points,     CL_FALSE,0,4 * simplex_count * sizeof(Real),mid_points_ptr,     0,nullptr,nullptr);
            clEnqueueWriteBuffer(command_queue,normals,        CL_FALSE,0,4 * simplex_count * sizeof(Real),normals_ptr,        0,nullptr,nullptr);
            clEnqueueWriteBuffer(command_queue,meas_directions,CL_FALSE,0,4 * meas_count    * sizeof(Real),meas_directions_ptr,0,nullptr,nullptr);
        }
        
        ~Helmholtz_OpenCL()
        {
            //clean up
            ret = clEnqueueUnmapMemObject(command_queue,mid_points_pin,     (void*)mid_points_ptr,     0,nullptr,nullptr);
            ret = clEnqueueUnmapMemObject(command_queue,normals_pin,        (void*)normals_ptr,        0,nullptr,nullptr);
            ret = clEnqueueUnmapMemObject(command_queue,areas_pin,          (void*)areas_ptr,          0,nullptr,nullptr);
            ret = clEnqueueUnmapMemObject(command_queue,single_diag_pin,    (void*)normals_ptr,        0,nullptr,nullptr);
            ret = clEnqueueUnmapMemObject(command_queue,tri_coords_pin,     (void*)normals_ptr,        0,nullptr,nullptr);
            ret = clEnqueueUnmapMemObject(command_queue,meas_directions_pin,(void*)meas_directions_ptr,0,nullptr,nullptr);
            ret = clEnqueueUnmapMemObject(command_queue,B_buf_pin,          (void*)B_ptr,              0,nullptr,nullptr);
            ret = clEnqueueUnmapMemObject(command_queue,C_buf_pin,          (void*)C_ptr,              0,nullptr,nullptr);
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
        
//#include "src/Helmholtz_Common/LoadParameters3.hpp"
        
#include "src/Helmholtz_Common/InputOutput.hpp"
        
#include "src/Helmholtz_OpenCL/StringManipulation.hpp"

#include "src/Helmholtz_OpenCL/RequireBuffers.hpp"
        
#include "src/Helmholtz_Common/ApplyMassInverse.hpp"
        
#include "src/Helmholtz_Common/ApplyBoundaryOperators.hpp"

#include "src/Helmholtz_Common/ApplyFarFieldOperators.hpp"

#include "src/Helmholtz_Common/ApplyNearFieldOperators.hpp"
        
#include "src/Helmholtz_Common/ApplySingleLayerDiagonal.hpp"

#include "src/Helmholtz_Common/CreateIncidentWave.hpp"

#include "src/Helmholtz_Common/CreateHerglotzWave.hpp"
        
#include "src/Helmholtz_OpenCL/BoundaryOperatorKernel.hpp"

#include "src/Helmholtz_OpenCL/FarFieldOperatorKernel.hpp"

#include "src/Helmholtz_OpenCL/NearFieldOperatorKernel.hpp"

#include "src/WaveFunctions_OpenCL/IncidentWaveKernel_Plane.hpp"

#include "src/WaveFunctions_OpenCL/IncidentWaveKernel_Radial.hpp"

#include "src/WaveFunctions_OpenCL/HerglotzWaveKernel.hpp"

#include "src/LinearAlgebraUtilities/HadamardProduct.hpp"

#include "src/Helmholtz_Common/FarField.hpp"
    
    public:
        
        const float * VertexCoordinates() const
        {
            return vertex_coords.data();
        }

        const float * TriangleCoordinates() const
        {
            return tri_coords_ptr;
        }
        
        std::string DeviceInfo()
        {
            std::string tag = ClassName()+"::DeviceInfo";
            
            std::stringstream s;
            
            constexpr Size_T str_len = 256;
            char str [str_len];
            
            s << tag << std::endl;
            
            ret = clGetDeviceInfo(device_id, CL_DEVICE_VENDOR , str_len, &str[0], nullptr);
            cl_check_ret( tag, "clGetDeviceInfo" );
            s << "CL_DEVICE_VENDOR                  " << " = " << &str[0] << std::endl;
            
            ret = clGetDeviceInfo(device_id, CL_DEVICE_NAME   , str_len, &str[0], nullptr);
            cl_check_ret( tag, "clGetDeviceInfo" );
            s << "CL_DEVICE_NAME                    " << " = " << &str[0] << std::endl;
            
            ret = clGetDeviceInfo(device_id, CL_DEVICE_VERSION, str_len, &str[0], nullptr);
            cl_check_ret( tag, "clGetDeviceInfo" );
            s << "CL_DEVICE_VERSION                 " << " = " << &str[0] << std::endl;
            
            ret = clGetDeviceInfo(device_id, CL_DRIVER_VERSION, str_len, &str[0], nullptr);
            cl_check_ret( tag, "clGetDeviceInfo" );
            s << "CL_DRIVER_VERSION                 " << " = " << &str[0] << std::endl;
            
            
            cl_ulong x;
            constexpr Size_T x_len = sizeof(x);
            
            ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE      , x_len, &x, nullptr);
            cl_check_ret( tag, "clGetDeviceInfo" );
            s << "CL_DEVICE_MAX_MEM_ALLOC_SIZE      " << " = " << x << std::endl;
            
            ret = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE         , x_len, &x, nullptr);
            cl_check_ret( tag, "clGetDeviceInfo" );
            s << "CL_DEVICE_GLOBAL_MEM_SIZE         " << " = " << x << std::endl;
            
            ret = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE          , x_len, &x, nullptr);
            cl_check_ret( tag, "clGetDeviceInfo" );
            s << "CL_DEVICE_LOCAL_MEM_SIZE          " << " = " << x << std::endl;
            
            ret = clGetDeviceInfo(device_id, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, x_len, &x, nullptr);
            cl_check_ret( tag, "clGetDeviceInfo" );
            s << "CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE" << " = " << x << std::endl;
            
            return s.str();
        }
        
    public:
        
        std::string ClassName() const
        {
            return std::string("Helmholtz_OpenCL");
        }
    };
        
} // namespace BAEMM
