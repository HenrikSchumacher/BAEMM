#pragma once

#ifdef __APPLE__
/// Use these while on a mac. Don't forget to issue the compiler flag `-framework Accelerate`.
///
    #include "submodules/Repulsor/submodules/Tensors/Accelerate.hpp"

    #define CL_TARGET_OPENCL_VERSION 120
    #include <OpenCL/OpenCL.h>
#else
/// This should work for OpenBLAS.
    #include "submodules/Repulsor/submodules/Tensors/OpenBLAS.hpp"

    #include <CL/cl.h>
#endif

#include "submodules/Repulsor/Repulsor.hpp"

#include "submodules/Repulsor/submodules/Tensors/Sparse.hpp"
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
        cl_int ret; /**< return value of the OpenCL commands for bug identification*/
        
        cl_device_id device_id = nullptr; /**< Parse the ID of desired GPU-device. */
        cl_context context; /**< Global context for command queues. */
        
        cl_kernel bdr_kernel;    /**< Globally saved OpenCL Kernel for the boundary operator.*/

        
        cl_command_queue command_queue;
        
        // OpenCL Device buffers
        cl_mem mid_points          = nullptr; /**< buffer for the midpoints of the simplices */
        cl_mem normals             = nullptr; /**< buffer for the simplex-normals */
        cl_mem meas_directions     = nullptr; /**< measurement directions (m x 3 tensor) */
        
        // buffers for in- and output
        cl_mem B_buf               = nullptr; /**< Buffer on GPU for input */
        cl_mem C_buf               = nullptr; /**< Buffer on GPU for output */

        LInt B_size = 0; /**< Size of B_buf */
        LInt C_size = 0; /**< Size of C_buf */

        // pin host memory (bigger bandwith with pinned memory)
        cl_mem mid_points_pin      = nullptr; /**< Pinned buffer for midpoints on host-device */
        cl_mem normals_pin         = nullptr; /**< Pinned buffer for normals on host-device */
        cl_mem meas_directions_pin = nullptr; /**< Pinned buffer for measurement directions on host-device */
        
        cl_mem B_buf_pin           = nullptr; /**< Pinned buffer for input on host-device */
        cl_mem C_buf_pin           = nullptr; /**< Pinned buffer for output on host-device */
        
        cl_mem m_kappa             = nullptr; /**< Globally saved buffer for the array of wavenumbers. This is necessary for calling the same (pre-compiled) boundary operator kernel multiple times. */
        cl_mem m_coeff             = nullptr; /**< Globally saved buffer for the array of coeffiients for boundary operators. This is necessary for calling the same (pre-compiled) boundary operator kernel multiple times. */
        cl_mem m_n                 = nullptr; /**< Globally saved buffer for number of simplices. This is necessary for calling the same (pre-compiled) boundary operator kernel multiple times. */
        cl_mem m_wave_count        = nullptr; /**< Globally saved buffer for total number of incident waves. This is necessary for calling the same (pre-compiled) boundary operator kernel multiple times. */
        

        // const char * clBuildOpts = nullptr;
       const char * clBuildOpts = "-cl-fast-relaxed-math";
//        const char * clBuildOpts = "-cl-finite-math-only -cl-mad-enable";
        
        static constexpr bool print_kernel_codeQ = false;
        
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

        /**
         * @brief Construct a new Helmholtz_OpenCL object.
         * 
         * @tparam ExtReal: External Real precision. We recommend double precision.
         * @tparam ExtInt: External Int type.
         * @param vertex_coords_: vertex_count_ x 3 real array containing the coordinates of the vertices. 
         * @param vertex_count_: Number of vertices of the parsed mesh. 
         * @param triangles_: simplex_count_ x 3 integer array representing the connectivity list of the mesh.
         * @param simplex_count_: Number of simplices of the parsed mesh. 
         * @param meas_directions_: meas_count_ x 3 real array storing the measurement directions on the S^2 for the far field.
         * @param meas_count_: Number of measurement directions on the S^2 for the far field. 
         * @param CPU_thread_count_: Number of threads the CPU shall use. 
         */
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
        ,   meas_count       ( int_cast<Int>(meas_count_)           )
        ,   vertex_coords    ( vertex_coords_, vertex_count,  3     )
        ,   triangles        ( triangles_,     simplex_count, 3     )
        ,   areas_lumped_inv ( vertex_count, Scalar::Zero<Real> )
        {
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
            
            // initialize the OpenCL buffers and host pointers
            InitializeBuffers(simplex_count,meas_directions_);
            Initialize();     
            
            // Write the buffers needed by all GPU-kernels to the device buffer.
            clEnqueueWriteBuffer(command_queue,mid_points,     CL_FALSE,0,4 * simplex_count * sizeof(Real),mid_points_ptr,     0,nullptr,nullptr);
            clEnqueueWriteBuffer(command_queue,normals,        CL_FALSE,0,4 * simplex_count * sizeof(Real),normals_ptr,        0,nullptr,nullptr);
            clEnqueueWriteBuffer(command_queue,meas_directions,CL_FALSE,0,4 * meas_count    * sizeof(Real),meas_directions_ptr,0,nullptr,nullptr);
        }
        
        /**
         * @brief Construct a new Helmholtz_OpenCL object.
         * 
         * @tparam ExtReal: External Real precision. We recommend double precision.
         * @tparam ExtInt: External Int type.
         * @param vertex_coords_: vertex_count_ x 3 real array containing the coordinates of the vertices. 
         * @param vertex_count_: Number of vertices of the parsed mesh. 
         * @param triangles_: simplex_count_ x 3 integer array representing the connectivity list of the mesh.
         * @param simplex_count_: Number of simplices of the parsed mesh. 
         * @param meas_directions_: meas_count_ x 3 real array storing the measurement directions on the S^2 for the far field.
         * @param meas_count_: Number of measurement directions on the S^2 for the far field. 
         * @param device_num: The ID of desired GPU-device. Always an integer. For instance: if you have multiple Nvidia CUDA GPUs use the bash command 'nvidia-smi' to get the information.
         * @param CPU_thread_count_: Number of threads the CPU shall use. 
         */
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
        ,   meas_count       ( int_cast<Int>(meas_count_)           )
        ,   vertex_coords    ( vertex_coords_, vertex_count,  3     )
        ,   triangles        ( triangles_,     simplex_count, 3     )
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
            
            ReleaseParameters();

            free(areas_ptr);
            free(single_diag_ptr);
            free(tri_coords_ptr);
            
            ret = clEnqueueUnmapMemObject(command_queue,mid_points_pin,     (void*)mid_points_ptr,     0,nullptr,nullptr);
            ret = clEnqueueUnmapMemObject(command_queue,normals_pin,        (void*)normals_ptr,        0,nullptr,nullptr);
            ret = clEnqueueUnmapMemObject(command_queue,meas_directions_pin,(void*)meas_directions_ptr,0,nullptr,nullptr);
            ret = clEnqueueUnmapMemObject(command_queue,B_buf_pin,          (void*)B_ptr,              0,nullptr,nullptr);
            ret = clEnqueueUnmapMemObject(command_queue,C_buf_pin,          (void*)C_ptr,              0,nullptr,nullptr);
            ret = clFlush(command_queue);
            ret = clFinish(command_queue);

            ret = clReleaseMemObject(mid_points_pin);
            ret = clReleaseMemObject(normals_pin);
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
        
#include "src/Helmholtz_Common/IncidentWaveKernel_Plane.hpp"

#include "src/Helmholtz_Common/IncidentWaveKernel_Radial.hpp"

#include "src/Helmholtz_Common/CreateHerglotzWave.hpp"
        
#include "src/Helmholtz_OpenCL/BoundaryOperatorKernel.hpp"

#include "src/Helmholtz_OpenCL/FarFieldOperatorKernel.hpp"

#include "src/Helmholtz_OpenCL/NearFieldOperatorKernel.hpp"

#include "src/Helmholtz_OpenCL/HerglotzWaveKernel.hpp"

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
