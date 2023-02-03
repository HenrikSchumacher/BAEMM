#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

#include "MatrixPanelRowMajor.hpp"

namespace HeavyMetal
{
    //https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/index.html#//apple_ref/doc/uid/TP40016642-CH27-SW1
    

    class BLAS_Capella
    {
    public:
        
        using Int        = uint32_t;
        using Real       = float;
        using Complex    = std::complex<Real>;
        
        using UInt        = uint32_t;
        
        using NS::StringEncoding::UTF8StringEncoding;
        
        static constexpr Real zero  = 0;
        static constexpr Real one   = 1;
        static constexpr Real two   = 2;
        static constexpr Real three = 3;
        
        static constexpr Real half  = one / two;
        static constexpr Real third = one / three;
        
        static constexpr Real pi      = M_PI;
        static constexpr Real two_pi  = two * pi;
        static constexpr Real four_pi = two * two_pi;
        
//        static constexpr Real one_over_two_pi  = one / two_pi;
        static constexpr Real one_over_four_pi = one / four_pi;
            
        
    private:
        
        MTL::Device* device;

        std::map<std::string, MTL::ComputePipelineState * > pipelines;
        
        MTL::CommandQueue * command_queue;
        
    public:
        
        BLAS_Capella() = delete;
        
        BLAS_Capella( MTL::Device* device_ )
        :   device( device_ )
        {
            command_queue = device->newCommandQueue();
            
            if( command_queue == nullptr )
            {
                std::cout << "Failed to find the command queue." << std::endl;
                return;
            }
        }
        
        ~BLAS_Capella() = default;

    private:
        
        MTL::ComputePipelineState * GetPipelineState(
            const std::string & fun_name,       // name of function in code string
            const std::string & code,           // string of actual Metal code
            const std::vector<std::string> & param_types,    // types of compile-time parameters (converted to string)
            const std::vector<std::string> & param_names,    // name of compile-time parameters
            const std::vector<std::string> & param_vals     // values of compile-time parameters
        )
        {
            std::stringstream fun_fullname_stream;
            
            fun_fullname_stream << fun_name;
            
            for( const auto & s : param_vals )
            {
                fun_fullname_stream << "_" << s;
            }
            
            std::string fun_fullname = fun_fullname_stream.str();
            
            std::string tag = "GetPipelineState(" + fun_fullname + ")";
            
//            tic(tag);
            
            if( pipelines.count(fun_fullname) == 0 )
            {
                std::stringstream full_code;
                
                
                if( param_types.size() != param_names.size() )
                {
                    eprint("CreatePipeline: param_types.size() != param_names.size().");
//                    toc(tag);
                    return nullptr;
                }
                
                
                if( param_types.size() != param_vals.size() )
                {
                    eprint("CreatePipeline: param_types.size() != param_vals.size().");
//                    toc(tag);
                    return nullptr;
                }
                
                std::size_t param_count = param_types.size();
                
                // Create compile-time constant. Will be prependend to code string.
                for( std::size_t i = 0; i < param_count; ++i )
                {
                    full_code << "constant constexpr " << param_types[i] << " " << param_names[i] << " = " << param_vals[i] <<";\n";
                }
                
                full_code << code;
                
                NS::String * code_NS_String = NS::String::string(full_code.str().c_str(), UTF8StringEncoding);
                
                NS::Error *error = nullptr;
                
                MTL::CompileOptions * opt = MTL::CompileOptions::alloc();
                
                opt->init();
                
                opt->setFastMathEnabled(true);
                
                
                MTL::Library * lib = device->newLibrary(
                    code_NS_String,
                    opt, // <-- crucial for distinguishing from the function that loads from file
                    &error
                );
                
                if( lib == nullptr )
                {
                    std::cout << "Failed to compile library from string for function "
                    << fun_fullname << ", error "
                    << error->description()->utf8String() << std::endl;
//                    std::exit(-1);
                    
                    return nullptr;
                }
                
                bool found = false;
                
                // Go through all functions in the library to find ours.
                for( NS::UInteger i = 0; i < lib->functionNames()->count(); ++i )
                {
                    found = true;
                    
                    auto name_nsstring = lib->functionNames()->object(i)->description();
                    
                    if( fun_name == name_nsstring->utf8String() )
                    {
                        // This MTL::Function object is needed only temporarily.
                        MTL::Function * fun = lib->newFunction(name_nsstring);
                        
                        // Create pipeline from function.
                        pipelines[fun_fullname] = device->newComputePipelineState(fun, &error);
                        
                        if( pipelines[fun_fullname] == nullptr )
                        {
                            std::cout << "Failed to created pipeline state object for "
                            << fun_name << ", error "
                            << error->description()->utf8String() << std::endl;
                            return nullptr;
                        }
                    }
                }
                
                if( found )
                {
//                    print(std::string("CreatePipeline: Found Metal kernel ") + fun_name +".");
//                    toc(tag);
                    return pipelines[fun_fullname];
                }
                else
                {
//                    eprint(std::string("CreatePipeline: Did not find Metal kernel ") + fun_name +" in source code.");
//                    toc(tag);
                    return nullptr;
                }
            }
            else
            {
//                toc(tag);
                return pipelines[fun_fullname];
            }
        }
        
    public:
        
        void GEMM_CM_Ref(
            const uint M,
            const uint N,
            const uint K,
            const float   alpha,
            MTL::Buffer * A,
            MTL::Buffer * B,
            const float   beta,
            MTL::Buffer * C
        )
        {
            cblas_sgemm(
                CblasColMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                alpha, reinterpret_cast<float*>(A->contents()), M,
                       reinterpret_cast<float*>(B->contents()), K,
                beta,  reinterpret_cast<float*>(C->contents()), M
            );
        }
        
        void GEMM_RM_Ref(
            const uint M,
            const uint N,
            const uint K,
            const float   alpha,
            MTL::Buffer * A,
            MTL::Buffer * B,
            const float   beta,
            MTL::Buffer * C
        )
        {
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                alpha, reinterpret_cast<float*>(A->contents()), K,
                       reinterpret_cast<float*>(B->contents()), N,
                beta,  reinterpret_cast<float*>(C->contents()), N
            );
        }
        
        void GEMM_RM_Naive(
            const uint M,
            const uint N,
            const uint K,
            const float   alpha,
            MTL::Buffer * A,
            MTL::Buffer * B,
            const float   beta,
            MTL::Buffer * C
        )
        {
            ptr<float> a = reinterpret_cast<float*>(A->contents());
            ptr<float> b = reinterpret_cast<float*>(B->contents());
            mut<float> c = reinterpret_cast<float*>(C->contents());
            
            zerofy_buffer(c,M*N,8);
            
            #pragma omp parallel for num_threads( 8 )
            for( uint k = 0; k < K; ++k )
            {
                for( uint i = 0; i < M; ++i )
                {
                    for( uint j = 0; j < N; ++j )
                    {
                        c[N*i+j]+= a[K*i+k] * b[N*k+j];
                    }
                }
            }
        }
        
        
        void GEMM_CM_Naive(
            const uint M,
            const uint N,
            const uint K,
            const float   alpha,
            MTL::Buffer * A,
            MTL::Buffer * B,
            const float   beta,
            MTL::Buffer * C
        )
        {
            ptr<float> a = reinterpret_cast<float*>(A->contents());
            ptr<float> b = reinterpret_cast<float*>(B->contents());
            mut<float> c = reinterpret_cast<float*>(C->contents());
            
            zerofy_buffer(c,M*N,8);
            
        #pragma omp parallel for num_threads( 8 )
            for( uint k = 0; k < K; ++k )
            {
                for( uint i = 0; i < M; ++i )
                {
                    for( uint j = 0; j < N; ++j )
                    {
                        c[M*j+i]+= a[M*k+i] * b[K*j+k];
                    }
                }
            }
        }
        
#include "AddReduce/AddReduce.hpp"

#include "GEMM/GEMM_CM.hpp"
        
#include "GEMM/GEMM_RM_NVidea.hpp"

#include "GEMM/GEMM_RM.hpp"
        

        
//#include "GEMM/GEMM_PRM.hpp"
        
    public:
        
        std::string ClassName() const
        {
            return "BLAS_Capella";
        }
        
    };
        
} // namespace HeavyMetal

