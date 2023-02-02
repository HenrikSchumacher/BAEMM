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
        
        ptic(tag);
        
        if( pipelines.count(fun_fullname) == 0 )
        {
            std::stringstream full_code;
            
            if( param_types.size() != param_names.size() )
            {
                eprint("CreatePipeline: param_types.size() != param_names.size().");
                ptoc(tag);
                return nullptr;
            }
            
            if( param_types.size() != param_vals.size() )
            {
                eprint("CreatePipeline: param_types.size() != param_vals.size().");
                ptoc(tag);
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
            
            MTL::Library * lib = device->newLibrary(
                code_NS_String,
                nullptr, // <-- crucial for distinguishing from the function that loads from file
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
                ptoc(tag);
                return pipelines[fun_fullname];
            }
            else
            {
                eprint(std::string("CreatePipeline: Did not find Metal kernel ") + fun_name +" in source code.");
                ptoc(tag);
                return nullptr;
            }
        }
        else
        {
            ptoc(tag);
            return pipelines[fun_fullname];
        }
    }
