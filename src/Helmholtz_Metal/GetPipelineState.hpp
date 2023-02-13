private:
    
    NS::SharedPtr<MTL::ComputePipelineState> GetPipelineState(
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
        
        std::string tag = ClassName()+"::GetPipelineState(" + fun_fullname + ")";
        
        ptic(tag);
        
        if( pipelines.count(fun_fullname) == 0 )
        {
            std::stringstream full_code;
            
            if( param_types.size() != param_names.size() )
            {
                eprint(tag+": param_types.size() != param_names.size().");
                ptoc(tag);
                std::exit(-1);
            }
            
            if( param_types.size() != param_vals.size() )
            {
                eprint(tag+": param_types.size() != param_vals.size().");
                ptoc(tag);
                std::exit(-1);
            }
            
            std::size_t param_count = param_types.size();
            
            // Create compile-time constant. Will be prependend to code string.
            for( std::size_t i = 0; i < param_count; ++i )
            {
                full_code << "constant constexpr " << param_types[i] << " " << param_names[i] << " = " << param_vals[i] <<";\n";
            }
            
            full_code << code;
            
            NS::SharedPtr<NS::String> code_NS_String = NS::TransferPtr( NS::String::string(full_code.str().c_str(), UTF8StringEncoding) );
            
            NS::Error * error = nullptr;
            
            NS::SharedPtr<MTL::Library> lib = NS::TransferPtr( device->newLibrary(
                    code_NS_String.get(),
                    nullptr, // <-- for distinguishing from the function that loads from file
                    &error
                )
            );
            
            if( lib.get() == nullptr )
            {
                eprint(tag+": Failed to compile library from string for function.");
                valprint("Error message", error->description()->utf8String() );
                std::exit(-1);
            }
            
            bool found = false;
            
            // Go through all functions in the library to find ours.
            for( NS::UInteger i = 0; i < lib->functionNames()->count(); ++i )
            {
                auto name_nsstring = lib->functionNames()->object(i)->description();
                
                if( fun_name == name_nsstring->utf8String() )
                {
                    found = true;
                
                    
                    // This MTL::Function object is needed only temporarily.
                    NS::SharedPtr<MTL::Function> fun = NS::TransferPtr(lib->newFunction(name_nsstring));
                    
                    // Create pipeline from function.
                    pipelines[fun_fullname] = NS::TransferPtr(device->newComputePipelineState(fun.get(), &error));
                    
                    if( pipelines[fun_fullname].get() == nullptr )
                    {
                        eprint(tag+": Failed to created pipeline state object.");
                        valprint("Error message", error->description()->utf8String() );
                        std::exit(-1);
                        ;
                    }
                }
            }
            
            if( found )
            {
//                print(tag+": Metal kernel found in source code.");
                ptoc(tag);
                return pipelines[fun_fullname];
            }
            else
            {
                eprint(tag+": Metal kernel not found in source code.");
                ptoc(tag);
                std::exit(-1);
            }
        }
        else
        {
            ptoc(tag);
            return pipelines[fun_fullname];
        }
    }
