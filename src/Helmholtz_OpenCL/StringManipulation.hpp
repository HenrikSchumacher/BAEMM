std::string CreateSourceString(
    const char * str, const Int block_size, const Int k_chunk_size
)
{
    std::stringstream s;
    
    s << "#define block_size      " << block_size      << std::endl;
    s << "#define k_chunk_size    " << k_chunk_size    << std::endl;
    
    s << "#define Re_single_layer " << Re_single_layer << std::endl;
    s << "#define Im_single_layer " << Im_single_layer << std::endl;

    s << "#define Re_double_layer " << Re_double_layer << std::endl;
    s << "#define Im_double_layer " << Im_double_layer << std::endl;

    s << "#define Re_adjdbl_layer " << Re_adjdbl_layer << std::endl;
    s << "#define Im_adjdbl_layer " << Im_adjdbl_layer << std::endl;
    
//    s << "__constant int block_size       = " << block_size      << ";" << std::endl;
//    s << "__constant int k_chunk_size     = " << k_chunk_size    << ";" << std::endl;

//      // Now we add booleans for the coefficients being !=0, important as we need global booleans for the threads to interpret them right simultaneously.
//    s << "__constant bool Re_single_layer = " << Re_single_layer << ";" << std::endl;
//    s << "__constant bool Im_single_layer = " << Im_single_layer << ";" << std::endl;
//
//    s << "__constant bool Re_double_layer = " << Re_double_layer << ";" << std::endl;
//    s << "__constant bool Im_double_layer = " << Im_double_layer << ";" << std::endl;
//
//    s << "__constant bool Re_adjdbl_layer = " << Re_adjdbl_layer << ";" << std::endl;
//    s << "__constant bool Im_adjdbl_layer = " << Im_adjdbl_layer << ";" << std::endl;

    logprint(s.str());
    
    s << str;
    
    return s.str();
}
