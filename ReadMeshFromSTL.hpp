#include "submodules/STL_Reader/stl_reader.h"

namespace BAEMM
{
    template<typename Real, typename Int>
    bool ReadMeshFromSTL(
        const std::filesystem::path & file,
        mref<Tensor2<Real,Int>> coords,
        mref<Tensor2<Int, Int>> simplices
    )
    {
        std::string tag = "ReadMeshFromSTL";
        
        ptic(tag);
        
        print("Reading mesh from file " + file.string() + ".");
        
        std::vector<float>        coords_;
        std::vector<float>        normals_;
        std::vector<unsigned int> simplices_;
        std::vector<unsigned int> solids_;
        
        
        try {
            stl_reader::ReadStlFile(
                file.string().c_str(), coords_, normals_, simplices_, solids_
            );
        }
        catch (std::exception & e)
        {
            std::cout << e.what() << std::endl;
            
            ptoc(tag);
            
            return false;
        }
        
        coords    = Tensor2<Real,Int>( coords_.data(),    static_cast<Int>(coords_.size()/3),     Int(3) );
        simplices = Tensor2<Int, Int>( simplices_.data(), static_cast<Int>(simplices_.size()/3) , Int(3) );
        
        ptoc(tag);
        
        return true;
    }
    
} // namespace BAEMM

