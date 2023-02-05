public:
    
    void Initialize_Metal()
    {
        tic(ClassName()+"::Initialize_Metal");
        
        const uint size  =     simplex_count * sizeof(Real);
        const uint size4 = 4 * simplex_count * sizeof(Real);
        
        areas      = device->newBuffer(size,  MTL::ResourceStorageModeManaged);
        mid_points = device->newBuffer(size4, MTL::ResourceStorageModeManaged);
        normals    = device->newBuffer(size4, MTL::ResourceStorageModeManaged);
        
        areas_ptr      = static_cast<Real *>(     areas->contents());
        mid_points_ptr = static_cast<Real *>(mid_points->contents());
        normals_ptr    = static_cast<Real *>(   normals->contents());
        
        command_queue = device->newCommandQueue();
        
        if( command_queue == nullptr )
        {
            std::cout << "Failed to find the command queue." << std::endl;
            return;
        }
        
        toc(ClassName()+"::Initialize_Metal");
    }
