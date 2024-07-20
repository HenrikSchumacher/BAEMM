#pragma once

#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <QuartzCore/QuartzCore.hpp>

namespace HeavyMetal
{

    template< uint rows_per_panel_, uint cols_per_panel_, typename Scalar_, typename Int_>
    class MatrixPanelRowMajor
    {
    public:
        
        using Scalar = Scalar_;
        using Int    = Int_;
        
        static constexpr Int panel_dim [2] = { rows_per_panel_, cols_per_panel_};
        
        static constexpr Int panel_size = panel_dim[0] * panel_dim[1];
        
        MatrixPanelRowMajor() = default;
        
        MatrixPanelRowMajor( Int row_count_, Int col_count_ )
        :   dim { row_count_, col_count_ }
        ,   panel_count {
                CeilDivide(dim[0], panel_dim[0]),
                CeilDivide(dim[1], panel_dim[1])
            }
        ,   dim_internal { panel_count[0] * panel_dim[0], panel_count[1] * panel_dim[1] }
        {
            safe_alloc( buffer, dim_internal[0] * dim_internal[1] );
        }
        
        ~MatrixPanelRowMajor()
        {
            safe_free( buffer );
        }
        
    protected:
        
        Int dim          [2] = {};
        Int panel_count  [2] = {};
        Int dim_internal [2] = {};
                            
        Scalar * restrict buffer = nullptr;
        

        public:

        force_inline Int Dimension( Int i  ) const
        {
            return dim[i];
        }
        
        
        force_inline mptr<Scalar> data()
        {
            return buffer;
        }
        
        force_inline cptr<Scalar> data() const
        {
            return buffer;
        }
        
        
        
        
        force_inline Int PanelIndex( Int p_i, Int p_j  ) const
        {
            return panel_count[1] * p_i + p_j;
        }
        
        force_inline mptr<Scalar> Panel( Int p_i, Int p_j )
        {
            // Returns pointer to upper left entry in panel {p_i, p_j}
            return &buffer[ panel_size * PanelIndex( p_i, p_j ) ];
        }
        
        force_inline cptr<Scalar> Panel( Int p_i, Int p_j ) const
        {
            // Returns pointer to upper left entry in panel {p_i, p_j}
            return &buffer[ panel_size * PanelIndex( p_i, p_j ) ];
        }
        
        
        template<bool rowcol>
        force_inline Int PanelPosition( Int p_k ) const
        {
            if constexpr ( rowcol == 0 ) // 0 means row
            {
                return p_k / panel_count[1];
            }
            else
            {
                return p_k % panel_count[1];
            }
        }
        
        
        force_inline Scalar & operator()( Int p_i, Int p_j, Int l_i, Int l_j )
        {
            return Panel(p_i,p_j)[ panel_dim[1] * l_i + l_j ];
        }
        
        force_inline const Scalar & operator()( Int p_i, Int p_j, Int l_i, Int l_j ) const
        {
            
            return Panel(p_i,p_j)[ panel_dim[1] * l_i + l_j ];
        }

        force_inline Scalar & operator()( Int g_i, Int g_j )
        {
            const Int p_i = g_i / panel_dim[0];
            const Int l_i = g_i % panel_dim[0];
            const Int p_j = g_j / panel_dim[1];
            const Int l_j = g_j % panel_dim[1];
            
            return operator()(p_i,p_j,l_i,l_j);
        }
        
        force_inline const Scalar & operator()( Int g_i, Int g_j ) const
        {
            const Int p_i = g_i / panel_dim[0];
            const Int l_i = g_i % panel_dim[0];
            const Int p_j = g_j / panel_dim[1];
            const Int l_j = g_j % panel_dim[1];
            
            return operator()(p_i,p_j,l_i,l_j);
        }
        
        
        void ToRowMajor( mptr<Scalar> B, Int CPU_thread_count ) const
        {
            // Slower than necessary, but should do it for now.
            const Int d_0 = dim[0];
            const Int d_1 = dim[1];

            #pragma omp parallel for num_threads(CPU_thread_count) schedule(static)
            for( Int i = 0; i < d_0; ++i )
            {
                for( Int j = 0; j < d_1; ++j )
                {
                    B[ d_1 * i + j] = A(i,j);
                }
            }
        }
        
        void FromRowMajor( mptr<Scalar> B, Int CPU_thread_count )
        {
            // Slower than necessary, but should do it for now.
            const Int d_0 = dim[0];
            const Int d_1 = dim[1];

            #pragma omp parallel for num_threads(CPU_thread_count) schedule(static)
            for( Int i = 0; i < d_0; ++i )
            {
                for( Int j = 0; j < d_1; ++j )
                {
                    this->operator()(i,j) = B[ d_1 * i + j];
                }
            }
        }
        
        void ToPanelRowMajor( mptr<Scalar> B, Int CPU_thread_count ) const
        {
            const Int d_0 = dim[0];
            const Int d_1 = dim[1];

            #pragma omp parallel for num_threads(CPU_thread_count) schedule(static)
            for( Int i = 0; i < d_0; ++i )
            {
                copy_buffer( &buffer[dim_internal[1]*i], &B[d_1 * i], d_1 );
            }
        }
        
        void FromPanelRowMajor( mptr<Scalar> B, Int CPU_thread_count )
        {
            const Int d_0 = dim[0];
            const Int d_1 = dim[1];

            #pragma omp parallel for num_threads(CPU_thread_count) schedule(static)
            for( Int i = 0; i < d_0; ++i )
            {
                copy_buffer( &B[d_1 * i], &buffer[dim_internal[1]*i], d_1 );
            }
        }
        
        
    }; // class MatrixPanelRowMajor
    
//    template< uint m, uint n, typename Scalar, typename Int>
//    void ToTensor( const MatrixPanelRowMajor<m,n,Scalar,Int> & A, Tensor2<Scalar,Int> & B )
//    {
//        // Slower than necessary, but should do it for now.
//        const Int d_0 = A.Dimensions(0);
//        const Int d_1 = A.Dimensions(1);
//
//        if( (B.Dimension(0) != d_0) || (B.Dimension(1) != d_1) )
//        {
//            B = Tensor2<Scalar,Int>( d_0, d_1 );
//        }
//
//        for( Int i = 0; i < d_0; ++i )
//        {
//            for( Int j = 0; j < d_1; ++j )
//            {
//                B(i,j) = A(i,j);
//            }
//        }
//    }
//
//    template< uint m, uint n, typename Scalar, typename Int>
//    void ToMatrixPanelRowMajor(
//        const Tensor2<Scalar,Int> & A,
//              MatrixPanelRowMajor<m,n,Scalar,Int> & B
//    )
//    {
//        // Slower than necessary, but should do it for now.
//        const Int d_0 = A.Dimensions(0);
//        const Int d_1 = A.Dimensions(1);
//
//        if( (B.Dimension(0) != d_0) || (B.Dimension(1) != d_1) )
//        {
//            B = MatrixPanelRowMajor<m,n,Scalar,Int>( d_0, d_1 );
//        }
//
//        for( Int i = 0; i < d_0; ++i )
//        {
//            for( Int j = 0; j < d_1; ++j )
//            {
//                B(i,j) = A(i,j);
//            }
//        }
//    }
    
} // namespace HeavyMetal
