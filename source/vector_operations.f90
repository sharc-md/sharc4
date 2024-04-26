module vector_operations
    implicit none
    
    public :: cross_product
    public :: norm
    public :: z3gram_schmidt_vec1_on_vec2    
contains

    ! Function to calculate the cross product of two 3D vectors
    function cross_product(vec1, vec2) result(cross)
        complex(8), allocatable, dimension(:) :: vec1, vec2, cross

        allocate(cross(3))
        
        cross(1) = vec1(2) * vec2(3) - vec1(3) * vec2(2)
        cross(2) = vec1(3) * vec2(1) - vec1(1) * vec2(3)
        cross(3) = vec1(1) * vec2(2) - vec1(2) * vec2(1)
    end function cross_product

    function norm(vec1) result(norm_vec1)
        complex(8), dimension(:) :: vec1
        complex(8), allocatable, dimension(:) :: norm_vec1
        allocate(norm_vec1(3))

        norm_vec1 = vec1/sqrt(sum(conjg(vec1)*vec1))
    end function norm

    function z3gram_schmidt_vec1_on_vec2(vec1, vec2) result(new_vec1)
        complex(8), dimension(:) :: vec1, vec2
        complex(8), allocatable, dimension(:) :: f, new_vec1
        
        allocate(f(3))
        allocate(new_vec1(3))
        
        f = sum(conjg(vec2)*vec1) / sum(conjg(vec2)*vec2)
        new_vec1 = vec1 - f * vec2
        new_vec1 = new_vec1 / sqrt(sum(conjg(new_vec1)*new_vec1))
    end function z3gram_schmidt_vec1_on_vec2

    
end module vector_operations
