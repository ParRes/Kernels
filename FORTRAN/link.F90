program main
  use iso_fortran_env
#ifdef _OPENMP
  use omp_lib
#endif
  implicit none
  integer :: err
  integer(kind=INT32) ::  order
  real(kind=REAL64), allocatable ::  A(:,:)
  real(kind=REAL64), allocatable ::  B(:,:)
  real(kind=REAL64), allocatable ::  C(:,:)
  real(kind=REAL64) :: alpha, beta
  integer(kind=INT32) :: i,j,k

  order = 100

  allocate( A(order,order), stat=err)
  if (err .ne. 0) stop 1

  allocate( B(order,order), stat=err )
  if (err .ne. 0) stop 1

  allocate( C(order,order), stat=err )
  if (err .ne. 0) stop 1

  do i=1, order
    A(:,i) = real(i-1,REAL64)
    B(:,i) = real(i-1,REAL64)
    C(:,i) = real(0,REAL64)
  enddo

  alpha = 1.0d0
  beta  = 1.0d0
  call dgemm('N', 'N',              &
             order, order, order,   &
             alpha, A, order,       &
                    B, order,       &
             beta,  C, order)

  deallocate( A )
  deallocate( B )
  deallocate( C )

end program main

