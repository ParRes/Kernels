!
! Copyright (c) 2017, Intel Corporation
! Copyright (c) 2021, NVIDIA
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions
! are met:
!
! * Redistributions of source code must retain the above copyright
!      notice, this list of conditions and the following disclaimer.
! * Redistributions in binary form must reproduce the above
!      copyright notice, this list of conditions and the following
!      disclaimer in the documentation and/or other materials provided
!      with the distribution.
! * Neither the name of Intel Corporation nor the names of its
!      contributors may be used to endorse or promote products
!      derived from this software without specific prior written
!      permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
! "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
! LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
! FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
! COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
! INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
! BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
! LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
! LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
! ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
! POSSIBILITY OF SUCH DAMAGE.

!*******************************************************************
!
!  NAME:    dgemm
!
!  PURPOSE: This program tests the efficiency with which a dense matrix
!           dense multiplication is carried out
!
!  USAGE:   The program takes as input the matrix order and
!           the number of times the matrix-matrix multiplication
!           is carried out.
!
!           <progname> <# iterations> <matrix order>
!
!           The output consists of diagnostics to make sure the
!           algorithm worked, and of timing statistics.
!
!  HISTORY: Written by Rob Van der Wijngaart, February 2009.
!           Converted to C++11 by Jeff Hammond, December, 2017.
!           Converted to Fortran by Jeff Hammond, December, 2017.
!
! *******************************************************************

subroutine prk_dgemm(order, tile_size, A, B, C)
  use, intrinsic :: iso_fortran_env
  implicit none
  integer(kind=INT32), intent(in) :: order, tile_size
  real(kind=REAL64), intent(in) ::  A(order,order)
  real(kind=REAL64), intent(in) ::  B(order,order)
  real(kind=REAL64), intent(inout) ::  C(order,order)
  !real(kind=REAL64) :: TA(tile_size,tile_size)
  real(kind=REAL64) :: TB(tile_size,tile_size)
  !real(kind=REAL64) :: TTB(tile_size,tile_size)
  !real(kind=REAL64) :: TC(tile_size,tile_size)
  integer(kind=INT32) :: i,j,k,it,jt,kt

  if (tile_size.lt.order) then
    do jt=1,order,tile_size
      do kt=1,order,tile_size
        TB(1:tile_size,1:tile_size) = B(kt:kt+tile_size,jt:jt+tile_size)
        !TTB = transpose(TB)
        !TTB = transpose(B(kt:kt+tile_size,jt:jt+tile_size))
        do it=1,order,tile_size
          !TA(1:tile_size,1:tile_size) = A(it:it+tile_size,kt:kt+tile_size)
          !!TC = C(it:it+tile_size,jt:jt+tile_size)
          do j=jt,min(order,jt+tile_size-1)
            do k=kt,min(order,kt+tile_size-1)
              do i=it,min(order,it+tile_size-1)
                !C(i,j) = C(i,j) + A(i,k) * B(k,j) ! original
                C(i,j) = C(i,j) + A(i,k) * TB(1+k-kt,1+j-jt) ! before TTB
                !C(i,j) = C(i,j) + A(i,k) * TTB(1+j-jt,1+k-kt) ! after TT
                !C(i,j) = C(i,j) + TA(1+i-it,1+k-kt) * TB(1+k-kt,1+j-jt) ! with TA
                !!TC(1+i-it,1+j-jt) = TC(1+i-it,1+j-jt) + TA(1+i-it,1+k-kt) * TB(1+k-kt,1+j-jt) ! with TA and TB
              enddo
            enddo
          enddo
          !!C(it:it+tile_size,jt:jt+tile_size) = C(it:it+tile_size,jt:jt+tile_size) + TC
          !!C(it:it+tile_size,jt:jt+tile_size) = TC
        enddo
      enddo
    enddo
  else
    do j=1,order
      do k=1,order
        do i=1,order
          C(i,j) = C(i,j) + A(i,k) * B(k,j)
        enddo
      enddo
    enddo
  endif
end subroutine prk_dgemm

program main
  use, intrinsic :: iso_fortran_env
  use prk
  implicit none
  integer :: err
  ! problem definition
  integer(kind=INT32) ::  iterations                ! number of times to do the kernel
  integer(kind=INT32) ::  order                     ! order of the matrix
  integer(kind=INT32) ::  tile_size
  real(kind=REAL64) :: forder                       ! order as a double
  real(kind=REAL64), allocatable ::  A(:,:)         ! buffer to hold input matrix
  real(kind=REAL64), allocatable ::  B(:,:)         ! buffer to hold input matrix
  real(kind=REAL64), allocatable ::  C(:,:)         ! buffer to hold output matrix
  integer(kind=INT64) :: nflops
  ! runtime variables
  integer(kind=INT32) :: i,j,k
  real(kind=REAL64) ::  checksum, reference, residuum
  real(kind=REAL64) ::  t0, t1, dgemm_time, avgtime ! timing parameters
  real(kind=REAL64), parameter ::  epsilon=1.0d-8   ! error tolerance

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  write(*,'(a25)') 'Parallel Research Kernels'
  write(*,'(a61)') 'Fortran Serial Dense matrix-matrix multiplication: C += A x B'

  call prk_get_arguments('dgemm',iterations=iterations,order=order,tile_size=tile_size)

  write(*,'(a22,i8)') 'Number of iterations = ', iterations
  write(*,'(a22,i8)') 'Matrix order         = ', order
  if (tile_size.ne.order) then
    write(*,'(a22,i8)') 'Tile size            = ', tile_size
  else
    write(*,'(a10)') 'Tiling off'
  endif

  ! ********************************************************************
  ! ** Allocate space for the input and output matrices
  ! ********************************************************************

  allocate( A(order,order), B(order,order), C(order,order), stat=err)
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation  returned ',err
    stop 1
  endif

  do i=1, order
    A(:,i) = real(i-1,REAL64)
    B(:,i) = real(i-1,REAL64)
    C(:,i) = real(0,REAL64)
  enddo

  t0 = 0

  do k=0,iterations
    if (k.eq.1) then
      t0 = prk_get_wtime()
    endif
    call prk_dgemm(order, tile_size, A, B, C)
  enddo

  t1 = prk_get_wtime()

  dgemm_time = t1 - t0

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  forder = real(order,REAL64)
  reference = 0.25d0 * forder**3 * (forder-1)**2 * (iterations+1)
  checksum = 0.0d0
  do j=1,order
    do i=1,order
      checksum = checksum + C(i,j)
    enddo
  enddo

  deallocate( A,B,C )

  residuum = abs(checksum-reference)/reference
  if (residuum .lt. epsilon) then
    write(*,'(a)') 'Solution validates'
    avgtime = dgemm_time/iterations
    nflops = 2 * int(order,INT64)**3
    write(*,'(a,f13.3,a,f10.6)') 'Rate (MF/s): ',(1.d-6*nflops)/avgtime, &
           ' Avg time (s): ', avgtime
  else
    write(*,'(a,e30.15)') 'Reference checksum = ', reference
    write(*,'(a,e30.15)') 'Actual checksum    = ', checksum
    stop 1
  endif

end program main

