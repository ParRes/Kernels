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

program main
  use, intrinsic :: iso_fortran_env
  use omp_lib
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
  integer(kind=INT32) :: i,j,k,q
  real(kind=REAL64) ::  checksum, reference, residuum
  real(kind=REAL64) ::  t0, t1, dgemm_time, avgtime ! timing parameters
  real(kind=REAL64), parameter ::  epsilon=1.0d-8    ! error tolerance

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  write(*,'(a25)') 'Parallel Research Kernels'
  write(*,'(a70)') 'Fortran OpenMP TARGET Dense matrix-matrix multiplication: C += A x B'

  call prk_get_arguments('dgemm',iterations=iterations,order=order,tile_size=tile_size)

  write(*,'(a22,i8)') 'Number of threads    = ', omp_get_max_threads()
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

  !$omp parallel do
  do i=1, order
    A(:,i) = real(i-1,REAL64)
    B(:,i) = real(i-1,REAL64)
    C(:,i) = real(0,REAL64)
  enddo
  !$omp end parallel do

  !$omp target data map(to: A,B) map(tofrom: C) map(to:order)

  t0 = 0

  do q=0,iterations
    if (q.eq.1) t0 = omp_get_wtime()
    !$omp target teams distribute parallel do simd collapse(2) schedule(static,1)
    do j=1,order
      do i=1,order
        do k=1,order
          C(i,j) = C(i,j) + A(i,k) * B(k,j)
        enddo
      enddo
    enddo
    !$omp end target teams distribute parallel do simd
  enddo

  t1 = omp_get_wtime()

  !$omp end target data

  dgemm_time = t1 - t0

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  forder = real(order,REAL64)
  reference = 0.25d0 * forder**3 * (forder-1)**2 * (iterations+1)
  checksum = 0.0d0
  !$omp parallel do simd reduction(+:checksum)
  do j=1,order
    do i=1,order
      checksum = checksum + C(i,j)
    enddo
  enddo
  !$omp end parallel do simd

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

