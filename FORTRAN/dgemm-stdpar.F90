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

function prk_get_wtime() result(t)
  use iso_fortran_env
  implicit none
  real(kind=REAL64) ::  t
  integer(kind=INT64) :: c, r
  call system_clock(count = c, count_rate = r)
  t = real(c,REAL64) / real(r,REAL64)
end function prk_get_wtime

subroutine prk_dgemm(order, tile_size, A, B, C)
  use iso_fortran_env
  implicit none
  integer(kind=INT32), intent(in) :: order, tile_size
  real(kind=REAL64), intent(in) ::  A(order,order)
  real(kind=REAL64), intent(in) ::  B(order,order)
  real(kind=REAL64), intent(inout) ::  C(order,order)
  integer(kind=INT32) :: i,j,k,it,jt,kt

  if (tile_size.lt.order) then
    do concurrent (jt=1:order:tile_size)
      do concurrent (kt=1:order:tile_size)
        do concurrent (it=1:order:tile_size)
          do j=jt,min(order,jt+tile_size-1)
            do k=kt,min(order,kt+tile_size-1)
              do i=it,min(order,it+tile_size-1)
                C(i,j) = C(i,j) + A(i,k) * B(k,j)
              enddo
            enddo
          enddo
        enddo
      enddo
    enddo
  else
    do concurrent (j=1:order)
      do concurrent (k=1:order)
        do concurrent (i=1:order)
          C(i,j) = C(i,j) + A(i,k) * B(k,j)
        enddo
      enddo
    enddo
  endif
end subroutine prk_dgemm

program main
  use iso_fortran_env
  implicit none
  real(kind=REAL64) :: prk_get_wtime
  ! for argument parsing
  integer :: err
  integer :: arglen
  character(len=32) :: argtmp
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
  write(*,'(a61)') 'Fortran STDPAR Dense matrix-matrix multiplication: C += A x B'

  if (command_argument_count().lt.2) then
    write(*,'(a17,i1)') 'argument count = ', command_argument_count()
    write(*,'(a66)')    'Usage: ./dgemm-pretty <# iterations> <matrix order> [<tile_size>]'
    stop 1
  endif

  iterations = 1
  call get_command_argument(1,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') iterations
  if (iterations .lt. 1) then
    write(*,'(a,i5)') 'ERROR: iterations must be >= 1 : ', iterations
    stop 1
  endif

  order = 1
  call get_command_argument(2,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') order
  if (order .lt. 1) then
    write(*,'(a,i5)') 'ERROR: order must be >= 1 : ', order
    stop 1
  endif

  tile_size = 32
  if (command_argument_count().gt.2) then
      call get_command_argument(3,argtmp,arglen,err)
      if (err.eq.0) read(argtmp,'(i32)') tile_size
  endif
  if ((tile_size.lt.1).or.(tile_size.gt.order)) then
    write(*,'(a20,i5,a22,i5)') 'WARNING: tile_size ',tile_size, &
                               ' must be >= 1 and <= ',order
    tile_size = order ! no tiling
  endif

  write(*,'(a,i8)') 'Number of iterations = ', iterations
  write(*,'(a,i8)') 'Matrix order         = ', order
  write(*,'(a,i8)') 'Tile size            = ', tile_size

  ! ********************************************************************
  ! ** Allocate space for the input and output matrices
  ! ********************************************************************

  allocate( A(order,order), stat=err)
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of A returned ',err
    stop 1
  endif

  allocate( B(order,order), stat=err )
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of B returned ',err
    stop 1
  endif

  allocate( C(order,order), stat=err )
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of C returned ',err
    stop 1
  endif

  do concurrent (i=1:order)
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

  deallocate( A )
  deallocate( B )

  forder = real(order,REAL64)
  reference = 0.25d0 * forder**3 * (forder-1)**2 * (iterations+1)
  checksum = 0.0d0
  do j=1,order
    do i=1,order
      checksum = checksum + C(i,j)
    enddo
  enddo

  deallocate( C )

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

