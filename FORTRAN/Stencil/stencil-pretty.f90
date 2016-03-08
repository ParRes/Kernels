!
! Copyright (c) 2013, Intel Corporation
!
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions
! are met:
!
! * Redistributions of source code must retain the above copyright
!       notice, this list of conditions and the following disclaimer.
! * Redistributions in binary form must reproduce the above
!       copyright notice, this list of conditions and the following
!       disclaimer in the documentation and/or other materials provided
!       with the distribution.
! * Neither the name of Intel Corporation nor the names of its
!       contributors may be used to endorse or promote products
!       derived from this software without specific prior written
!       permission.
!
! THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
! "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, ACLUDAG, BUT NOT
! LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
! FOR A PARTICULAR PURPOSE ARE DISCLAIMED. A NO EVENT SHALL THE
! COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, ADIRECT,
! ACIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (ACLUDAG,
! BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
! LOSS OF USE, DATA, OR PROFITS; OR BUSAESS ATERRUPTION) HOWEVER
! CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER A CONTRACT, STRICT
! LIABILITY, OR TORT (ACLUDAG NEGLIGENCE OR OTHERWISE) ARISAG A
! ANY WAY B OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
! POSSIBILITY OF SUCH DAMAGE.
!
!
! *******************************************************************
!
! NAME:    Stencil
!
! PURPOSE: This program tests the efficiency with which a space-invariant,
!          linear, symmetric filter (stencil) can be applied to a square
!          grid or image.
!
! USAGE:   The program takes as input the linear
!          dimension of the grid, and the number of iterations on the grid
!
!                <progname> <iterations> <grid size>
!
!          The output consists of diagnostics to make sure the
!          algorithm worked, and of timing statistics.
!
! FUNCTIONS CALLED:
!
!          Other than standard C functions, the following functions are used in
!          this program:
!          wtime()
!
! HISTORY: - Written by Rob Van der Wijngaart, February 2009.
!          - RvdW: Removed unrolling pragmas for clarity;
!            added constant to array "in" at end of each iteration to force
!            refreshing of neighbor data in parallel versions; August 2013
!          - Converted to Fortran by Jeff Hammond, January-February 2016.
!
! *******************************************************************

function prk_get_wtime() result(t)
  use iso_fortran_env
  real(kind=REAL64) ::  t
  integer(kind=INT64) :: c, r
  call system_clock(count = c, count_rate = r)
  t = real(c,REAL64) / real(r,REAL64)
end function prk_get_wtime

program main
  use iso_fortran_env
  implicit none
  real(kind=REAL64) :: prk_get_wtime
  ! for argument parsing
  integer :: err
  integer :: arglen
  character(len=32) :: argtmp
  ! problem definition
  integer(kind=INT32) :: iterations                     ! number of times to run the pipeline algorithm
  integer(kind=INT32) ::  n                             ! linear grid dimension
  integer(kind=INT32) ::  stencil_size                  ! number of points in stencil
  integer(kind=INT32), parameter :: r=RADIUS            ! radius of stencil
  real(kind=REAL64) :: W(-r:r,-r:r)                     ! weights of points in the stencil
  real(kind=REAL64), allocatable :: A(:,:), B(:,:)      ! grid values
  real(kind=REAL64), parameter :: cx=1.0, cy=1.0
  ! runtime variables
  integer(kind=INT32) :: i, j, k
  integer(kind=INT64) :: flops                          ! floating point ops per iteration
  real(kind=REAL64) :: norm, reference_norm             ! L1 norm of solution
  integer(kind=INT64) :: active_points                  ! interior of grid with respect to stencil
  real(kind=REAL64) :: t0, t1, stencil_time, avgtime    ! timing parameters
  real(kind=REAL64), parameter ::  epsilon=1.D-8        ! error tolerance

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

#ifndef PRKVERSION
#warning Your common/make.defs is missing PRKVERSION
#define PRKVERSION "N/A"
#endif
  write(*,'(a,a)') 'Parallel Research Kernels version ', PRKVERSION
  write(*,'(a)')   'Pretty stencil execution on 2D grid'

  if (command_argument_count().lt.2) then
    write(*,'(a,i1)') 'argument count = ', command_argument_count()
    write(*,'(a,a)')  'Usage: ./stencil <# iterations> ',             &
                      '<array dimension>'
    stop 1
  endif

  iterations = 1
  call get_command_argument(1,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') iterations
  if (iterations .lt. 1) then
    write(*,'(a,i5)') 'ERROR: iterations must be >= 1 : ', iterations
    stop 1
  endif

  n = 1
  call get_command_argument(2,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') n
  if (n .lt. 1) then
    write(*,'(a,i5)') 'ERROR: array dimension must be >= 1 : ', n
    stop 1
  endif

  if (r .lt. 1) then
    write(*,'(a,i5,a)') 'ERROR: Stencil radius ',r,' should be positive'
    stop 1
  endif

  if ((2*r+1) .gt. n) then
    write(*,'(a,i5,a,i5)') 'ERROR: Stencil radius ',r,&
                           ' exceeds grid size ',n
    stop 1
  endif

  allocate( A(n,n), stat=err)
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of A returned ',err
    stop 1
  endif

  allocate( B(n,n), stat=err )
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of B returned ',err
    stop 1
  endif

  norm = 0
  active_points = int(n-2*r,INT64)**2

  write(*,'(a,i8)') 'Grid size            = ', n
  write(*,'(a,i8)') 'Radius of stencil    = ', r
  write(*,'(a,a)')  'Type of stencil      = ', &
#ifdef STAR
                   'star'
  stencil_size = 4*r+1
#else
                   'stencil'
  stencil_size = (2*r+1)**2
#endif
  write(*,'(a)') 'Data type            = double precision'
  write(*,'(a)') 'Compact representation of stencil loop body'
  write(*,'(a,i8)') 'Number of iterations = ', iterations

  ! fill the stencil weights to reflect a discrete divergence operator
  W = 0
#ifdef STAR
  do i=1,r
    W(0, i) =  1/real(2*i*r,REAL64)
    W(0,-i) = -1/real(2*i*r,REAL64)
    W( i,0) =  1/real(2*i*r,REAL64)
    W(-i,0) = -1/real(2*i*r,REAL64)
  enddo
#else
  ! Jeff: check that this is correct with the new W indexing
  do j=1,r
    do i=-j+1,j-1
      W( i, j) =  1/real(4*j*(2*j-1)*r,REAL64)
      W( i,-j) = -1/real(4*j*(2*j-1)*r,REAL64)
      W( j, i) =  1/real(4*j*(2*j-1)*r,REAL64)
      W(-j, i) = -1/real(4*j*(2*j-1)*r,REAL64)
    enddo
    W( j, j)  =  1/real(4*j*r,REAL64)
    W(-j,-j)  = -1/real(4*j*r,REAL64)
  enddo
#endif

  ! intialize the input and output arrays
  do concurrent (i=1:n, j=1:n)
    A(i,j) = cx*i+cy*j
  enddo
  !B(r+1:n-r,r+1:n-r) = 0 ! minimal
  B = 0 ! sufficient

  do k=0,iterations

    ! start timer after a warmup iteration
    if (k.eq.1) t0 = prk_get_wtime()

    ! Apply the stencil operator
#ifdef STAR
    B(r+1:n-r,r+1:n-r) = B(r+1:n-r,r+1:n-r) + W(0,0) * A(r+1:n-r,r+1:n-r)
    do j=1,r
      B(r+1:n-r,r+1:n-r) = B(r+1:n-r,r+1:n-r)                    &
                         + W( 0,-j) * A(r+1  :n-r,  r-j+1:n-r-j) &
                         + W( 0, j) * A(r+1  :n-r,  r+j+1:n-r+j) &
                         + W(-j, 0) * A(r-j+1:n-r-j,r+1  :n-r  ) &
                         + W( j, 0) * A(r+j+1:n-r+j,r+1  :n-r  )
    enddo
#else
    do j=-r,r
      do i=-r,r
        B(r+1:n-r,r+1:n-r) = B(r+1:n-r,r+1:n-r) + W(i,j) * A(r+i+1:n-r+i,r+j+1:n-r+j)
      enddo
    enddo
#endif

    ! add constant to solution to force refresh of neighbor data, if any
    A = A + 1

  enddo ! iterations

  t1 = prk_get_wtime()

  norm = sum(sum(abs(B(r+1:n-r,r+1:n-r)),1))

  stencil_time = t1 - t0
  norm = norm / real(active_points,REAL64)

  !******************************************************************************
  !* Analyze and output results.
  !******************************************************************************

  deallocate( B )
  deallocate( A )

  ! Jeff: valgrind says that this is branching on uninitialized value (norm),
  !       but this is nonsense since norm is initialized to 0.0 at line 167.

  ! verify correctness
  reference_norm = real(iterations+1,REAL64) * (cx + cy);
  if (abs(norm-reference_norm) .gt. epsilon) then
    write(*,'(a,f13.6,a,f13.6)') 'ERROR: L1 norm = ', norm, &
                                 ' Reference L1 norm = ', reference_norm
  else
    write(*,'(a)') 'Solution validates'
#ifdef VERBOSE
    write(*,'(a,f13.6,a,f13.6)') 'VERBOSE: L1 norm = ', norm, &
                                 ' Reference L1 norm = ', reference_norm
#endif
  endif

  flops = int(2*stencil_size+1,INT64) * active_points
  avgtime = stencil_time/iterations
  write(*,'(a,f13.6,a,f13.6)') 'Rate (MFlops/s): ',1.0d-6*flops/avgtime, &
                               ' Avg time (s): ',avgtime

  stop

end program main
