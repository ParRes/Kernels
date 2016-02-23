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
  integer :: me, npes
  logical :: printer
  ! for argument parsing
  integer :: err
  integer :: arglen
  character(len=32) :: argtmp
  ! problem definition
  integer(kind=INT32) :: iterations                     ! number of times to run the pipeline algorithm
  integer(kind=INT32) ::  n                             ! linear grid dimension
  integer(kind=INT32) ::  stencil_size                  ! number of points in stencil
  logical ::  tiling                                    ! boolean indication loop nest blocking
  integer(kind=INT32) ::  tile_size                     ! loop nest block factor
  integer(kind=INT32), parameter :: r=RADIUS            ! radius of stencil
  real(kind=REAL64) :: W(-r:r,-r:r)                     ! weights of points in the stencil
  real(kind=REAL64), allocatable :: A(:,:), B(:,:)      ! grid values
  real(kind=REAL64), parameter :: cx=1.0, cy=1.0
  ! runtime variables
  integer(kind=INT32) :: i, j, k
  integer(kind=INT32) :: ii, jj, it, jt
  integer(kind=INT64) :: flops                          ! floating point ops per iteration
  real(kind=REAL64) :: norm, reference_norm             ! L1 norm of solution
  integer(kind=INT64) :: active_points                  ! interior of grid with respect to stencil
  real(kind=REAL64) :: t0, t1, stencil_time, avgtime    ! timing parameters
  real(kind=REAL64), parameter ::  epsilon=1.D-8        ! error tolerance

  me   = this_image()-1 ! use 0-based indexing of PEs
  npes = num_images()
  printer = (me.eq.0)

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

#ifndef PRKVERSION
#warning Your common/make.defs is missing PRKVERSION
#define PRKVERSION "N/A"
#endif
  if (printer) then
    write(*,'(a,a)') 'Parallel Research Kernels version ', PRKVERSION
    write(*,'(a)')   'Fortran coarray stencil execution on 2D grid'
  endif

  if (command_argument_count().lt.2) then
    if (printer) then
      write(*,'(a,i1)') 'argument count = ', command_argument_count()
      write(*,'(a,a)')  'Usage: ./stencil <# iterations> ',             &
                        '<array dimension> [tile_size]'
    endif
    stop 1
  endif

  iterations = 1
  call get_command_argument(1,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') iterations
  if (iterations .lt. 1) then
    if (printer) then
      write(*,'(a,i5)') 'ERROR: iterations must be >= 1 : ', iterations
    endif
    stop 1
  endif

  n = 1
  call get_command_argument(2,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') n
  if (n .lt. 1) then
    if (printer) then
      write(*,'(a,i5)') 'ERROR: array dimension must be >= 1 : ', n
    endif
    stop 1
  endif

  tiling    = .false.
  tile_size = 0
  if (command_argument_count().gt.2) then
    call get_command_argument(3,argtmp,arglen,err)
    if (err.eq.0) read(argtmp,'(i32)') tile_size
    if ((tile_size .lt. 1).or.(tile_size.gt.n)) then
      write(*,'(a,i5,a,i5)') 'WARNING: tile_size ',tile_size,&
                             ' must be >= 1 and <= ',n
    else
      tiling = .true.
    endif
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
    write(6,'(a20,i3,a10,i5)') 'allocation of A returned ',err,' at image ',me
    stop 1
  endif

  allocate( B(n,n), stat=err )
  if (err .ne. 0) then
    write(6,'(a20,i3,a10,i5)') 'allocation of B returned ',err,' at image ',me
    stop 1
  endif

  norm = 0.0
  active_points = int(n-2*r,INT64)**2

  if (printer) then
    write(6,'(a23,i8)') 'Number of images     = ', num_images()
    write(*,'(a,i8)') 'Grid size            = ', n
    write(*,'(a,i8)') 'Radius of stencil    = ', r
    write(*,'(a,a)')  'Type of stencil      = ', &
#ifdef STAR
                      'star'
#else
                      'stencil'
#endif
    write(*,'(a)') 'Data type            = double precision'
    write(*,'(a)') 'Compact representation of stencil loop body'
    if (tiling) then
        write(*,'(a,i5)') 'Tile size            = ', tile_size
    else
        write(*,'(a)') 'Untiled'
    endif
    write(*,'(a,i8)') 'Number of iterations = ', iterations
  endif
#ifdef STAR
  stencil_size = 4*r+1
#else
  stencil_size = (2*r+1)**2
#endif

  ! fill the stencil weights to reflect a discrete divergence operator
  W = 0
#ifdef STAR
  do ii=1,r
    W(0, ii) =  1/real(2*ii*r,REAL64)
    W(0,-ii) = -1/real(2*ii*r,REAL64)
    W( ii,0) =  1/real(2*ii*r,REAL64)
    W(-ii,0) = -1/real(2*ii*r,REAL64)
  enddo
#else
  ! Jeff: check that this is correct with the new W indexing
  do jj=1,r
    do ii=-jj+1,jj-1
      W( ii, jj) =  1/real(4*jj*(2*jj-1)*r,REAL64)
      W( ii,-jj) = -1/real(4*jj*(2*jj-1)*r,REAL64)
      W( jj, ii) =  1/real(4*jj*(2*jj-1)*r,REAL64)
      W(-jj, ii) = -1/real(4*jj*(2*jj-1)*r,REAL64)
    enddo
    W( jj, jj)  =  1/real(4*jj*r,REAL64)
    W(-jj,-jj)  = -1/real(4*jj*r,REAL64)
  enddo
#endif

  ! intialize the input and output arrays
  do j=1,n
    do i=1,n
      A(i,j) = cx*i+cy*j
    enddo
  enddo
  B = 0

  do k=0,iterations

    ! start timer after a warmup iteration
    if (k.eq.1) then
       sync all
       t0 = prk_get_wtime()
    endif

    ! Apply the stencil operator
    if (.not.tiling) then
      do j=r,n-r-1
        do i=r,n-r-1
#ifdef STAR
            ! do not use Intel Fortran unroll directive here (slows down)
            do jj=-r,r
              B(i+1,j+1) = B(i+1,j+1) + W(0,jj) * A(i+1,j+jj+1)
            enddo
            do ii=-r,-1
              B(i+1,j+1) = B(i+1,j+1) + W(ii,0) * A(i+ii+1,j+1)
            enddo
            do ii=1,r
              B(i+1,j+1) = B(i+1,j+1) + W(ii,0) * A(i+ii+1,j+1)
            enddo
#else
            do jj=-r,r
              do ii=-r,r
                B(i+1,j+1) = B(i+1,j+1) + W(ii,jj) * A(i+ii+1,j+jj+1)
              enddo
            enddo
#endif
        enddo
      enddo
    else ! tiling
      do jt=r,n-r-1,tile_size
        do it=r,n-r-1,tile_size
          do j=jt,min(n-r-1,jt+tile_size-1)
            do i=it,min(n-r-1,it+tile_size-1)
#ifdef STAR
                do jj=-r,r
                  B(i+1,j+1) = B(i+1,j+1) + W(0,jj) * A(i+1,j+jj+1)
                enddo
                do ii=-r,-1
                  B(i+1,j+1) = B(i+1,j+1) + W(ii,0) * A(i+ii+1,j+1)
                enddo
                do ii=1,r
                  B(i+1,j+1) = B(i+1,j+1) + W(ii,0) * A(i+ii+1,j+1)
                enddo
#else
                do jj=-r,r
                  do ii=-r,r
                    B(i+1,j+1) = B(i+1,j+1) + W(ii,jj) * A(i+ii+1,j+jj+1)
                  enddo
                enddo
#endif
            enddo
          enddo
        enddo
      enddo
    endif ! tiling

    ! add constant to solution to force refresh of neighbor data, if any
    A = A + 1

  enddo ! iterations

  sync all
  t1 = prk_get_wtime()

  ! compute L1 norm in parallel
  do j=r,n-r
    do i=r,n-r
      norm = norm + abs(B(i,j))
    enddo
  enddo

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
