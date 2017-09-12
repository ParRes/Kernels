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
!          - Converted to CAF by Alessandro Fanfarillo, February 2016.
!          - Small fixes for OpenCoarrays `stop 1` issue work around by
!            Izaak "Zaak" Beekman, March 2017
! *************************************************************************

function prk_get_wtime() result(t)
  use iso_fortran_env
  implicit none
  real(kind=REAL64) ::  t
  integer(kind=INT64) :: c, r
  call system_clock(count = c, count_rate = r)
  t = real(c,REAL64) / real(r,REAL64)
end function prk_get_wtime

subroutine initialize_w(is_star,r,W)
  use iso_fortran_env
  implicit none
  logical, intent(in) :: is_star
  integer(kind=INT32), intent(in) :: r
  real(kind=REAL64), intent(inout) :: W(-r:r,-r:r)
  integer(kind=INT32) :: ii, jj
  ! fill the stencil weights to reflect a discrete divergence operator
  W = 0.0d0
  if (is_star) then
    do ii=1,r
      W(0, ii) =  1.0d0/real(2*ii*r,REAL64)
      W(0,-ii) = -1.0d0/real(2*ii*r,REAL64)
      W( ii,0) =  1.0d0/real(2*ii*r,REAL64)
      W(-ii,0) = -1.0d0/real(2*ii*r,REAL64)
    enddo
  else
    ! Jeff: check that this is correct with the new W indexing
    do jj=1,r
      do ii=-jj+1,jj-1
        W( ii, jj) =  1.0d0/real(4*jj*(2*jj-1)*r,REAL64)
        W( ii,-jj) = -1.0d0/real(4*jj*(2*jj-1)*r,REAL64)
        W( jj, ii) =  1.0d0/real(4*jj*(2*jj-1)*r,REAL64)
        W(-jj, ii) = -1.0d0/real(4*jj*(2*jj-1)*r,REAL64)
      enddo
      W( jj, jj)  =  1.0d0/real(4*jj*r,REAL64)
      W(-jj,-jj)  = -1.0d0/real(4*jj*r,REAL64)
    enddo
  endif
end subroutine initialize_w

subroutine apply_stencil(is_star,tiling,tile_size,r,n,W,A,B)
  use iso_fortran_env
  implicit none
  logical, intent(in) :: is_star, tiling
  integer(kind=INT32), intent(in) :: tile_size, r, n
  real(kind=REAL64), intent(in) :: W(-r:r,-r:r)
  real(kind=REAL64), intent(in) :: A(n,n)
  real(kind=REAL64), intent(inout) :: B(n,n)
  integer(kind=INT32) :: i, j, ii, jj, it, jt
  if (is_star) then
    if (.not.tiling) then
      !$omp do
      do j=r,n-r-1
        do i=r,n-r-1
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
        enddo
      enddo
      !$omp end do
    else ! tiling
      !$omp do
      do jt=r,n-r-1,tile_size
        do it=r,n-r-1,tile_size
          do j=jt,min(n-r-1,jt+tile_size-1)
            do i=it,min(n-r-1,it+tile_size-1)
              do jj=-r,r
                B(i+1,j+1) = B(i+1,j+1) + W(0,jj) * A(i+1,j+jj+1)
              enddo
              do ii=-r,-1
                B(i+1,j+1) = B(i+1,j+1) + W(ii,0) * A(i+ii+1,j+1)
              enddo
              do ii=1,r
                B(i+1,j+1) = B(i+1,j+1) + W(ii,0) * A(i+ii+1,j+1)
              enddo
            enddo
          enddo
        enddo
      enddo
      !$omp end do
    endif ! tiling
  else ! grid
    if (.not.tiling) then
      !$omp do
      do j=r,n-r-1
        do i=r,n-r-1
          do jj=-r,r
            do ii=-r,r
              B(i+1,j+1) = B(i+1,j+1) + W(ii,jj) * A(i+ii+1,j+jj+1)
            enddo
          enddo
        enddo
      enddo
      !$omp end do
    else ! tiling
      !$omp do
      do jt=r,n-r-1,tile_size
        do it=r,n-r-1,tile_size
          do j=jt,min(n-r-1,jt+tile_size-1)
            do i=it,min(n-r-1,it+tile_size-1)
              do jj=-r,r
                do ii=-r,r
                  B(i+1,j+1) = B(i+1,j+1) + W(ii,jj) * A(i+ii+1,j+jj+1)
                enddo
              enddo
            enddo
          enddo
        enddo
      enddo
      !$omp end do
    endif ! tiling
  endif ! star
end subroutine apply_stencil

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
  integer(kind=INT32),allocatable :: nc[:,:],nr[:,:]    ! local grid dimension
  integer(kind=INT32) ::  stencil_size                  ! number of points in stencil
  integer(kind=INT32) ::  tile_size                     ! loop nest block factor
  logical :: tiling                                     ! boolean indication loop nest blocking
  logical :: is_star                                    ! true = star, false = grid
  integer(kind=INT32), parameter :: r=RADIUS            ! radius of stencil
  real(kind=REAL64) :: W(-r:r,-r:r)                     ! weights of points in the stencil
  real(kind=REAL64), allocatable :: A(:,:)[:,:], B(:,:) ! grid values
  real(kind=REAL64), parameter :: cx=1.d0, cy=1.d0
  ! runtime variables
  integer(kind=INT32) :: i, j, k
  integer(kind=INT32) :: ii, jj, it, jt
  integer(kind=INT64) :: flops                          ! floating point ops per iteration
  real(kind=REAL64) :: norm[*], reference_norm          ! L1 norm of solution
  integer(kind=INT64) :: active_points                  ! interior of grid with respect to stencil
  real(kind=REAL64) :: t0, t1, stencil_time, avgtime    ! timing parameters
  real(kind=REAL64), parameter ::  epsilon=1.D-8        ! error tolerance
  integer :: me,np,dims(2),coords(2),start_i,end_i,start_j,end_j,nc_g,nr_g
  integer :: nr_t,nc_t,nc_b,nc_l,nr_l,nr_r
  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  np = num_images()
  me = this_image()
  if (me == 1) then
    write(*,'(a25)') 'Parallel Research Kernels'
    write(*,'(a44)') 'Fortran coarray stencil execution on 2D grid'
  endif

  if (command_argument_count().lt.2) then
    if (me == 1) then
      write(*,'(a17,i1)') 'argument count = ', command_argument_count()
      write(*,'(a32,a29)') 'Usage: ./stencil <# iterations> ', &
                        '<array dimension> [tile_size]'
    endif
    stop 1
  endif

  iterations = 1
  call get_command_argument(1,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') iterations
  if (iterations .lt. 1) then
    if (me == 1) then
      write(*,'(a,i5)') 'ERROR: iterations must be >= 1 : ', iterations
    endif
    stop 1
  endif

  n = 1
  call get_command_argument(2,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') n
  if (n .lt. 1) then
    if (me == 1) then
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

  ! TODO: parse runtime input for star/grid
#ifdef STAR
  is_star = .true.
#else
  is_star = .false.
#endif

  ! TODO: parse runtime input for radius

  if (r .lt. 1) then
    if (me == 1) then
      write(*,'(a,i5,a)') 'ERROR: Stencil radius ',r,' should be positive'
    endif
    stop 1
  else if ((2*r+1) .gt. n) then
    if (me == 1) then
      write(*,'(a,i5,a,i5)') 'ERROR: Stencil radius ',r,&
                             ' exceeds grid size ',n
    endif
    stop 1
  endif

!  Collectives are part of Fortran 2015
!  call co_broadcast(n,source_image=1)
!  call co_broadcast(iterations,source_image=1)

  dims(1) = int(sqrt(real(np)))
  dims(2) = int(np/dims(1))

  i=1
  do while(dims(1)*dims(2) /= np)
     dims(1) = int(sqrt(real(np)))+i
     dims(2) = int(np/dims(1))
     if(i<0) then
        i = (i*(-1)) + 1
     else
        i = i * (-1)
     endif
  enddo

  allocate(nr[dims(1),*],nc[dims(1),*])

  nr = n/dims(1)
  nc = n/dims(2)

  coords = this_image(nr)

  nr_g = nr; nc_g = nc

  if(coords(1) <= modulo(n,nr)) nr = nr + 1
  if(coords(2) <= modulo(n,nc)) nc = nc + 1

  if(modulo(n,nr) > 0) nr_g = nr_g + 1
  if(modulo(n,nc) > 0) nc_g = nc_g + 1

  allocate( A(1-r:nr_g+r,1-r:nc_g+r)[dims(1),*], stat=err)
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of A returned ',err
    stop 1
  endif

  allocate( B(1:nr_g,1:nc_g), stat=err )
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of B returned ',err
    stop 1
  endif

  norm = 0.d0
  active_points = int(n-2*r,INT64)**2

  if (me == 1) then
    write(*,'(a,i8)') 'Number of images     = ',num_images()
    write(*,'(a,i8)') 'Number of iterations = ', iterations
    write(*,'(a,i8)') 'Grid size            = ', n
    write(*,'(a,i8)') 'Radius of stencil    = ', r
    if (is_star) then
      write(*,'(a,a)')  'Type of stencil      = star'
      stencil_size = 4*r+1
    else
      write(*,'(a,a)')  'Type of stencil      = grid'
      stencil_size = (2*r+1)**2
    endif
    !write(*,'(a)') 'Data type            = double precision'
    !write(*,'(a)') 'Compact representation of stencil loop body'
    if (tiling) then
      write(*,'(a,i5)') 'Tile size            = ', tile_size
    else
      write(*,'(a)') 'Untiled'
    endif
  endif

  call initialize_w(is_star,r,W)

  ! Getting the remote size of the upper and left images
  ! in order to initialize correctly the local grid A.
  nr_g = 0; nc_g = 0
  do k=1,coords(1)-1
     nr_g = nr_g + nr[k,coords(2)]
  enddo
  do k=1,coords(2)-1
     nc_g = nc_g + nc[coords(1),k]
  enddo

  ! intialize the input and output arrays
  do j=1,nc
    do i=1,nr
      A(i,j) = cx*(i+nr_g)+cy*(j+nc_g)
    enddo
  enddo

  do j=1,nc
    do i=1,nr
      B(i,j) = 0.d0
    enddo
  enddo

  start_i = 0
  start_j = 0
  end_i = nr - 1
  end_j = nc - 1

  if(coords(1) == 1)       start_i = r
  if(coords(1) == dims(1)) end_i = nr - r - 1
  if(coords(2) == 1)       start_j = r
  if(coords(2) == dims(2)) end_j = nc - r - 1

  ! top remote dimensions
  if(coords(1)>1) then
     nr_t = nr[coords(1)-1,coords(2)]
     nc_t = nc[coords(1)-1,coords(2)]
  endif

  ! bottom remote dimension
  if(coords(1)<dims(1)) nc_b = nc[coords(1)+1,coords(2)]

  ! left remote dimensions
  if(coords(2)>1) then
     nr_l = nr[coords(1),coords(2)-1]
     nc_l = nc[coords(1),coords(2)-1]
  endif

  !right remote dimension
  if(coords(2)<dims(2)) nr_r = nr[coords(1),coords(2)+1]

  t0 = 0

  sync all

  do k=0,iterations

     ! exchanging data in y-direction
     !top
     if(coords(1)>1) then
        a(1-r:0,1:nc_t) = a(nr_t-r+1:nr_t,1:nc_t)[coords(1)-1,coords(2)]
     endif
     !bottom
     if(coords(1)<dims(1)) then
        a(nr+1:nr+r,1:nc_b) = a(1:r,1:nc_b)[coords(1)+1,coords(2)]
     endif
     sync all
     !exchanging data in x-direction
     !left
     if(coords(2)>1) then
        a(1-r:nr_l+r,1-r:0) = a(1-r:nr_l+r,nc_l-r+1:nc_l)[coords(1),coords(2)-1]
     endif
     !right
     if(coords(2)<dims(2)) then
        a(1-r:nr_r+r,nc+1:nc+r) = a(1-r:nr_r+r,1:r)[coords(1),coords(2)+1]
     endif

     sync all

    ! start timer after a warmup iteration
    if (k.eq.1) then
       t0 = prk_get_wtime()
       sync all
    endif

    ! Apply the stencil operator
    if (.not.tiling) then
      do j=start_j,end_j
        do i=start_i,end_i
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
    do j=1,nc
      do i=1,nr
        A(i,j) = A(i,j) + 1.d0
      enddo
    enddo

    sync all

  enddo ! iterations

  t1 = prk_get_wtime()

  sync all

  ! compute L1 norm

  start_i = 1
  start_j = 1
  end_i = nr
  end_j = nc

  if(coords(1) == 1)       start_i = r
  if(coords(2) == 1)       start_j = r
  if(coords(1) == dims(1)) end_i = nr - r
  if(coords(2) == dims(2)) end_j = nc - r

  do j=start_j,end_j
    do i=start_i,end_i
      norm = norm + abs(B(i,j))
    enddo
  enddo

  stencil_time = t1 - t0
!  Collectives are part of Fortran 2015
!  call co_sum(norm,result_image=1)
  sync all
  if(me == 1) then
     do i=2,np
        norm = norm + norm[i]
     enddo
     norm = norm / real(active_points,REAL64)
  endif

  !******************************************************************************
  !* Analyze and output results.
  !******************************************************************************

  deallocate( B )
  deallocate( A )

  ! Jeff: valgrind says that this is branching on uninitialized value (norm),
  !       but this is nonsense since norm is initialized to 0 at line 167.

  ! verify correctness
  if(me == 1) then
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
  endif

end program main
