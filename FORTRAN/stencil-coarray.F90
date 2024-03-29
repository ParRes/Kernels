!
! Copyright (c) 2013, Intel Corporation
! Copyright (c) 2021, NVIDIA
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

subroutine apply_stencil(is_star,tiling,tile_size,r,n,W,A,B)
  use, intrinsic :: iso_fortran_env
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
  use, intrinsic :: iso_fortran_env
  use prk
  implicit none
  integer :: err
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

    call prk_get_arguments('stencil',iterations=iterations,order=n,tile_size=tile_size)

  endif

  call co_broadcast(iterations,source_image=1)
  call co_broadcast(n,source_image=1)
  call co_broadcast(tile_size,source_image=1)

  ! TODO: parse runtime input for star/grid
#ifdef STAR
  is_star = .true.
#else
  is_star = .false.
#endif

  tiling = (tile_size.ne.n)

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
    if (tiling) then
      write(*,'(a,i5)') 'Tile size            = ', tile_size
    else
      write(*,'(a)') 'Untiled'
    endif
  endif

  ! ********************************************************************
  ! ** Allocate space for the input and perform the computation
  ! ********************************************************************

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

  allocate( A(1-r:nr_g+r,1-r:nc_g+r)[dims(1),*], B(1:nr_g,1:nc_g), stat=err)
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation returned ',err
    stop 1
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
  stencil_time = t1 - t0

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

  norm = 0.d0
  do j=start_j,end_j
    do i=start_i,end_i
      norm = norm + abs(B(i,j))
    enddo
  enddo

  active_points = int(n-2*r,INT64)**2
  call co_sum(norm,result_image=1)
  norm = norm / real(active_points,REAL64)

  !******************************************************************************
  !* Analyze and output results.
  !******************************************************************************

  deallocate( A,B )

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
