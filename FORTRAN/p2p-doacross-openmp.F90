!
! Copyright (c) 2015, Intel Corporation
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
! NAME:    Pipeline
!
! PURPOSE: This program tests the efficiency with which point-to-point
!          synchronization can be carried out. It does so by executing
!          a pipelined algorithm on an m*n grid. The first array dimension
!          is distributed among the threads (stripwise decomposition).
!
! USAGE:   The program takes as input the
!          dimensions of the grid, and the number of iterations on the grid
!
!                <progname> <iterations> <m> <n>
!
!          The output consists of diagnostics to make sure the
!          algorithm worked, and of timing statistics.
!
! FUNCTIONS CALLED:
!
!          Other than standard C functions, the following
!          functions are used in this program:
!
! HISTORY: - Written by Rob Van der Wijngaart, February 2009.
!            Converted to Fortran by Jeff Hammond, January 2016.
! *******************************************************************

program main
  use, intrinsic :: iso_fortran_env
  use omp_lib
  use prk
  implicit none
  integer :: err
  ! problem definition
  integer(kind=INT32) :: iterations                     ! number of times to run the pipeline algorithm
  integer(kind=INT32) :: m, n
  real(kind=REAL64) :: corner_val                       ! verification value at top right corner of grid
  real(kind=REAL64), allocatable :: grid(:,:)           ! array holding grid values
  ! runtime variables
  integer(kind=INT32) :: i, j, k
  real(kind=REAL64) ::  t0, t1, pipeline_time, avgtime  ! timing parameters
  real(kind=REAL64), parameter ::  epsilon=1.D-8        ! error tolerance

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  write(*,'(a25)') 'Parallel Research Kernels'
  write(*,'(a44)') 'Fortran OpenMP pipeline execution on 2D grid'

  call prk_get_arguments('p2p',iterations=iterations,dimx=m,dimy=n,tilex=mc,tiley=nc)

  write(*,'(a27,i8)')    'Number of threads        = ', omp_get_max_threads()
  write(*,'(a27,i8)')    'Number of iterations     = ', iterations
  write(*,'(a27,i8,i8)') 'Grid sizes               = ', m, n

  allocate( grid(m,n), stat=err)
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of grid returned ',err
    stop 1
  endif

  !$omp parallel default(none)                                  &
  !$omp&  shared(grid,t0,t1,iterations,pipeline_time)           &
  !$omp&  firstprivate(m,n)                                     &
  !$omp&  private(i,j,k,corner_val)

  !$omp do collapse(2)
  do j=1,n
    do i=1,m
      grid(i,j) = 0.0d0
    enddo
  enddo
  !$omp end do
  ! it is debatable whether these loops should be parallel
  !$omp do
  do j=1,n
    grid(1,j) = real(j-1,REAL64)
  enddo
  !$omp end do
  !$omp do
  do i=1,m
    grid(i,1) = real(i-1,REAL64)
  enddo
  !$omp end do

  do k=0,iterations

    !  start timer after a warmup iteration
    if (k.eq.1) then
      !$omp barrier
      !$omp master
      t0 = omp_get_wtime()
      !$omp end master
    endif

    !$omp do ordered(2) collapse(2)
    do j=2,n
      do i=2,m
        !$omp ordered depend(sink:j,i-1) depend(sink:j-1,i) depend(sink:j-1,i-1)
        grid(i,j) = grid(i-1,j) + grid(i,j-1) - grid(i-1,j-1)
        !$omp ordered depend(source)
      enddo
    enddo
    !$omp end do

    !$omp master
    grid(1,1) = -grid(m,n)
    !$omp end master

    !$omp barrier

  enddo ! iterations

  !$omp barrier
  !$omp master
  t1 = omp_get_wtime()
  pipeline_time = t1 - t0
  !$omp end master

  !$omp end parallel

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  ! verify correctness, using top right value
  corner_val = real((iterations+1)*(n+m-2),REAL64);
  if (abs(grid(m,n)-corner_val)/corner_val .gt. epsilon) then
    write(*,'(a,f10.2,a,f10.2)') 'ERROR: checksum ',grid(m,n), &
            ' does not match verification value ', corner_val
    stop 1
  endif

  write(*,'(a)') 'Solution validates'
  avgtime = pipeline_time/iterations
  write(*,'(a,f13.6,a,f10.6)') 'Rate (MFlop/s): ',2.d-6*real((m-1)*(n-1),INT64)/avgtime, &
         ' Avg time (s): ', avgtime

  deallocate( grid )

end program
