!
! Copyright (c) 2015, Intel Corporation
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
!          - Converted to Coarray Fortran
!            by Alessandro Fanfarillo and Jeff Hammond, February 2016.
!          - Minor bug fixes by Izaak "Zaak" Beekman, March 2017
! ********************************************************************

function prk_get_wtime() result(t)
  use iso_fortran_env
  implicit none
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
  integer(kind=INT32) :: m, n, m_local, max_m_local
  real(kind=REAL64) :: corner_val                       ! verification value at top right corner of grid
  real(kind=REAL64), allocatable :: grid(:,:)[:]           ! array holding grid values
  ! runtime variables
  integer(kind=INT32) ::  i, j, k
  integer ::  me, np, prev, next !, stat
  real(kind=REAL64) ::  t0, t1, pipeline_time, avgtime  ! timing parameters
  real(kind=REAL64), parameter ::  epsilon=1.D-8        ! error tolerance

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  ! WARNING: this code uses base-1 indexing of images - others use base-0.
  me = this_image()
  np = num_images()

  ! co_broadcast is part of Fortran 2015, so we will not assume it yet.
  if(me == 1) then
     write(*,'(a25)') 'Parallel Research Kernels'
     write(*,'(a45)') 'Fortran coarray pipeline execution on 2D grid'
  endif

  if (command_argument_count().lt.3) then
     if(me == 1) then
       write(*,'(a17,i1)') 'argument count = ', command_argument_count()
       write(*,'(a34,a38)')  'Usage: ./synch_p2p <# iterations> ',  &
                             '<array x-dimension> <array y-dimension>'
     endif
     stop 1
  endif

  iterations = 1
  call get_command_argument(1,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') iterations

  m = 1
  call get_command_argument(2,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') m

  n = 1
  call get_command_argument(3,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') n

  if (iterations .lt. 1) then
     write(*,'(a,i5)') 'ERROR: iterations must be >= 1 : ', iterations
     stop 1
  endif

  if ((m .lt. 1).or.(n .lt. 1)) then
     write(*,'(a,i5,i5)') 'ERROR: array dimensions must be >= 1 : ', m, n
     stop 1
  endif

  ! co_max is part of Fortran 2015, so we will not assume it. This is present
  ! in OpenCoarrays and has been for a while, when used with GFortran >= 6.
  ! Instead, we will just allocate more than necessary in some cases.
  m_local = int(m/np)
  !if((me-1) < mod(m,np)) m_local = m_local + 1
  max_m_local = m_local + 1
  !call co_max(max_m_local)

  allocate( grid(max_m_local,n)[*], stat=err)

  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of grid returned ',err
    stop 1
  endif

  if(me == 1) then
     write(*,'(a,i8)')    'Number of threads        = ', num_images()
     write(*,'(a,i8)')    'Number of iterations     = ', iterations
     write(*,'(a,i8,i8)') 'Grid sizes               = ', m, n
  endif

  do j=1,n
    do i=1,m_local
      grid(i,j) = 0.0d0
    enddo
  enddo

  if(me == 1) then
     do j=1,n
        grid(1,j) = real(j-1,REAL64)
     enddo
     do i=1,m_local
        grid(i,1) = real(i-1,REAL64)
     enddo
  endif

  prev = me - 1
  next = me + 1

  t0 = 0

  do k=0,iterations

    !  start timer after a warmup iteration
    if (k.eq.1) then
       sync all
       t0 = prk_get_wtime()
    endif

    do j=2,n
       if(me > 1) sync images(prev)
       do i=2,m_local
          grid(i,j) = grid(i-1,j) + grid(i,j-1) - grid(i-1,j-1)
       enddo
       if(me /= np) then
          grid(1,j)[next] = grid(m_local,j)
          sync images(next)
       endif
    enddo
    ! copy top right corner value to bottom left corner to create dependency; we
    ! need a barrier to make sure the latest value is used. This also guarantees
    ! that the flags for the next iteration (if any) are not getting clobbered
   if(me == np) then
      corner_val = -grid(m_local,n)
      grid(1,1)[1] = corner_val
      sync images(1)
   else if(me == 1) then
      sync images(np)
   endif

  enddo ! iterations

  sync all

  t1 = prk_get_wtime()

  pipeline_time = t1 - t0

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  ! verify correctness, using top right value
  corner_val = real((iterations+1)*(n+m_local-2),REAL64);
  if(me == np) then
     if (abs(grid(m_local,n)-corner_val)/corner_val .gt. epsilon) then
        write(*,'(a,f10.2,a,f10.2)') 'ERROR: checksum ',grid(m_local,n), &
             ' does not match verification value ', corner_val
        stop 1
     endif
     write(*,'(a)') 'Solution validates'

     avgtime = pipeline_time/iterations
     write(*,'(a,f13.6,a,f10.6)') 'Rate (MFlop/s): ',2.d-6*real((m-1)*(n-1),INT64)/avgtime, &
          ' Avg time (s): ', avgtime
  endif

  sync all

  deallocate( grid )

end program
