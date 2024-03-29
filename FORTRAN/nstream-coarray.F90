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
! NAME:    nstream
!
! PURPOSE: To compute memory bandwidth when adding a vector of a given
!          number of double precision values to the scalar multiple of
!          another vector of the same length, and storing the result in
!          a third vector.
!
! USAGE:   The program takes as input the number
!          of iterations to loop over the triad vectors, the length of the
!          vectors, and the offset between vectors
!
!          <progname> <# iterations> <vector length> <offset>
!
!          The output consists of diagnostics to make sure the
!          algorithm worked, and of timing statistics.
!
! NOTES:   Bandwidth is determined as the number of words read, plus the
!          number of words written, times the size of the words, divided
!          by the execution time. For a vector length of N, the total
!          number of words read and written is 4*N*sizeof(double).
!
!
! HISTORY: This code is loosely based on the Stream benchmark by John
!          McCalpin, but does not follow all the Stream rules. Hence,
!          reported results should not be associated with Stream in
!          external publications
!
!          Converted to C++11 by Jeff Hammond, November May 2017.
!
! *******************************************************************

program main
  use, intrinsic :: iso_fortran_env
  use prk
  implicit none
  integer :: me, np, p
  integer :: err
  ! problem definition
  integer(kind=INT32) :: iterations
  integer(kind=INT64) :: length, offset
  integer(kind=INT32) :: co_iterations[*]
  integer(kind=INT64) :: co_length[*]
  real(kind=REAL64), allocatable ::  A(:)[:]
  real(kind=REAL64), allocatable ::  B(:)[:]
  real(kind=REAL64), allocatable ::  C(:)[:]
  real(kind=REAL64) :: scalar
  integer(kind=INT64) :: bytes
  ! runtime variables
  integer(kind=INT64) :: i
  integer(kind=INT32) :: k
  real(kind=REAL64) ::  asum, ar, br, cr
  real(kind=REAL64) ::  co_asum[*]
  real(kind=REAL64) ::  t0, t1, nstream_time, avgtime
  real(kind=REAL64), parameter ::  epsilon=1.D-8

  me = this_image()
  np = num_images()

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  if (me.eq.1) then
    write(*,'(a25)') 'Parallel Research Kernels'
    write(*,'(a48)') 'Fortran coarray STREAM triad: A = B + scalar * C'

    call prk_get_arguments('nstream',iterations=iterations,length=length,offset=offset)

    write(*,'(a23,i12)') 'Number of images     = ', np
    write(*,'(a23,i12)') 'Number of iterations = ', iterations
    write(*,'(a23,i12)') 'Vector length        = ', length
    write(*,'(a23,i12)') 'Offset               = ', offset

    ! co_broadcast is 2018 and not available in all coarray implementations
    do p=1,np
      co_iterations[p] = iterations
      co_length[p]     = length
    enddo
  endif
  sync all
  if (me.ne.1) then
    ! copy broadcast inputs to local variables
    iterations = co_iterations[this_image()]
    length     = co_length[this_image()]
  endif

  ! ********************************************************************
  ! ** Allocate space and perform the computation
  ! ********************************************************************

  allocate( A(length)[*], B(length)[*], C(length)[*], stat=err)
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation returned ',err
    error stop 1
  endif

  do concurrent (i=1:length)
    A(i) = 0
    B(i) = 2
    C(i) = 2
  enddo
  sync all

  scalar = 3

  t0 = 0

  do k=0,iterations
    if (k.eq.1) then
      sync all ! barrier
      t0 = prk_get_wtime()
    endif

    do concurrent (i=1:length)
      A(i) = A(i) + B(i) + scalar * C(i)
    enddo
  enddo ! iterations

  sync all
  t1 = prk_get_wtime()

  nstream_time = t1 - t0

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  ar  = 0
  br  = 2
  cr  = 2
  do k=0,iterations
      ar = ar + br + scalar * cr;
  enddo

  asum = 0
  do concurrent (i=1:length)
    asum = asum + abs(A(i)-ar)
  enddo

  ! reduction via gather
  co_asum[me] = asum
  sync all
  asum = 0
  if (me.eq.1) then
    do p=1,np
      asum = asum + co_asum[p]
    enddo
  endif

  deallocate( A,B,C )

  if (abs(asum) .gt. epsilon) then
    if (me.eq.1) then
      write(*,'(a35)') 'Failed Validation on output array'
      write(*,'(a30,f30.15)') '       Expected value: ', ar
      write(*,'(a30,f30.15)') '       Observed value: ', A(1)
      write(*,'(a35)')  'ERROR: solution did not validate'
      error stop 1
    endif
  else
    if (me.eq.1) write(*,'(a17)') 'Solution validates'
    avgtime = nstream_time/iterations;
    bytes = 4 * np * length * storage_size(A)/8
    if (me.eq.1) then
      write(*,'(a12,f15.3,1x,a12,e15.6)')    &
              'Rate (MB/s): ', 1.d-6*bytes/avgtime, &
              'Avg time (s): ', avgtime
    endif
  endif

end program main

