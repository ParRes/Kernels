!
! Copyright (c) 2017, Intel Corporation
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
  integer(kind=INT32) ::  iterations, offset
  integer(kind=INT64) ::  length
  real(kind=REAL64), allocatable ::  A(:)
  real(kind=REAL64), allocatable ::  B(:)
  real(kind=REAL64), allocatable ::  C(:)
  real(kind=REAL64) :: scalar
  integer(kind=INT64) :: bytes
  ! runtime variables
  integer(kind=INT32) :: k
  real(kind=REAL64) ::  asum, ar, br, cr, ref
  real(kind=REAL64) ::  t0, t1, nstream_time, avgtime
  real(kind=REAL64), parameter ::  epsilon=1.D-8

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  write(*,'(a25)') 'Parallel Research Kernels'
  write(*,'(a47)') 'Fortran Pretty STREAM triad: A = B + scalar * C'

  if (command_argument_count().lt.2) then
    write(*,'(a17,i1)') 'argument count = ', command_argument_count()
    write(*,'(a62)')    'Usage: ./transpose <# iterations> <vector length> [<offset>]'
    stop 1
  endif

  iterations = 1
  call get_command_argument(1,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') iterations
  if (iterations .lt. 1) then
    write(*,'(a,i5)') 'ERROR: iterations must be >= 1 : ', iterations
    stop 1
  endif

  length = 1
  call get_command_argument(2,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') length
  if (length .lt. 1) then
    write(*,'(a,i5)') 'ERROR: length must be nonnegative : ', length
    stop 1
  endif

  offset = 0
  if (command_argument_count().gt.2) then
    call get_command_argument(3,argtmp,arglen,err)
    if (err.eq.0) read(argtmp,'(i32)') offset
    if (offset .lt. 0) then
      write(*,'(a,i5)') 'ERROR: offset must be positive : ', offset
      stop 1
    endif
  endif

  write(*,'(a,i12)') 'Number of iterations = ', iterations
  write(*,'(a,i12)') 'Vector length        = ', length
  write(*,'(a,i12)') 'Offset               = ', offset

  ! ********************************************************************
  ! ** Allocate space for the input and transpose matrix
  ! ********************************************************************

  allocate( A(length), stat=err)
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of A returned ',err
    stop 1
  endif

  allocate( B(length), stat=err )
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of B returned ',err
    stop 1
  endif

  allocate( C(length), stat=err )
  if (err .ne. 0) then
    write(*,'(a,i3)') 'allocation of C returned ',err
    stop 1
  endif

  A = 0
  B = 2
  C = 2

  scalar = 3

  t0 = 0

  do k=0,iterations
    ! start timer after a warmup iteration
    if (k.eq.1) t0 = prk_get_wtime()
    A = A + B + scalar * C
  enddo ! iterations

  t1 = prk_get_wtime()
  nstream_time = t1 - t0

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  ar  = 0
  br  = 2
  cr  = 2
  ref = 0
  do k=0,iterations
      ar = ar + br + scalar * cr;
  enddo

  ar = ar * length

  asum = sum(abs(A))

  deallocate( C )
  deallocate( B )
  deallocate( A )

  if (abs(asum-ar) .gt. epsilon) then
    write(*,'(a35)') 'Failed Validation on output array'
    write(*,'(a30,f30.15)') '       Expected checksum: ', ar
    write(*,'(a30,f30.15)') '       Observed checksum: ', asum
    write(*,'(a35)')  'ERROR: solution did not validate'
    stop 1
  else
    write(*,'(a17)') 'Solution validates'
    avgtime = nstream_time/iterations;
    bytes = 4 * int(length,INT64) * storage_size(A)/8
    write(*,'(a12,f15.3,1x,a12,e15.6)')    &
        'Rate (MB/s): ', 1.d-6*bytes/avgtime, &
        'Avg time (s): ', avgtime
  endif

end program main

