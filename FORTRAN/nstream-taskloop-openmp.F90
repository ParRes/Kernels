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
  use iso_fortran_env
  use omp_lib
  use prk
  implicit none
  integer :: err
  ! problem definition
  integer(kind=INT32) :: iterations
  integer(kind=INT64) :: length, offset
  real(kind=REAL64), allocatable ::  A(:)
  real(kind=REAL64), allocatable ::  B(:)
  real(kind=REAL64), allocatable ::  C(:)
  real(kind=REAL64) :: scalar
  integer(kind=INT64) :: bytes
  ! runtime variables
  integer(kind=INT64) :: i
  integer(kind=INT32) :: k
  real(kind=REAL64) ::  asum, ar, br, cr
  real(kind=REAL64) ::  t0, t1, nstream_time, avgtime
  real(kind=REAL64), parameter ::  epsilon=1.D-8

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  write(*,'(a25)') 'Parallel Research Kernels'
  write(*,'(a47)') 'Fortran OpenMP TASKLOOP STREAM triad: A = B + scalar * C'

  call prk_get_arguments('nstream',iterations=iterations,length=length,offset=offset)

  write(*,'(a23,i12)') 'Number of threads    = ', omp_get_max_threads()
  write(*,'(a23,i12)') 'Number of iterations = ', iterations
  write(*,'(a23,i12)') 'Vector length        = ', length
  write(*,'(a23,i12)') 'Offset               = ', offset

  ! ********************************************************************
  ! ** Allocate space and perform the computation
  ! ********************************************************************

  allocate( A(length), B(length), C(length), stat=err)
  if (err .ne. 0) then
    write(*,'(a20,i3)') 'allocation returned ',err
    stop 1
  endif

  scalar = 3

  !$omp parallel default(none)                    &
  !$omp&  shared(A,B,C,scalar,nstream_time)       &
  !$omp&  firstprivate(length,iterations,offset)  &
  !$omp&  private(i,k,t0,t1)
  !$omp master

  !$omp taskloop firstprivate(length,offset) shared(A,B,C) private(i)
  do i=1,length
    A(i) = 0
    B(i) = 2
    C(i) = 2
  enddo
  !$omp end taskloop

  t0 = 0

  !$omp taskwait

  do k=0,iterations
    if (k.eq.1) then
      t0 = omp_get_wtime()
    endif

    !$omp taskloop firstprivate(length,offset) shared(A,B,C) private(i)
    do i=1,length
      A(i) = A(i) + B(i) + scalar * C(i)
    enddo
    !$omp end taskloop

    !$omp taskwait

  enddo ! iterations

  t1 = omp_get_wtime()
  nstream_time = t1 - t0


  !$omp end master
  !$omp end parallel

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
  !$omp parallel do reduction(+:asum)
  do i=1,length
    asum = asum + abs(A(i)-ar)
  enddo
  !$omp end parallel do

  deallocate( A,B,C )

  if (abs(asum) .gt. epsilon) then
    write(*,'(a35)') 'Failed Validation on output array'
    write(*,'(a30,f30.15)') '       Expected value: ', ar
    write(*,'(a30,f30.15)') '       Observed value: ', A(1)
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

