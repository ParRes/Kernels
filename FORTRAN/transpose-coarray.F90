!
! Copyright (c) 2015, Intel Corporation
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
! NAME:    transpose
!
! PURPOSE: This program measures the time for the transpose of a
!          column-major stored matrix into a row-major stored matrix.
!
! USAGE:   Program input is the matrix order and the number of times to
!          repeat the operation:
!
!          transpose <matrix_size> <# iterations> [tile size]
!
!          An optional parameter specifies the tile size used to divide the
!          individual matrix blocks for improved cache and TLB performance.
!
!          The output consists of diagnostics to make sure the
!          transpose worked and timing statistics.
!
! HISTORY: Written by  Rob Van der Wijngaart, February 2009.
!          Converted to Fortran by Jeff Hammond, January 2015
!          Small fixes to work around OpenCoarrays `stop` issue by
!          Izaak "Zaak" Beekman
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
  integer :: me, npes
  logical :: printer
  ! for argument parsing
  integer :: err
  integer :: arglen
  character(len=32) :: argtmp
  ! problem definition
  integer(kind=INT32) ::  iterations                ! number of times to do the transpose
  integer(kind=INT32) ::  order                     ! order of a the matrix
  real(kind=REAL64), allocatable ::  A(:,:)[:]      ! buffer to hold original matrix
  real(kind=REAL64), allocatable ::  B(:,:)[:]      ! buffer to hold transposed matrix
  real(kind=REAL64), allocatable ::  T(:,:)         ! temporary to hold tile
  integer(kind=INT64) ::  bytes                     ! combined size of matrices
  ! distributed data helpers
  integer(kind=INT32) :: col_per_pe                 ! columns per PE = order/npes
  integer(kind=INT32) :: col_start, row_start
  ! runtime variables
  integer(kind=INT32) ::  i, j, k, p, q
  integer(kind=INT32) ::  it, jt, tile_size
  real(kind=REAL64) ::  abserr, addit, temp         ! squared error
  real(kind=REAL64) ::  t0, t1, trans_time, avgtime ! timing parameters
  real(kind=REAL64), parameter ::  epsilon=1.D-8    ! error tolerance

  me   = this_image()-1 ! use 0-based indexing of PEs
  npes = num_images()
  printer = (me.eq.0)

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  if (printer) then
    write(6,'(a25)') 'Parallel Research Kernels'
    write(6,'(a41)') 'Fortran coarray Matrix transpose: B = A^T'
  endif

  if (command_argument_count().lt.2) then
    if (printer) then
      write(*,'(a17,i1)') 'argument count = ', command_argument_count()
      write(6,'(a62)')    'Usage: ./transpose <# iterations> <matrix order> [<tile_size>]'
    endif
    stop 1
  endif

  iterations = 1
  call get_command_argument(1,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') iterations
  if (iterations .lt. 1) then
    if (printer) then
      write(6,'(a35,i5)') 'ERROR: iterations must be >= 1 : ', iterations
    endif
    stop 1
  endif

  order = 1
  call get_command_argument(2,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') order
  if (order .lt. 1) then
    if (printer) then
      write(6,'(a30,i5)') 'ERROR: order must be >= 1 : ', order
    endif
    stop 1
  endif
  if (modulo(order,npes).gt.0) then
    if (printer) then
      write(6,'(a20,i5,a35,i5)') 'ERROR: matrix order ',order,&
                        ' should be divisible by # images ',npes
    endif
    stop 1
  endif
  col_per_pe = order/npes

  ! same default as the C implementation
  tile_size = 32
  if (command_argument_count().gt.2) then
      call get_command_argument(3,argtmp,arglen,err)
      if (err.eq.0) read(argtmp,'(i32)') tile_size
  endif
  if ((tile_size .lt. 1).or.(tile_size.gt.order)) then
    write(6,'(a20,i5,a22,i5)') 'WARNING: tile_size ',tile_size,&
                           ' must be >= 1 and <= ',order
    tile_size = order ! no tiling
  endif

  ! ********************************************************************
  ! ** Allocate space for the input and transpose matrix
  ! ********************************************************************

  allocate( A(order,col_per_pe)[*], stat=err)
  if (err .ne. 0) then
    write(6,'(a20,i3,a10,i5)') 'allocation of A returned ',err,' at image ',me
    stop 1
  endif

  allocate( B(order,col_per_pe)[*], stat=err )
  if (err .ne. 0) then
    write(6,'(a20,i3,a10,i5)') 'allocation of B returned ',err,' at image ',me
    stop 1
  endif

  allocate( T(col_per_pe,col_per_pe), stat=err )
  if (err .ne. 0) then
    write(6,'(a20,i3,a10,i5)') 'allocation of T returned ',err,' at image ',me
    stop 1
  endif

  bytes = 2 * int(order,INT64) * int(order,INT64) * storage_size(A)/8

  if (printer) then
    write(6,'(a23,i8)') 'Number of images     = ', num_images()
    write(6,'(a23,i8)') 'Number of iterations = ', iterations
    write(6,'(a23,i8)') 'Matrix order         = ', order
    write(6,'(a23,i8)') 'Tile size            = ', tile_size
  endif

  ! initialization
  ! local column index j corresponds to global column index col_per_pe*me+j
  if ((tile_size.gt.1).and.(tile_size.lt.order)) then
    do concurrent (jt=1:col_per_pe:tile_size)
      do concurrent (it=1:order:tile_size)
        do j=jt,min(col_per_pe,jt+tile_size-1)
          do i=it,min(order,it+tile_size-1)
            A(i,j) = real(order,REAL64) * real(col_per_pe*me+j-1,REAL64) + real(i-1,REAL64)
            B(i,j) = 0.0
          enddo
        enddo
      enddo
    enddo
  else
    do concurrent (j=1:col_per_pe)
      do i=1,order
        A(i,j) = real(order,REAL64) * real(col_per_pe*me+j-1,REAL64) + real(i-1,REAL64)
        B(i,j) = 0.0
      enddo
    enddo
  endif
  sync all ! barrier to ensure initialization is finished at all PEs

  t0 = 0

  do k=0,iterations

    if (k.eq.1) then
      sync all ! barrier
      t0 = prk_get_wtime()
    endif

    ! we shift the loop range from [0,npes-1] to [me,me+npes-1]
    ! to balance communication.  if everyone starts at 0, they will
    ! take turns blasting each image in the system with get operations.
    ! side note: this trick is used extensively in NWChem.
    do q=me,me+npes-1
      p = modulo(q,npes)
      ! Step 1: Gather A tile from remote image
      row_start = me*col_per_pe
      ! * fully explicit version
      !do i=1,col_per_pe
      !  do j=1,col_per_pe
      !    T(j,i) = A(row_start+j,i)[p+1]
      !  enddo
      !enddo
      ! * half explicit, half colon
      !do i=1,col_per_pe
      !    T(:,i) = A(row_start+1:row_start+col_per_pe,i)[p+1]
      !enddo
      ! * full colon
      T(:,:) = A(row_start+1:row_start+col_per_pe,:)[p+1]
      ! Step 2: Transpose tile into B matrix
      col_start = p*col_per_pe
      ! Transpose the  matrix; only use tiling if the tile size is smaller than the matrix
      if ((tile_size.gt.1).and.(tile_size.lt.order)) then
        do concurrent (jt=1:col_per_pe:tile_size)
          do concurrent (it=1:col_per_pe:tile_size)
            do j=jt,min(col_per_pe,jt+tile_size-1)
              do i=it,min(col_per_pe,it+tile_size-1)
                B(col_start+i,j) = B(col_start+i,j) + T(j,i)
              enddo
            enddo
          enddo
        enddo
      else ! untiled
        ! * fully explicit version
        !do j=1,col_per_pe
        !  do i=1,col_per_pe
        !    B(col_start+i,j) = B(col_start+i,j) + T(j,i)
        !  enddo
        !enddo
        ! * half explicit, half colon
        do concurrent (j=1:col_per_pe)
          B(col_start+1:col_start+col_per_pe,j) = B(col_start+1:col_start+col_per_pe,j) + T(j,:)
        enddo
      endif
    enddo
    sync all
    ! Step 3: Update A matrix
    ! * fully explicit version
    !do j=1,col_per_pe
    !  do i=1,order
    !    A(i,j) = A(i,j) + 1.0
    !  enddo
    !enddo
    ! * half explicit, half colon
    do concurrent (j=1:col_per_pe)
       A(:,j) = A(:,j) + 1.0
    enddo
    ! * fully implicit version
    !A = A + 1.0
    sync all

  enddo ! iterations

  sync all ! barrier
  t1 = prk_get_wtime()
  trans_time = t1 - t0

  deallocate( T )
  deallocate( A )

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  abserr = 0.0;
  addit = (0.5*iterations) * (iterations+1.0)
  do j=1,col_per_pe
    do i=1,order
      temp = ((real(order,REAL64)*real(i-1,REAL64))+real(col_per_pe*me+j-1,REAL64)) &
           * real(iterations+1,REAL64) + addit
      abserr = abserr + abs(B(i,j) - temp)
    enddo
  enddo

  deallocate( B )

  if (abserr .lt. (epsilon/npes)) then
    if (printer) then
      write(6,'(a)') 'Solution validates'
      avgtime = trans_time/iterations
      write(6,'(a12,f13.6,a17,f10.6)') 'Rate (MB/s): ',&
              (1.d-6*bytes/avgtime),' Avg time (s): ', avgtime
    endif
  else
    if (printer) then
      write(6,'(a30,f13.6,a18,f13.6)') 'ERROR: Aggregate squared error ', &
              abserr,' exceeds threshold ',(epsilon/npes)
    endif
    stop 1
  endif

end program main
