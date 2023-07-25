!
! Copyright (c) 2015, Intel Corporation
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

program main
  use, intrinsic :: iso_fortran_env
  use prk
  implicit none
  integer :: err
  integer :: me, np
  logical :: printer
  ! problem definition
  integer(kind=INT32) ::  iterations                ! number of times to do the transpose
  integer(kind=INT32) ::  order                     ! order of a the matrix
  real(kind=REAL64), allocatable ::  A(:,:)[:]      ! buffer to hold original matrix
  real(kind=REAL64), allocatable ::  B(:,:)[:]      ! buffer to hold transposed matrix
  real(kind=REAL64), allocatable ::  T(:,:)         ! temporary to hold tile
  integer(kind=INT64) ::  bytes                     ! combined size of matrices
  ! distributed data helpers
  integer(kind=INT32) :: block_order                ! columns per PE = order/np
  integer(kind=INT32) :: col_start, row_start
  ! runtime variables
  integer(kind=INT32) ::  i, j, k, p, q
  integer(kind=INT32) ::  it, jt, tile_size
  real(kind=REAL64) ::  abserr, addit, temp         ! squared error
  real(kind=REAL64) ::  t0, t1, trans_time, avgtime ! timing parameters
  real(kind=REAL64), parameter ::  epsilon=1.D-8    ! error tolerance

  me   = this_image()-1 ! use 0-based indexing of PEs
  np = num_images()
  printer = (me.eq.0)

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  if (printer) then
    call prk_get_arguments('transpose',iterations=iterations,order=order,tile_size=tile_size)
    write(6,'(a25)') 'Parallel Research Kernels'
    write(6,'(a41)') 'Fortran coarray Matrix transpose: B = A^T'
    write(6,'(a23,i8)') 'Number of images     = ', np
    write(6,'(a23,i8)') 'Number of iterations = ', iterations
    write(6,'(a23,i8)') 'Matrix order         = ', order
    write(6,'(a23,i8)') 'Tile size            = ', tile_size
  endif
  call co_broadcast(iterations,1)
  call co_broadcast(order,1)
  call co_broadcast(tile_size,1)

  if (modulo(order,np).gt.0) then
    if (printer) then
      write(6,'(a20,i5,a35,i5)') 'ERROR: matrix order ',order,&
                        ' should be divisible by # images ',np
    endif
    stop 1
  endif
  block_order = order/np

  ! ********************************************************************
  ! ** Allocate space for the input and transpose matrix
  ! ********************************************************************

  allocate( A(order,block_order)[*], B(order,block_order)[*], T(block_order,block_order), stat=err)
  if (err .ne. 0) then
    write(6,'(a20,i3,a10,i5)') 'allocation returned ',err,' at image ',me
    stop 1
  endif

  ! initialization
  ! local column index j corresponds to global column index block_order*me+j
  if ((tile_size.gt.1).and.(tile_size.lt.order)) then
    do concurrent (jt=1:block_order:tile_size, &
                   it=1:order:tile_size)
        do j=jt,min(block_order,jt+tile_size-1)
          do i=it,min(order,it+tile_size-1)
            A(i,j) = real(order,REAL64) * real(block_order*me+j-1,REAL64) + real(i-1,REAL64)
            B(i,j) = 0.0
          enddo
        enddo
    enddo
  else
    do concurrent (j=1:block_order)
      do i=1,order
        A(i,j) = real(order,REAL64) * real(block_order*me+j-1,REAL64) + real(i-1,REAL64)
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

    ! we shift the loop range from [0,np-1] to [me,me+np-1]
    ! to balance communication.  if everyone starts at 0, they will
    ! take turns blasting each image in the system with get operations.
    ! side note: this trick is used extensively in NWChem.
    do q=me,me+np-1
      p = modulo(q,np)
      ! Step 1: Gather A tile from remote image
      row_start = me*block_order
      ! * fully explicit version
      !do i=1,block_order
      !  do j=1,block_order
      !    T(j,i) = A(row_start+j,i)[p+1]
      !  enddo
      !enddo
      ! * half explicit, half colon
      !do i=1,block_order
      !    T(:,i) = A(row_start+1:row_start+block_order,i)[p+1]
      !enddo
      ! * full colon
      T(:,:) = A(row_start+1:row_start+block_order,:)[p+1]
      ! Step 2: Transpose tile into B matrix
      col_start = p*block_order
      ! Transpose the  matrix; only use tiling if the tile size is smaller than the matrix
      if ((tile_size.gt.1).and.(tile_size.lt.order)) then
        do concurrent (jt=1:block_order:tile_size, &
                       it=1:block_order:tile_size)
            do j=jt,min(block_order,jt+tile_size-1)
              do i=it,min(block_order,it+tile_size-1)
                B(col_start+i,j) = B(col_start+i,j) + T(j,i)
              enddo
            enddo
        enddo
      else ! untiled
        ! * fully explicit version
        !do j=1,block_order
        !  do i=1,block_order
        !    B(col_start+i,j) = B(col_start+i,j) + T(j,i)
        !  enddo
        !enddo
        ! * half explicit, half colon
        do concurrent (j=1:block_order)
          B(col_start+1:col_start+block_order,j) = B(col_start+1:col_start+block_order,j) + T(j,:)
        enddo
      endif
    enddo
    sync all
    ! Step 3: Update A matrix
    ! * fully explicit version
    !do j=1,block_order
    !  do i=1,order
    !    A(i,j) = A(i,j) + 1.0
    !  enddo
    !enddo
    ! * half explicit, half colon
    do concurrent (j=1:block_order)
       A(:,j) = A(:,j) + 1.0
    enddo
    ! * fully implicit version
    !A = A + 1.0
    sync all

  enddo ! iterations

  t1 = prk_get_wtime()
  trans_time = t1 - t0

  deallocate( A,T )

  ! ********************************************************************
  ! ** Analyze and output results.
  ! ********************************************************************

  abserr = 0.0;
  addit = (0.5*iterations) * (iterations+1.0)
  do j=1,block_order
    do i=1,order
      temp = ((real(order,REAL64)*real(i-1,REAL64))+real(block_order*me+j-1,REAL64)) &
           * real(iterations+1,REAL64) + addit
      abserr = abserr + abs(B(i,j) - temp)
    enddo
  enddo

  deallocate( B )

  if (abserr .lt. (epsilon/np)) then
    if (printer) then
      write(6,'(a)') 'Solution validates'
      avgtime = trans_time/iterations
      bytes = 2 * int(order,INT64) * int(order,INT64) * storage_size(A(1,1))/8
      write(6,'(a12,f13.6,a17,f10.6)') 'Rate (MB/s): ',&
              (1.d-6*bytes/avgtime),' Avg time (s): ', avgtime
    endif
  else
    if (printer) then
      write(6,'(a30,f13.6,a18,f13.6)') 'ERROR: Aggregate squared error ', &
              abserr,' exceeds threshold ',(epsilon/np)
    endif
    stop 1
  endif

end program main
