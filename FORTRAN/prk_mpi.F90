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

module prk_mpi
  contains
    subroutine mpi_print_matrix(mat,clabel)
      use, intrinsic :: iso_fortran_env
      use mpi_f08
      use prk
      implicit none
      real(kind=REAL64), intent(in) :: mat(:,:)
      character(*), intent(in), optional :: clabel
      integer(kind=INT32) :: r, me, np
      flush(6)
      call MPI_Comm_rank(MPI_COMM_WORLD, me)
      call MPI_Comm_size(MPI_COMM_WORLD, np)
      call MPI_Barrier(MPI_COMM_WORLD)
      flush(6)
      if (me.eq.0) print*,clabel
      flush(6)
      call MPI_Barrier(MPI_COMM_WORLD)
      flush(6)
      do r=0,np-1
        if (me.eq.r) then
          call print_matrix(mat,me)
        endif
        call MPI_Barrier(MPI_COMM_WORLD)
      enddo
      flush(6)
    end subroutine
end module prk_mpi
