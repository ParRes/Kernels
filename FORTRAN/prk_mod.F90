module prk
  contains
      function prk_get_wtime() result(t)
        use, intrinsic :: iso_fortran_env
        implicit none
        real(kind=REAL64) ::  t
        integer(kind=INT64) :: c, r
        call system_clock(count = c, count_rate = r)
        t = real(c,REAL64) / real(r,REAL64)
      end function prk_get_wtime

      subroutine prk_get_arguments(kernel,           & ! which kernel am i parsing?
                                   iterations,       & ! everything
                                   length, offset,   & ! nstream
                                   gpu_block_size,   & ! nstream GPU only
                                   order, tile_size, & ! transpose, stencil, dgemm
                                   dimx, dimy,       & ! p2p
                                   tilex, tiley,     & ! p2p
                                   stencil, radius)    ! not supported in implementations yet
        use, intrinsic :: iso_fortran_env
        implicit none
        character(len=*),    intent(in)  :: kernel
        integer(kind=INT32), intent(out) :: iterations
        integer(kind=INT64), intent(out), optional :: length, offset    ! nstream
        integer(kind=INT32), intent(out), optional :: gpu_block_size    ! nstream GPU only
        integer(kind=INT32), intent(out), optional :: order, tile_size  ! transpose, stencil, dgemm
        integer(kind=INT32), intent(out), optional :: dimx, dimy        ! p2p
        integer(kind=INT32), intent(out), optional :: tilex, tiley      ! p2p
        integer(kind=INT32), intent(out), optional :: radius            ! stencil
        character(len=4),    intent(out), optional :: stencil           ! stencil

        integer(kind=INT32), parameter :: deadbeef = -559038737 ! 3735928559 as int32
        integer :: argc,arglen,err,a,p,q
        character(len=64) :: argtmp

        iterations = 10
        if (present(length)) then
          length = 1024*1024*32
        endif
        if (present(offset)) then
          offset = 0
        endif
        if (present(gpu_block_size)) then
          gpu_block_size = 256
        endif
        if (present(order)) then
          order = 1024
        endif
        if (present(tile_size)) then
          tile_size = 32
        endif
        if (present(stencil)) then
          stencil = 'star'
        endif
        if (present(radius)) then
          radius = 2
        endif
        if (present(dimx)) then
          dimx = 1024
        endif
        if (present(dimy)) then
          dimy = deadbeef
        endif
        if (present(tilex)) then
          tilex = 0
        endif
        if (present(tiley)) then
          tiley = deadbeef
        endif

#ifndef PRK_NO_ARGUMENTS
        if (kernel(1:7).eq.'nstream') then
          if (.not.present(length)) then
            print*,'You cannot parse nstream arguments without length'
            stop
          endif
        else if (    (kernel(1:9).eq.'transpose')     &
                 .or.(kernel(1:7).eq.'stencil')       &
                 .or.(kernel(1:5).eq.'dgemm') ) then
          if (.not.present(order)) then
            print*,'You cannot parse ',kernel,' arguments without order'
            stop
          endif
        else if (kernel(1:3).eq.'p2p') then
          if (.not.present(dimx)) then
            print*,'You cannot parse ',kernel,' arguments without dimx'
            stop
          endif
        else
          print*,kernel,'is not supported yet'
          stop
        endif

        argc = command_argument_count()

        if (argc.lt.2 ) then
          write(*,'(a17,i2)') 'argument count = ', command_argument_count()
          if (kernel(1:7).eq.'nstream') then
            if (present(gpu_block_size)) then
              write(*,'(a62)') 'Old Usage: <program> <# iterations> <vector length> [<gpu_block_size>]'
              write(*,'(a87)') 'New Usage: <program> iterations=<# iterations> length=<vector length>', &
                                                                               '[block_size=<gpu_block_size>]'
            else
              write(*,'(a62)') 'Old Usage: <program> <# iterations> <vector length> [<offset>]'
              write(*,'(a87)') 'New Usage: <program> iterations=<# iterations> length=<vector length> [offset=<offset>]'
            endif
          else if (    (kernel(1:9).eq.'transpose')     &
                   .or.(kernel(1:7).eq.'stencil')       &
                   .or.(kernel(1:5).eq.'dgemm') ) then
            write(*,'(a57)') 'Old Usage: <program> <# iterations> <order> [<tile_size>]'
            write(*,'(a84)') 'New Usage: <program> iterations=<# iterations> order=<order> [tile_size=<tile_size>]'
          else if (kernel(1:3).eq.'p2p') then
            if (present(dimy)) then
              write(*,'(a75)') 'Old Usage: <program> <# iterations> <array x-dimension> <array y-dimension>'
              write(*,'(a57)') '          [<tilesize x-dimension> <tilesize y-dimension>]'
              write(*,'(a46)') 'New Usage: <program> iterations=<# iterations>'
              write(*,'(a61)') '           dimx=<array x-dimension> dimy=<array y-dimension>'
              write(*,'(a69)') '           [tilex=<tilesize x-dimension> tiley=<tilesize y-dimension>]'
           else
              write(*,'(a57)') 'Old Usage: <program> <# iterations> <array x-dimension> [<tilesize x-dimension>]'
              write(*,'(a84)') 'New Usage: <program> iterations=<# iterations> dimx=<array x-dimension>', &
                               '[tilex=<tilesize x-dimension>]'
           endif
          endif
          STOP
        endif

        do a=1,argc
          call get_command_argument(a,argtmp,arglen,err)
          if (err.eq.0) then
            p = index(argtmp,"=")
            if (p.eq.0) then
              if (a.eq.1) then
                read(argtmp,'(i10)') iterations
              else if (a.eq.2) then
                if (present(length)) then
                  read(argtmp,'(i15)') length
                else if (present(order)) then
                  read(argtmp,'(i7)') order
                else if (present(dimx)) then
                  read(argtmp,'(i7)') dimx
                endif
              else if (a.eq.3) then
                if (present(offset)) then
                  read(argtmp,'(i15)') offset
                else if (present(tile_size)) then
                  read(argtmp,'(i3)') tile_size
                else if (present(dimy)) then
                  read(argtmp,'(i7)') dimy
                else if (.not.present(dimy).and.present(tilex)) then
                  read(argtmp,'(i7)') tilex
                endif
              elseif (a.eq.4) then
                if (present(dimx).and.present(dimy).and.present(tilex)) then
                  read(argtmp,'(i7)') tilex
                endif
              elseif (a.eq.5) then
                if (present(dimx).and.present(dimy).and. &
                    present(tilex).and.present(tiley)) then
                  read(argtmp,'(i7)') tiley
                endif
              else
                print*,'too many positional arguments:',argc
              endif
            else ! found an =
              ! look for iterations
              q = index(argtmp(1:p-1),"it")
              if (q.eq.1) then
                read(argtmp(p+1:arglen),'(i10)') iterations
              endif
              ! look for length
              if (present(length)) then
                q = index(argtmp(1:p-1),"len")
                if (q.eq.1) then
                    read(argtmp(p+1:arglen),'(i15)') length
                endif
              endif
              ! look for offset
              if (present(offset)) then
                q = index(argtmp(1:p-1),"off")
                if (q.eq.1) then
                  read(argtmp(p+1:arglen),'(i15)') offset
                endif
              endif
              ! look for order
              if (present(order)) then
                q = index(argtmp(1:p-1),"ord")
                if (q.eq.1) then
                  read(argtmp(p+1:arglen),'(i7)') order
                endif
              endif
              ! look for tile_size
              if (present(tile_size)) then
                q = index(argtmp(1:p-1),"tile")
                if (q.eq.1) then
                  read(argtmp(p+1:arglen),'(i3)') tile_size
                endif
              endif
              ! look for radius
              if (present(radius)) then
                q = index(argtmp(1:p-1),"rad")
                if (q.eq.1) then
                  read(argtmp(p+1:arglen),'(i1)') radius
                endif
              endif
              ! look for gpu_block_size
              if (present(gpu_block_size)) then
                q = index(argtmp(1:p-1),"block")
                if (q.ne.0) then
                  read(argtmp(p+1:arglen),'(i5)') gpu_block_size
                endif
              endif
              ! look for dimx
              if (present(dimx)) then
                q = index(argtmp(1:p-1),"dimx")
                if (q.eq.1) then
                  read(argtmp(p+1:arglen),'(i7)') dimx
                endif
                ! look for tilex
                if (present(tilex)) then
                  q = index(argtmp(1:p-1),"tilex")
                  if (q.eq.1) then
                    read(argtmp(p+1:arglen),'(i3)') tilex
                  endif
                endif
                ! look for dimy
                if (present(dimy)) then
                  q = index(argtmp(1:p-1),"dimy")
                  if (q.eq.1) then
                    read(argtmp(p+1:arglen),'(i7)') dimy
                  endif
                  ! look for tiley
                  if (present(tiley)) then
                    q = index(argtmp(1:p-1),"tiley")
                    if (q.eq.1) then
                      read(argtmp(p+1:arglen),'(i3)') tiley
                    endif
                  endif
                endif
              endif
              ! end looking
            endif
          endif
        enddo

        ! check all the relevant arguments for validity
        if (iterations .lt. 1) then
          write(*,'(a,i5)') 'ERROR: iterations must be positive : ', iterations
          stop 1
        endif

        ! nstream
        if (present(length)) then
          if (length .lt. 1) then
            write(*,'(a,i15)') 'ERROR: length must be positive : ', length
            stop 1
          endif
          if (present(offset)) then
            if (offset .lt. 0) then
              write(*,'(a,i15)') 'ERROR: offset must be nonnegative : ', offset
              stop 1
            endif
          endif
        endif

        ! transpose, stencil, dgemm
        if (present(order)) then
          if (order .lt. 1) then
            write(*,'(a,i7)') 'ERROR: order must be positive : ', order
            stop 1
          endif
          if (present(tile_size)) then
            if ((tile_size .lt. 1).or.(tile_size.gt.order)) then
              write(*,'(a18,i3,a22,i5)') 'WARNING: tile_size ',tile_size,&
                                         ' must be between 1 and ',order  
              tile_size = order ! no tiling
            endif
          endif
          ! stencil
          if (present(radius)) then
            if (radius .lt. 1) then
              write(*,'(a27)') 'ERROR: radius must be between 1 and 9'
              stop 1
            endif
          endif
        endif

        ! p2p
        if (present(dimx)) then
          if (dimx.lt.1) then
            write(*,'(a,i7)') 'ERROR: dimx must be positive : ', dimx
            stop 1
          endif
          if (present(tilex)) then
            if ((tilex.lt.1).or.(tilex.gt.dimx)) then
              write(*,'(a,i7)') 'WARNING: tilex invalid - ignoring'
              tilex = dimx
            endif
          endif
          if (present(dimy)) then
            ! user did not provide it, so we assume square array
            if (dimy.eq.deadbeef) then
              dimy = dimx
            endif
            if (dimy.lt.1) then
              write(*,'(a,i7)') 'ERROR: dimy must be positive : ', dimy
              stop 1
            endif
            if (present(tiley)) then
              ! user did not provide it, so we assume square array
              if (tiley.eq.deadbeef) then
                tiley = tilex
              endif
              if ((tiley.lt.1).or.(tiley.gt.dimy)) then
                write(*,'(a,i7)') 'WARNING: tiley invalid - ignoring'
                tiley = dimy
              endif
            endif
          endif
        endif
#endif
      end subroutine

      subroutine initialize_w(is_star,r,W)
        use, intrinsic :: iso_fortran_env
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

      subroutine print_matrix(mat, label)
        use, intrinsic :: iso_fortran_env
        implicit none
        real(kind=REAL64), intent(in) :: mat(:,:)
        integer(kind=INT32), intent(in), optional :: label
        integer(kind=INT32) :: dims(2)
        integer(kind=INT32) :: i,j
        dims(1) = size(mat,1)
        dims(2) = size(mat,2)
        do i=1,dims(1)
          if (present(label)) write(6,'(i5,a1)', advance='no') label,':'
          do j=1,dims(2)
            write(6,'(f10.1)', advance='no') mat(i,j)
          end do
          write(6,'(a1)',advance='yes') ''
        end do
        flush(6)
      end subroutine print_matrix

end module prk
