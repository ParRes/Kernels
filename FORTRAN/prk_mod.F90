module prk
  contains
      function prk_get_wtime() result(t)
        use iso_fortran_env
        implicit none
        real(kind=REAL64) ::  t
        integer(kind=INT64) :: c, r
        call system_clock(count = c, count_rate = r)
        t = real(c,REAL64) / real(r,REAL64)
      end function prk_get_wtime

      subroutine prk_get_arguments(kernel,           & ! which kernel am i parsing?
                                   iterations,       & ! everything
                                   length, offset,   & ! nstream
                                   order, tile_size, & ! transpose, stencil, dgemm
                                   stencil, radius)    ! not supported in implementations yet
        use iso_fortran_env
        implicit none
        character(len=*),    intent(in)  :: kernel
        integer(kind=INT32), intent(out) :: iterations
        integer(kind=INT64), intent(out), optional :: length, offset    ! nstream
        integer(kind=INT32), intent(out), optional :: order, tile_size  ! transpose, stencil, dgemm
        integer(kind=INT32), intent(out), optional :: radius            ! stencil
        character(len=4),    intent(out), optional :: stencil           ! stencil

        integer :: argc,arglen,err,a,p,q
        character(len=64) :: argtmp

        iterations = 10
        if (present(length)) then
          length = 1024*1024*32
        endif
        if (present(offset)) then
          offset = 0
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

#ifndef PRK_NO_ARGUMENTS
        if (kernel(1:7).eq.'nstream') then
          if (present(length)) then
            length = 0
          else
            print*,'You cannot parse nstream arguments without length'
            stop
          endif
        else if (    (kernel(1:9).eq.'transpose')     &
                 .or.(kernel(1:7).eq.'stencil')       &
                 .or.(kernel(1:5).eq.'dgemm') ) then
          if (present(order)) then
            order = 0
          else
            print*,'You cannot parse ',kernel,' arguments without order'
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
            write(*,'(a62)') 'Old Usage: <program> <# iterations> <vector length> [<offset>]'
            write(*,'(a87)') 'New Usage: <program> iterations=<# iterations> length=<vector length> [offset=<offset>]'
          else if (    (kernel(1:9).eq.'transpose')     &
                   .or.(kernel(1:7).eq.'stencil')       &
                   .or.(kernel(1:5).eq.'dgemm') ) then
            write(*,'(a57)') 'Old Usage: <program> <# iterations> <order> [<tile_size>]'
            write(*,'(a84)') 'New Usage: <program> iterations=<# iterations> order=<order> [tile_size=<tile_size>]'
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
                endif
              else if (a.eq.3) then
                if (present(offset)) then
                  read(argtmp,'(i15)') offset
                endif
                if (present(tile_size)) then
                  read(argtmp,'(i3)') tile_size
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
              write(*,'(a,i3)') 'ERROR: radius must be positive : ', radius
              stop 1
            endif
          endif
        endif
#endif
      end subroutine

      subroutine initialize_w(is_star,r,W)
        use iso_fortran_env
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
        use iso_fortran_env
        implicit none
        real(kind=REAL64), intent(in) :: mat(:,:)
        integer(kind=INT32), intent(in), optional :: label
        integer(kind=INT32) :: dims(2)
        integer(kind=INT32) :: i,j
        dims(1) = size(mat,1)
        dims(2) = size(mat,2)
        do i=1,dims(1)
          write(6,'(i5,a1)', advance='no') label,':'
          do j=1,dims(2)
            if (present(label)) then
              write(6,'(f10.1)', advance='no') mat(i,j)
            end if
          end do
          write(6,'(a1)',advance='yes') ''
        end do
        flush(6)
      end subroutine print_matrix

end module prk
