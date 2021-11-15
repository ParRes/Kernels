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

end module prk
