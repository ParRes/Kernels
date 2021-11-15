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
end module prk
