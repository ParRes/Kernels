subroutine star1(n, in, out)
use iso_fortran_env
implicit none
integer(kind=INT32), intent(in) :: n
real(kind=REAL64), intent(in) :: in(n,n)
real(kind=REAL64), intent(inout) :: out(n,n)
integer(kind=INT32) :: i,j
    !$omp taskloop
    do i=1,n-1-1
      !$omp simd
      do j=1,n-1-1
        out(i,j) = out(i,j) &
                 + in(i+0,j-1) * real(-0.5,REAL64) &
                 + in(i-1,j+0) * real(-0.5,REAL64) &
                 + in(i+1,j+0) * real(0.5,REAL64) &
                 + in(i+0,j+1) * real(0.5,REAL64) &
                 + real(0.0,REAL64)
      end do
      !$omp end simd
    end do
    !$omp end taskloop
end subroutine

subroutine star2(n, in, out)
use iso_fortran_env
implicit none
integer(kind=INT32), intent(in) :: n
real(kind=REAL64), intent(in) :: in(n,n)
real(kind=REAL64), intent(inout) :: out(n,n)
integer(kind=INT32) :: i,j
    !$omp taskloop
    do i=2,n-2-1
      !$omp simd
      do j=2,n-2-1
        out(i,j) = out(i,j) &
                 + in(i+0,j-2) * real(-0.125,REAL64) &
                 + in(i+0,j-1) * real(-0.25,REAL64) &
                 + in(i-2,j+0) * real(-0.125,REAL64) &
                 + in(i-1,j+0) * real(-0.25,REAL64) &
                 + in(i+1,j+0) * real(0.25,REAL64) &
                 + in(i+2,j+0) * real(0.125,REAL64) &
                 + in(i+0,j+1) * real(0.25,REAL64) &
                 + in(i+0,j+2) * real(0.125,REAL64) &
                 + real(0.0,REAL64)
      end do
      !$omp end simd
    end do
    !$omp end taskloop
end subroutine

subroutine star3(n, in, out)
use iso_fortran_env
implicit none
integer(kind=INT32), intent(in) :: n
real(kind=REAL64), intent(in) :: in(n,n)
real(kind=REAL64), intent(inout) :: out(n,n)
integer(kind=INT32) :: i,j
    !$omp taskloop
    do i=3,n-3-1
      !$omp simd
      do j=3,n-3-1
        out(i,j) = out(i,j) &
                 + in(i+0,j-3) * real(-0.05555555555555555,REAL64) &
                 + in(i+0,j-2) * real(-0.08333333333333333,REAL64) &
                 + in(i+0,j-1) * real(-0.16666666666666666,REAL64) &
                 + in(i-3,j+0) * real(-0.05555555555555555,REAL64) &
                 + in(i-2,j+0) * real(-0.08333333333333333,REAL64) &
                 + in(i-1,j+0) * real(-0.16666666666666666,REAL64) &
                 + in(i+1,j+0) * real(0.16666666666666666,REAL64) &
                 + in(i+2,j+0) * real(0.08333333333333333,REAL64) &
                 + in(i+3,j+0) * real(0.05555555555555555,REAL64) &
                 + in(i+0,j+1) * real(0.16666666666666666,REAL64) &
                 + in(i+0,j+2) * real(0.08333333333333333,REAL64) &
                 + in(i+0,j+3) * real(0.05555555555555555,REAL64) &
                 + real(0.0,REAL64)
      end do
      !$omp end simd
    end do
    !$omp end taskloop
end subroutine

subroutine star4(n, in, out)
use iso_fortran_env
implicit none
integer(kind=INT32), intent(in) :: n
real(kind=REAL64), intent(in) :: in(n,n)
real(kind=REAL64), intent(inout) :: out(n,n)
integer(kind=INT32) :: i,j
    !$omp taskloop
    do i=4,n-4-1
      !$omp simd
      do j=4,n-4-1
        out(i,j) = out(i,j) &
                 + in(i+0,j-4) * real(-0.03125,REAL64) &
                 + in(i+0,j-3) * real(-0.041666666666666664,REAL64) &
                 + in(i+0,j-2) * real(-0.0625,REAL64) &
                 + in(i+0,j-1) * real(-0.125,REAL64) &
                 + in(i-4,j+0) * real(-0.03125,REAL64) &
                 + in(i-3,j+0) * real(-0.041666666666666664,REAL64) &
                 + in(i-2,j+0) * real(-0.0625,REAL64) &
                 + in(i-1,j+0) * real(-0.125,REAL64) &
                 + in(i+1,j+0) * real(0.125,REAL64) &
                 + in(i+2,j+0) * real(0.0625,REAL64) &
                 + in(i+3,j+0) * real(0.041666666666666664,REAL64) &
                 + in(i+4,j+0) * real(0.03125,REAL64) &
                 + in(i+0,j+1) * real(0.125,REAL64) &
                 + in(i+0,j+2) * real(0.0625,REAL64) &
                 + in(i+0,j+3) * real(0.041666666666666664,REAL64) &
                 + in(i+0,j+4) * real(0.03125,REAL64) &
                 + real(0.0,REAL64)
      end do
      !$omp end simd
    end do
    !$omp end taskloop
end subroutine

subroutine star5(n, in, out)
use iso_fortran_env
implicit none
integer(kind=INT32), intent(in) :: n
real(kind=REAL64), intent(in) :: in(n,n)
real(kind=REAL64), intent(inout) :: out(n,n)
integer(kind=INT32) :: i,j
    !$omp taskloop
    do i=5,n-5-1
      !$omp simd
      do j=5,n-5-1
        out(i,j) = out(i,j) &
                 + in(i+0,j-5) * real(-0.02,REAL64) &
                 + in(i+0,j-4) * real(-0.025,REAL64) &
                 + in(i+0,j-3) * real(-0.03333333333333333,REAL64) &
                 + in(i+0,j-2) * real(-0.05,REAL64) &
                 + in(i+0,j-1) * real(-0.1,REAL64) &
                 + in(i-5,j+0) * real(-0.02,REAL64) &
                 + in(i-4,j+0) * real(-0.025,REAL64) &
                 + in(i-3,j+0) * real(-0.03333333333333333,REAL64) &
                 + in(i-2,j+0) * real(-0.05,REAL64) &
                 + in(i-1,j+0) * real(-0.1,REAL64) &
                 + in(i+1,j+0) * real(0.1,REAL64) &
                 + in(i+2,j+0) * real(0.05,REAL64) &
                 + in(i+3,j+0) * real(0.03333333333333333,REAL64) &
                 + in(i+4,j+0) * real(0.025,REAL64) &
                 + in(i+5,j+0) * real(0.02,REAL64) &
                 + in(i+0,j+1) * real(0.1,REAL64) &
                 + in(i+0,j+2) * real(0.05,REAL64) &
                 + in(i+0,j+3) * real(0.03333333333333333,REAL64) &
                 + in(i+0,j+4) * real(0.025,REAL64) &
                 + in(i+0,j+5) * real(0.02,REAL64) &
                 + real(0.0,REAL64)
      end do
      !$omp end simd
    end do
    !$omp end taskloop
end subroutine

subroutine star6(n, in, out)
use iso_fortran_env
implicit none
integer(kind=INT32), intent(in) :: n
real(kind=REAL64), intent(in) :: in(n,n)
real(kind=REAL64), intent(inout) :: out(n,n)
integer(kind=INT32) :: i,j
    !$omp taskloop
    do i=6,n-6-1
      !$omp simd
      do j=6,n-6-1
        out(i,j) = out(i,j) &
                 + in(i+0,j-6) * real(-0.013888888888888888,REAL64) &
                 + in(i+0,j-5) * real(-0.016666666666666666,REAL64) &
                 + in(i+0,j-4) * real(-0.020833333333333332,REAL64) &
                 + in(i+0,j-3) * real(-0.027777777777777776,REAL64) &
                 + in(i+0,j-2) * real(-0.041666666666666664,REAL64) &
                 + in(i+0,j-1) * real(-0.08333333333333333,REAL64) &
                 + in(i-6,j+0) * real(-0.013888888888888888,REAL64) &
                 + in(i-5,j+0) * real(-0.016666666666666666,REAL64) &
                 + in(i-4,j+0) * real(-0.020833333333333332,REAL64) &
                 + in(i-3,j+0) * real(-0.027777777777777776,REAL64) &
                 + in(i-2,j+0) * real(-0.041666666666666664,REAL64) &
                 + in(i-1,j+0) * real(-0.08333333333333333,REAL64) &
                 + in(i+1,j+0) * real(0.08333333333333333,REAL64) &
                 + in(i+2,j+0) * real(0.041666666666666664,REAL64) &
                 + in(i+3,j+0) * real(0.027777777777777776,REAL64) &
                 + in(i+4,j+0) * real(0.020833333333333332,REAL64) &
                 + in(i+5,j+0) * real(0.016666666666666666,REAL64) &
                 + in(i+6,j+0) * real(0.013888888888888888,REAL64) &
                 + in(i+0,j+1) * real(0.08333333333333333,REAL64) &
                 + in(i+0,j+2) * real(0.041666666666666664,REAL64) &
                 + in(i+0,j+3) * real(0.027777777777777776,REAL64) &
                 + in(i+0,j+4) * real(0.020833333333333332,REAL64) &
                 + in(i+0,j+5) * real(0.016666666666666666,REAL64) &
                 + in(i+0,j+6) * real(0.013888888888888888,REAL64) &
                 + real(0.0,REAL64)
      end do
      !$omp end simd
    end do
    !$omp end taskloop
end subroutine

subroutine star7(n, in, out)
use iso_fortran_env
implicit none
integer(kind=INT32), intent(in) :: n
real(kind=REAL64), intent(in) :: in(n,n)
real(kind=REAL64), intent(inout) :: out(n,n)
integer(kind=INT32) :: i,j
    !$omp taskloop
    do i=7,n-7-1
      !$omp simd
      do j=7,n-7-1
        out(i,j) = out(i,j) &
                 + in(i+0,j-7) * real(-0.01020408163265306,REAL64) &
                 + in(i+0,j-6) * real(-0.011904761904761904,REAL64) &
                 + in(i+0,j-5) * real(-0.014285714285714285,REAL64) &
                 + in(i+0,j-4) * real(-0.017857142857142856,REAL64) &
                 + in(i+0,j-3) * real(-0.023809523809523808,REAL64) &
                 + in(i+0,j-2) * real(-0.03571428571428571,REAL64) &
                 + in(i+0,j-1) * real(-0.07142857142857142,REAL64) &
                 + in(i-7,j+0) * real(-0.01020408163265306,REAL64) &
                 + in(i-6,j+0) * real(-0.011904761904761904,REAL64) &
                 + in(i-5,j+0) * real(-0.014285714285714285,REAL64) &
                 + in(i-4,j+0) * real(-0.017857142857142856,REAL64) &
                 + in(i-3,j+0) * real(-0.023809523809523808,REAL64) &
                 + in(i-2,j+0) * real(-0.03571428571428571,REAL64) &
                 + in(i-1,j+0) * real(-0.07142857142857142,REAL64) &
                 + in(i+1,j+0) * real(0.07142857142857142,REAL64) &
                 + in(i+2,j+0) * real(0.03571428571428571,REAL64) &
                 + in(i+3,j+0) * real(0.023809523809523808,REAL64) &
                 + in(i+4,j+0) * real(0.017857142857142856,REAL64) &
                 + in(i+5,j+0) * real(0.014285714285714285,REAL64) &
                 + in(i+6,j+0) * real(0.011904761904761904,REAL64) &
                 + in(i+7,j+0) * real(0.01020408163265306,REAL64) &
                 + in(i+0,j+1) * real(0.07142857142857142,REAL64) &
                 + in(i+0,j+2) * real(0.03571428571428571,REAL64) &
                 + in(i+0,j+3) * real(0.023809523809523808,REAL64) &
                 + in(i+0,j+4) * real(0.017857142857142856,REAL64) &
                 + in(i+0,j+5) * real(0.014285714285714285,REAL64) &
                 + in(i+0,j+6) * real(0.011904761904761904,REAL64) &
                 + in(i+0,j+7) * real(0.01020408163265306,REAL64) &
                 + real(0.0,REAL64)
      end do
      !$omp end simd
    end do
    !$omp end taskloop
end subroutine

subroutine star8(n, in, out)
use iso_fortran_env
implicit none
integer(kind=INT32), intent(in) :: n
real(kind=REAL64), intent(in) :: in(n,n)
real(kind=REAL64), intent(inout) :: out(n,n)
integer(kind=INT32) :: i,j
    !$omp taskloop
    do i=8,n-8-1
      !$omp simd
      do j=8,n-8-1
        out(i,j) = out(i,j) &
                 + in(i+0,j-8) * real(-0.0078125,REAL64) &
                 + in(i+0,j-7) * real(-0.008928571428571428,REAL64) &
                 + in(i+0,j-6) * real(-0.010416666666666666,REAL64) &
                 + in(i+0,j-5) * real(-0.0125,REAL64) &
                 + in(i+0,j-4) * real(-0.015625,REAL64) &
                 + in(i+0,j-3) * real(-0.020833333333333332,REAL64) &
                 + in(i+0,j-2) * real(-0.03125,REAL64) &
                 + in(i+0,j-1) * real(-0.0625,REAL64) &
                 + in(i-8,j+0) * real(-0.0078125,REAL64) &
                 + in(i-7,j+0) * real(-0.008928571428571428,REAL64) &
                 + in(i-6,j+0) * real(-0.010416666666666666,REAL64) &
                 + in(i-5,j+0) * real(-0.0125,REAL64) &
                 + in(i-4,j+0) * real(-0.015625,REAL64) &
                 + in(i-3,j+0) * real(-0.020833333333333332,REAL64) &
                 + in(i-2,j+0) * real(-0.03125,REAL64) &
                 + in(i-1,j+0) * real(-0.0625,REAL64) &
                 + in(i+1,j+0) * real(0.0625,REAL64) &
                 + in(i+2,j+0) * real(0.03125,REAL64) &
                 + in(i+3,j+0) * real(0.020833333333333332,REAL64) &
                 + in(i+4,j+0) * real(0.015625,REAL64) &
                 + in(i+5,j+0) * real(0.0125,REAL64) &
                 + in(i+6,j+0) * real(0.010416666666666666,REAL64) &
                 + in(i+7,j+0) * real(0.008928571428571428,REAL64) &
                 + in(i+8,j+0) * real(0.0078125,REAL64) &
                 + in(i+0,j+1) * real(0.0625,REAL64) &
                 + in(i+0,j+2) * real(0.03125,REAL64) &
                 + in(i+0,j+3) * real(0.020833333333333332,REAL64) &
                 + in(i+0,j+4) * real(0.015625,REAL64) &
                 + in(i+0,j+5) * real(0.0125,REAL64) &
                 + in(i+0,j+6) * real(0.010416666666666666,REAL64) &
                 + in(i+0,j+7) * real(0.008928571428571428,REAL64) &
                 + in(i+0,j+8) * real(0.0078125,REAL64) &
                 + real(0.0,REAL64)
      end do
      !$omp end simd
    end do
    !$omp end taskloop
end subroutine

subroutine star9(n, in, out)
use iso_fortran_env
implicit none
integer(kind=INT32), intent(in) :: n
real(kind=REAL64), intent(in) :: in(n,n)
real(kind=REAL64), intent(inout) :: out(n,n)
integer(kind=INT32) :: i,j
    !$omp taskloop
    do i=9,n-9-1
      !$omp simd
      do j=9,n-9-1
        out(i,j) = out(i,j) &
                 + in(i+0,j-9) * real(-0.006172839506172839,REAL64) &
                 + in(i+0,j-8) * real(-0.006944444444444444,REAL64) &
                 + in(i+0,j-7) * real(-0.007936507936507936,REAL64) &
                 + in(i+0,j-6) * real(-0.009259259259259259,REAL64) &
                 + in(i+0,j-5) * real(-0.011111111111111112,REAL64) &
                 + in(i+0,j-4) * real(-0.013888888888888888,REAL64) &
                 + in(i+0,j-3) * real(-0.018518518518518517,REAL64) &
                 + in(i+0,j-2) * real(-0.027777777777777776,REAL64) &
                 + in(i+0,j-1) * real(-0.05555555555555555,REAL64) &
                 + in(i-9,j+0) * real(-0.006172839506172839,REAL64) &
                 + in(i-8,j+0) * real(-0.006944444444444444,REAL64) &
                 + in(i-7,j+0) * real(-0.007936507936507936,REAL64) &
                 + in(i-6,j+0) * real(-0.009259259259259259,REAL64) &
                 + in(i-5,j+0) * real(-0.011111111111111112,REAL64) &
                 + in(i-4,j+0) * real(-0.013888888888888888,REAL64) &
                 + in(i-3,j+0) * real(-0.018518518518518517,REAL64) &
                 + in(i-2,j+0) * real(-0.027777777777777776,REAL64) &
                 + in(i-1,j+0) * real(-0.05555555555555555,REAL64) &
                 + in(i+1,j+0) * real(0.05555555555555555,REAL64) &
                 + in(i+2,j+0) * real(0.027777777777777776,REAL64) &
                 + in(i+3,j+0) * real(0.018518518518518517,REAL64) &
                 + in(i+4,j+0) * real(0.013888888888888888,REAL64) &
                 + in(i+5,j+0) * real(0.011111111111111112,REAL64) &
                 + in(i+6,j+0) * real(0.009259259259259259,REAL64) &
                 + in(i+7,j+0) * real(0.007936507936507936,REAL64) &
                 + in(i+8,j+0) * real(0.006944444444444444,REAL64) &
                 + in(i+9,j+0) * real(0.006172839506172839,REAL64) &
                 + in(i+0,j+1) * real(0.05555555555555555,REAL64) &
                 + in(i+0,j+2) * real(0.027777777777777776,REAL64) &
                 + in(i+0,j+3) * real(0.018518518518518517,REAL64) &
                 + in(i+0,j+4) * real(0.013888888888888888,REAL64) &
                 + in(i+0,j+5) * real(0.011111111111111112,REAL64) &
                 + in(i+0,j+6) * real(0.009259259259259259,REAL64) &
                 + in(i+0,j+7) * real(0.007936507936507936,REAL64) &
                 + in(i+0,j+8) * real(0.006944444444444444,REAL64) &
                 + in(i+0,j+9) * real(0.006172839506172839,REAL64) &
                 + real(0.0,REAL64)
      end do
      !$omp end simd
    end do
    !$omp end taskloop
end subroutine

subroutine grid1(n, in, out)
use iso_fortran_env
implicit none
integer(kind=INT32), intent(in) :: n
real(kind=REAL64), intent(in) :: in(n,n)
real(kind=REAL64), intent(inout) :: out(n,n)
integer(kind=INT32) :: i,j
    !$omp taskloop
    do i=1,n-1-1
      !$omp simd
      do j=1,n-1-1
        out(i,j) = out(i,j) &
                 + in(i-1,j-1) * real(-0.25,REAL64) &
                 + in(i+1,j-1) * real(-0.25,REAL64) &
                 + in(i-1,j+1) * real(-0.25,REAL64) &
                 + in(i+1,j+1) * real(0.25,REAL64) &
                 + real(0.0,REAL64)
      end do
      !$omp end simd
    end do
    !$omp end taskloop
end subroutine

subroutine grid2(n, in, out)
use iso_fortran_env
implicit none
integer(kind=INT32), intent(in) :: n
real(kind=REAL64), intent(in) :: in(n,n)
real(kind=REAL64), intent(inout) :: out(n,n)
integer(kind=INT32) :: i,j
    !$omp taskloop
    do i=2,n-2-1
      !$omp simd
      do j=2,n-2-1
        out(i,j) = out(i,j) &
                 + in(i-2,j-2) * real(-0.0625,REAL64) &
                 + in(i+1,j-2) * real(-0.020833333333333332,REAL64) &
                 + in(i+2,j-2) * real(-0.020833333333333332,REAL64) &
                 + in(i-1,j-1) * real(-0.125,REAL64) &
                 + in(i+1,j-1) * real(-0.125,REAL64) &
                 + in(i+2,j-1) * real(-0.125,REAL64) &
                 + in(i-2,j+1) * real(-0.020833333333333332,REAL64) &
                 + in(i-1,j+1) * real(-0.125,REAL64) &
                 + in(i+1,j+1) * real(0.125,REAL64) &
                 + in(i+2,j+1) * real(0.020833333333333332,REAL64) &
                 + in(i-2,j+2) * real(-0.020833333333333332,REAL64) &
                 + in(i-1,j+2) * real(-0.125,REAL64) &
                 + in(i+1,j+2) * real(0.020833333333333332,REAL64) &
                 + in(i+2,j+2) * real(0.0625,REAL64) &
                 + real(0.0,REAL64)
      end do
      !$omp end simd
    end do
    !$omp end taskloop
end subroutine

subroutine grid3(n, in, out)
use iso_fortran_env
implicit none
integer(kind=INT32), intent(in) :: n
real(kind=REAL64), intent(in) :: in(n,n)
real(kind=REAL64), intent(inout) :: out(n,n)
integer(kind=INT32) :: i,j
    !$omp taskloop
    do i=3,n-3-1
      !$omp simd
      do j=3,n-3-1
        out(i,j) = out(i,j) &
                 + in(i-3,j-3) * real(-0.027777777777777776,REAL64) &
                 + in(i+1,j-3) * real(-0.005555555555555556,REAL64) &
                 + in(i+2,j-3) * real(-0.005555555555555556,REAL64) &
                 + in(i+3,j-3) * real(-0.005555555555555556,REAL64) &
                 + in(i-2,j-2) * real(-0.041666666666666664,REAL64) &
                 + in(i+1,j-2) * real(-0.013888888888888888,REAL64) &
                 + in(i+2,j-2) * real(-0.013888888888888888,REAL64) &
                 + in(i+3,j-2) * real(-0.013888888888888888,REAL64) &
                 + in(i-1,j-1) * real(-0.08333333333333333,REAL64) &
                 + in(i+1,j-1) * real(-0.08333333333333333,REAL64) &
                 + in(i+2,j-1) * real(-0.08333333333333333,REAL64) &
                 + in(i+3,j-1) * real(-0.08333333333333333,REAL64) &
                 + in(i-3,j+1) * real(-0.005555555555555556,REAL64) &
                 + in(i-2,j+1) * real(-0.013888888888888888,REAL64) &
                 + in(i-1,j+1) * real(-0.08333333333333333,REAL64) &
                 + in(i+1,j+1) * real(0.08333333333333333,REAL64) &
                 + in(i+2,j+1) * real(0.013888888888888888,REAL64) &
                 + in(i+3,j+1) * real(0.005555555555555556,REAL64) &
                 + in(i-3,j+2) * real(-0.005555555555555556,REAL64) &
                 + in(i-2,j+2) * real(-0.013888888888888888,REAL64) &
                 + in(i-1,j+2) * real(-0.08333333333333333,REAL64) &
                 + in(i+1,j+2) * real(0.013888888888888888,REAL64) &
                 + in(i+2,j+2) * real(0.041666666666666664,REAL64) &
                 + in(i+3,j+2) * real(0.005555555555555556,REAL64) &
                 + in(i-3,j+3) * real(-0.005555555555555556,REAL64) &
                 + in(i-2,j+3) * real(-0.013888888888888888,REAL64) &
                 + in(i-1,j+3) * real(-0.08333333333333333,REAL64) &
                 + in(i+1,j+3) * real(0.005555555555555556,REAL64) &
                 + in(i+2,j+3) * real(0.005555555555555556,REAL64) &
                 + in(i+3,j+3) * real(0.027777777777777776,REAL64) &
                 + real(0.0,REAL64)
      end do
      !$omp end simd
    end do
    !$omp end taskloop
end subroutine

subroutine grid4(n, in, out)
use iso_fortran_env
implicit none
integer(kind=INT32), intent(in) :: n
real(kind=REAL64), intent(in) :: in(n,n)
real(kind=REAL64), intent(inout) :: out(n,n)
integer(kind=INT32) :: i,j
    !$omp taskloop
    do i=4,n-4-1
      !$omp simd
      do j=4,n-4-1
        out(i,j) = out(i,j) &
                 + in(i-4,j-4) * real(-0.015625,REAL64) &
                 + in(i+1,j-4) * real(-0.002232142857142857,REAL64) &
                 + in(i+2,j-4) * real(-0.002232142857142857,REAL64) &
                 + in(i+3,j-4) * real(-0.002232142857142857,REAL64) &
                 + in(i+4,j-4) * real(-0.002232142857142857,REAL64) &
                 + in(i-3,j-3) * real(-0.020833333333333332,REAL64) &
                 + in(i+1,j-3) * real(-0.004166666666666667,REAL64) &
                 + in(i+2,j-3) * real(-0.004166666666666667,REAL64) &
                 + in(i+3,j-3) * real(-0.004166666666666667,REAL64) &
                 + in(i+4,j-3) * real(-0.004166666666666667,REAL64) &
                 + in(i-2,j-2) * real(-0.03125,REAL64) &
                 + in(i+1,j-2) * real(-0.010416666666666666,REAL64) &
                 + in(i+2,j-2) * real(-0.010416666666666666,REAL64) &
                 + in(i+3,j-2) * real(-0.010416666666666666,REAL64) &
                 + in(i+4,j-2) * real(-0.010416666666666666,REAL64) &
                 + in(i-1,j-1) * real(-0.0625,REAL64) &
                 + in(i+1,j-1) * real(-0.0625,REAL64) &
                 + in(i+2,j-1) * real(-0.0625,REAL64) &
                 + in(i+3,j-1) * real(-0.0625,REAL64) &
                 + in(i+4,j-1) * real(-0.0625,REAL64) &
                 + in(i-4,j+1) * real(-0.002232142857142857,REAL64) &
                 + in(i-3,j+1) * real(-0.004166666666666667,REAL64) &
                 + in(i-2,j+1) * real(-0.010416666666666666,REAL64) &
                 + in(i-1,j+1) * real(-0.0625,REAL64) &
                 + in(i+1,j+1) * real(0.0625,REAL64) &
                 + in(i+2,j+1) * real(0.010416666666666666,REAL64) &
                 + in(i+3,j+1) * real(0.004166666666666667,REAL64) &
                 + in(i+4,j+1) * real(0.002232142857142857,REAL64) &
                 + in(i-4,j+2) * real(-0.002232142857142857,REAL64) &
                 + in(i-3,j+2) * real(-0.004166666666666667,REAL64) &
                 + in(i-2,j+2) * real(-0.010416666666666666,REAL64) &
                 + in(i-1,j+2) * real(-0.0625,REAL64) &
                 + in(i+1,j+2) * real(0.010416666666666666,REAL64) &
                 + in(i+2,j+2) * real(0.03125,REAL64) &
                 + in(i+3,j+2) * real(0.004166666666666667,REAL64) &
                 + in(i+4,j+2) * real(0.002232142857142857,REAL64) &
                 + in(i-4,j+3) * real(-0.002232142857142857,REAL64) &
                 + in(i-3,j+3) * real(-0.004166666666666667,REAL64) &
                 + in(i-2,j+3) * real(-0.010416666666666666,REAL64) &
                 + in(i-1,j+3) * real(-0.0625,REAL64) &
                 + in(i+1,j+3) * real(0.004166666666666667,REAL64) &
                 + in(i+2,j+3) * real(0.004166666666666667,REAL64) &
                 + in(i+3,j+3) * real(0.020833333333333332,REAL64) &
                 + in(i+4,j+3) * real(0.002232142857142857,REAL64) &
                 + in(i-4,j+4) * real(-0.002232142857142857,REAL64) &
                 + in(i-3,j+4) * real(-0.004166666666666667,REAL64) &
                 + in(i-2,j+4) * real(-0.010416666666666666,REAL64) &
                 + in(i-1,j+4) * real(-0.0625,REAL64) &
                 + in(i+1,j+4) * real(0.002232142857142857,REAL64) &
                 + in(i+2,j+4) * real(0.002232142857142857,REAL64) &
                 + in(i+3,j+4) * real(0.002232142857142857,REAL64) &
                 + in(i+4,j+4) * real(0.015625,REAL64) &
                 + real(0.0,REAL64)
      end do
      !$omp end simd
    end do
    !$omp end taskloop
end subroutine

subroutine grid5(n, in, out)
use iso_fortran_env
implicit none
integer(kind=INT32), intent(in) :: n
real(kind=REAL64), intent(in) :: in(n,n)
real(kind=REAL64), intent(inout) :: out(n,n)
integer(kind=INT32) :: i,j
    !$omp taskloop
    do i=5,n-5-1
      !$omp simd
      do j=5,n-5-1
        out(i,j) = out(i,j) &
                 + in(i-5,j-5) * real(-0.01,REAL64) &
                 + in(i+1,j-5) * real(-0.0011111111111111111,REAL64) &
                 + in(i+2,j-5) * real(-0.0011111111111111111,REAL64) &
                 + in(i+3,j-5) * real(-0.0011111111111111111,REAL64) &
                 + in(i+4,j-5) * real(-0.0011111111111111111,REAL64) &
                 + in(i+5,j-5) * real(-0.0011111111111111111,REAL64) &
                 + in(i-4,j-4) * real(-0.0125,REAL64) &
                 + in(i+1,j-4) * real(-0.0017857142857142857,REAL64) &
                 + in(i+2,j-4) * real(-0.0017857142857142857,REAL64) &
                 + in(i+3,j-4) * real(-0.0017857142857142857,REAL64) &
                 + in(i+4,j-4) * real(-0.0017857142857142857,REAL64) &
                 + in(i+5,j-4) * real(-0.0017857142857142857,REAL64) &
                 + in(i-3,j-3) * real(-0.016666666666666666,REAL64) &
                 + in(i+1,j-3) * real(-0.0033333333333333335,REAL64) &
                 + in(i+2,j-3) * real(-0.0033333333333333335,REAL64) &
                 + in(i+3,j-3) * real(-0.0033333333333333335,REAL64) &
                 + in(i+4,j-3) * real(-0.0033333333333333335,REAL64) &
                 + in(i+5,j-3) * real(-0.0033333333333333335,REAL64) &
                 + in(i-2,j-2) * real(-0.025,REAL64) &
                 + in(i+1,j-2) * real(-0.008333333333333333,REAL64) &
                 + in(i+2,j-2) * real(-0.008333333333333333,REAL64) &
                 + in(i+3,j-2) * real(-0.008333333333333333,REAL64) &
                 + in(i+4,j-2) * real(-0.008333333333333333,REAL64) &
                 + in(i+5,j-2) * real(-0.008333333333333333,REAL64) &
                 + in(i-1,j-1) * real(-0.05,REAL64) &
                 + in(i+1,j-1) * real(-0.05,REAL64) &
                 + in(i+2,j-1) * real(-0.05,REAL64) &
                 + in(i+3,j-1) * real(-0.05,REAL64) &
                 + in(i+4,j-1) * real(-0.05,REAL64) &
                 + in(i+5,j-1) * real(-0.05,REAL64) &
                 + in(i-5,j+1) * real(-0.0011111111111111111,REAL64) &
                 + in(i-4,j+1) * real(-0.0017857142857142857,REAL64) &
                 + in(i-3,j+1) * real(-0.0033333333333333335,REAL64) &
                 + in(i-2,j+1) * real(-0.008333333333333333,REAL64) &
                 + in(i-1,j+1) * real(-0.05,REAL64) &
                 + in(i+1,j+1) * real(0.05,REAL64) &
                 + in(i+2,j+1) * real(0.008333333333333333,REAL64) &
                 + in(i+3,j+1) * real(0.0033333333333333335,REAL64) &
                 + in(i+4,j+1) * real(0.0017857142857142857,REAL64) &
                 + in(i+5,j+1) * real(0.0011111111111111111,REAL64) &
                 + in(i-5,j+2) * real(-0.0011111111111111111,REAL64) &
                 + in(i-4,j+2) * real(-0.0017857142857142857,REAL64) &
                 + in(i-3,j+2) * real(-0.0033333333333333335,REAL64) &
                 + in(i-2,j+2) * real(-0.008333333333333333,REAL64) &
                 + in(i-1,j+2) * real(-0.05,REAL64) &
                 + in(i+1,j+2) * real(0.008333333333333333,REAL64) &
                 + in(i+2,j+2) * real(0.025,REAL64) &
                 + in(i+3,j+2) * real(0.0033333333333333335,REAL64) &
                 + in(i+4,j+2) * real(0.0017857142857142857,REAL64) &
                 + in(i+5,j+2) * real(0.0011111111111111111,REAL64) &
                 + in(i-5,j+3) * real(-0.0011111111111111111,REAL64) &
                 + in(i-4,j+3) * real(-0.0017857142857142857,REAL64) &
                 + in(i-3,j+3) * real(-0.0033333333333333335,REAL64) &
                 + in(i-2,j+3) * real(-0.008333333333333333,REAL64) &
                 + in(i-1,j+3) * real(-0.05,REAL64) &
                 + in(i+1,j+3) * real(0.0033333333333333335,REAL64) &
                 + in(i+2,j+3) * real(0.0033333333333333335,REAL64) &
                 + in(i+3,j+3) * real(0.016666666666666666,REAL64) &
                 + in(i+4,j+3) * real(0.0017857142857142857,REAL64) &
                 + in(i+5,j+3) * real(0.0011111111111111111,REAL64) &
                 + in(i-5,j+4) * real(-0.0011111111111111111,REAL64) &
                 + in(i-4,j+4) * real(-0.0017857142857142857,REAL64) &
                 + in(i-3,j+4) * real(-0.0033333333333333335,REAL64) &
                 + in(i-2,j+4) * real(-0.008333333333333333,REAL64) &
                 + in(i-1,j+4) * real(-0.05,REAL64) &
                 + in(i+1,j+4) * real(0.0017857142857142857,REAL64) &
                 + in(i+2,j+4) * real(0.0017857142857142857,REAL64) &
                 + in(i+3,j+4) * real(0.0017857142857142857,REAL64) &
                 + in(i+4,j+4) * real(0.0125,REAL64) &
                 + in(i+5,j+4) * real(0.0011111111111111111,REAL64) &
                 + in(i-5,j+5) * real(-0.0011111111111111111,REAL64) &
                 + in(i-4,j+5) * real(-0.0017857142857142857,REAL64) &
                 + in(i-3,j+5) * real(-0.0033333333333333335,REAL64) &
                 + in(i-2,j+5) * real(-0.008333333333333333,REAL64) &
                 + in(i-1,j+5) * real(-0.05,REAL64) &
                 + in(i+1,j+5) * real(0.0011111111111111111,REAL64) &
                 + in(i+2,j+5) * real(0.0011111111111111111,REAL64) &
                 + in(i+3,j+5) * real(0.0011111111111111111,REAL64) &
                 + in(i+4,j+5) * real(0.0011111111111111111,REAL64) &
                 + in(i+5,j+5) * real(0.01,REAL64) &
                 + real(0.0,REAL64)
      end do
      !$omp end simd
    end do
    !$omp end taskloop
end subroutine

subroutine grid6(n, in, out)
use iso_fortran_env
implicit none
integer(kind=INT32), intent(in) :: n
real(kind=REAL64), intent(in) :: in(n,n)
real(kind=REAL64), intent(inout) :: out(n,n)
integer(kind=INT32) :: i,j
    !$omp taskloop
    do i=6,n-6-1
      !$omp simd
      do j=6,n-6-1
        out(i,j) = out(i,j) &
                 + in(i-6,j-6) * real(-0.006944444444444444,REAL64) &
                 + in(i+1,j-6) * real(-0.0006313131313131314,REAL64) &
                 + in(i+2,j-6) * real(-0.0006313131313131314,REAL64) &
                 + in(i+3,j-6) * real(-0.0006313131313131314,REAL64) &
                 + in(i+4,j-6) * real(-0.0006313131313131314,REAL64) &
                 + in(i+5,j-6) * real(-0.0006313131313131314,REAL64) &
                 + in(i+6,j-6) * real(-0.0006313131313131314,REAL64) &
                 + in(i-5,j-5) * real(-0.008333333333333333,REAL64) &
                 + in(i+1,j-5) * real(-0.000925925925925926,REAL64) &
                 + in(i+2,j-5) * real(-0.000925925925925926,REAL64) &
                 + in(i+3,j-5) * real(-0.000925925925925926,REAL64) &
                 + in(i+4,j-5) * real(-0.000925925925925926,REAL64) &
                 + in(i+5,j-5) * real(-0.000925925925925926,REAL64) &
                 + in(i+6,j-5) * real(-0.000925925925925926,REAL64) &
                 + in(i-4,j-4) * real(-0.010416666666666666,REAL64) &
                 + in(i+1,j-4) * real(-0.001488095238095238,REAL64) &
                 + in(i+2,j-4) * real(-0.001488095238095238,REAL64) &
                 + in(i+3,j-4) * real(-0.001488095238095238,REAL64) &
                 + in(i+4,j-4) * real(-0.001488095238095238,REAL64) &
                 + in(i+5,j-4) * real(-0.001488095238095238,REAL64) &
                 + in(i+6,j-4) * real(-0.001488095238095238,REAL64) &
                 + in(i-3,j-3) * real(-0.013888888888888888,REAL64) &
                 + in(i+1,j-3) * real(-0.002777777777777778,REAL64) &
                 + in(i+2,j-3) * real(-0.002777777777777778,REAL64) &
                 + in(i+3,j-3) * real(-0.002777777777777778,REAL64) &
                 + in(i+4,j-3) * real(-0.002777777777777778,REAL64) &
                 + in(i+5,j-3) * real(-0.002777777777777778,REAL64) &
                 + in(i+6,j-3) * real(-0.002777777777777778,REAL64) &
                 + in(i-2,j-2) * real(-0.020833333333333332,REAL64) &
                 + in(i+1,j-2) * real(-0.006944444444444444,REAL64) &
                 + in(i+2,j-2) * real(-0.006944444444444444,REAL64) &
                 + in(i+3,j-2) * real(-0.006944444444444444,REAL64) &
                 + in(i+4,j-2) * real(-0.006944444444444444,REAL64) &
                 + in(i+5,j-2) * real(-0.006944444444444444,REAL64) &
                 + in(i+6,j-2) * real(-0.006944444444444444,REAL64) &
                 + in(i-1,j-1) * real(-0.041666666666666664,REAL64) &
                 + in(i+1,j-1) * real(-0.041666666666666664,REAL64) &
                 + in(i+2,j-1) * real(-0.041666666666666664,REAL64) &
                 + in(i+3,j-1) * real(-0.041666666666666664,REAL64) &
                 + in(i+4,j-1) * real(-0.041666666666666664,REAL64) &
                 + in(i+5,j-1) * real(-0.041666666666666664,REAL64) &
                 + in(i+6,j-1) * real(-0.041666666666666664,REAL64) &
                 + in(i-6,j+1) * real(-0.0006313131313131314,REAL64) &
                 + in(i-5,j+1) * real(-0.000925925925925926,REAL64) &
                 + in(i-4,j+1) * real(-0.001488095238095238,REAL64) &
                 + in(i-3,j+1) * real(-0.002777777777777778,REAL64) &
                 + in(i-2,j+1) * real(-0.006944444444444444,REAL64) &
                 + in(i-1,j+1) * real(-0.041666666666666664,REAL64) &
                 + in(i+1,j+1) * real(0.041666666666666664,REAL64) &
                 + in(i+2,j+1) * real(0.006944444444444444,REAL64) &
                 + in(i+3,j+1) * real(0.002777777777777778,REAL64) &
                 + in(i+4,j+1) * real(0.001488095238095238,REAL64) &
                 + in(i+5,j+1) * real(0.000925925925925926,REAL64) &
                 + in(i+6,j+1) * real(0.0006313131313131314,REAL64) &
                 + in(i-6,j+2) * real(-0.0006313131313131314,REAL64) &
                 + in(i-5,j+2) * real(-0.000925925925925926,REAL64) &
                 + in(i-4,j+2) * real(-0.001488095238095238,REAL64) &
                 + in(i-3,j+2) * real(-0.002777777777777778,REAL64) &
                 + in(i-2,j+2) * real(-0.006944444444444444,REAL64) &
                 + in(i-1,j+2) * real(-0.041666666666666664,REAL64) &
                 + in(i+1,j+2) * real(0.006944444444444444,REAL64) &
                 + in(i+2,j+2) * real(0.020833333333333332,REAL64) &
                 + in(i+3,j+2) * real(0.002777777777777778,REAL64) &
                 + in(i+4,j+2) * real(0.001488095238095238,REAL64) &
                 + in(i+5,j+2) * real(0.000925925925925926,REAL64) &
                 + in(i+6,j+2) * real(0.0006313131313131314,REAL64) &
                 + in(i-6,j+3) * real(-0.0006313131313131314,REAL64) &
                 + in(i-5,j+3) * real(-0.000925925925925926,REAL64) &
                 + in(i-4,j+3) * real(-0.001488095238095238,REAL64) &
                 + in(i-3,j+3) * real(-0.002777777777777778,REAL64) &
                 + in(i-2,j+3) * real(-0.006944444444444444,REAL64) &
                 + in(i-1,j+3) * real(-0.041666666666666664,REAL64) &
                 + in(i+1,j+3) * real(0.002777777777777778,REAL64) &
                 + in(i+2,j+3) * real(0.002777777777777778,REAL64) &
                 + in(i+3,j+3) * real(0.013888888888888888,REAL64) &
                 + in(i+4,j+3) * real(0.001488095238095238,REAL64) &
                 + in(i+5,j+3) * real(0.000925925925925926,REAL64) &
                 + in(i+6,j+3) * real(0.0006313131313131314,REAL64) &
                 + in(i-6,j+4) * real(-0.0006313131313131314,REAL64) &
                 + in(i-5,j+4) * real(-0.000925925925925926,REAL64) &
                 + in(i-4,j+4) * real(-0.001488095238095238,REAL64) &
                 + in(i-3,j+4) * real(-0.002777777777777778,REAL64) &
                 + in(i-2,j+4) * real(-0.006944444444444444,REAL64) &
                 + in(i-1,j+4) * real(-0.041666666666666664,REAL64) &
                 + in(i+1,j+4) * real(0.001488095238095238,REAL64) &
                 + in(i+2,j+4) * real(0.001488095238095238,REAL64) &
                 + in(i+3,j+4) * real(0.001488095238095238,REAL64) &
                 + in(i+4,j+4) * real(0.010416666666666666,REAL64) &
                 + in(i+5,j+4) * real(0.000925925925925926,REAL64) &
                 + in(i+6,j+4) * real(0.0006313131313131314,REAL64) &
                 + in(i-6,j+5) * real(-0.0006313131313131314,REAL64) &
                 + in(i-5,j+5) * real(-0.000925925925925926,REAL64) &
                 + in(i-4,j+5) * real(-0.001488095238095238,REAL64) &
                 + in(i-3,j+5) * real(-0.002777777777777778,REAL64) &
                 + in(i-2,j+5) * real(-0.006944444444444444,REAL64) &
                 + in(i-1,j+5) * real(-0.041666666666666664,REAL64) &
                 + in(i+1,j+5) * real(0.000925925925925926,REAL64) &
                 + in(i+2,j+5) * real(0.000925925925925926,REAL64) &
                 + in(i+3,j+5) * real(0.000925925925925926,REAL64) &
                 + in(i+4,j+5) * real(0.000925925925925926,REAL64) &
                 + in(i+5,j+5) * real(0.008333333333333333,REAL64) &
                 + in(i+6,j+5) * real(0.0006313131313131314,REAL64) &
                 + in(i-6,j+6) * real(-0.0006313131313131314,REAL64) &
                 + in(i-5,j+6) * real(-0.000925925925925926,REAL64) &
                 + in(i-4,j+6) * real(-0.001488095238095238,REAL64) &
                 + in(i-3,j+6) * real(-0.002777777777777778,REAL64) &
                 + in(i-2,j+6) * real(-0.006944444444444444,REAL64) &
                 + in(i-1,j+6) * real(-0.041666666666666664,REAL64) &
                 + in(i+1,j+6) * real(0.0006313131313131314,REAL64) &
                 + in(i+2,j+6) * real(0.0006313131313131314,REAL64) &
                 + in(i+3,j+6) * real(0.0006313131313131314,REAL64) &
                 + in(i+4,j+6) * real(0.0006313131313131314,REAL64) &
                 + in(i+5,j+6) * real(0.0006313131313131314,REAL64) &
                 + in(i+6,j+6) * real(0.006944444444444444,REAL64) &
                 + real(0.0,REAL64)
      end do
      !$omp end simd
    end do
    !$omp end taskloop
end subroutine

subroutine grid7(n, in, out)
use iso_fortran_env
implicit none
integer(kind=INT32), intent(in) :: n
real(kind=REAL64), intent(in) :: in(n,n)
real(kind=REAL64), intent(inout) :: out(n,n)
integer(kind=INT32) :: i,j
    !$omp taskloop
    do i=7,n-7-1
      !$omp simd
      do j=7,n-7-1
        out(i,j) = out(i,j) &
                 + in(i-7,j-7) * real(-0.00510204081632653,REAL64) &
                 + in(i+1,j-7) * real(-0.0003924646781789639,REAL64) &
                 + in(i+2,j-7) * real(-0.0003924646781789639,REAL64) &
                 + in(i+3,j-7) * real(-0.0003924646781789639,REAL64) &
                 + in(i+4,j-7) * real(-0.0003924646781789639,REAL64) &
                 + in(i+5,j-7) * real(-0.0003924646781789639,REAL64) &
                 + in(i+6,j-7) * real(-0.0003924646781789639,REAL64) &
                 + in(i+7,j-7) * real(-0.0003924646781789639,REAL64) &
                 + in(i-6,j-6) * real(-0.005952380952380952,REAL64) &
                 + in(i+1,j-6) * real(-0.0005411255411255411,REAL64) &
                 + in(i+2,j-6) * real(-0.0005411255411255411,REAL64) &
                 + in(i+3,j-6) * real(-0.0005411255411255411,REAL64) &
                 + in(i+4,j-6) * real(-0.0005411255411255411,REAL64) &
                 + in(i+5,j-6) * real(-0.0005411255411255411,REAL64) &
                 + in(i+6,j-6) * real(-0.0005411255411255411,REAL64) &
                 + in(i+7,j-6) * real(-0.0005411255411255411,REAL64) &
                 + in(i-5,j-5) * real(-0.007142857142857143,REAL64) &
                 + in(i+1,j-5) * real(-0.0007936507936507937,REAL64) &
                 + in(i+2,j-5) * real(-0.0007936507936507937,REAL64) &
                 + in(i+3,j-5) * real(-0.0007936507936507937,REAL64) &
                 + in(i+4,j-5) * real(-0.0007936507936507937,REAL64) &
                 + in(i+5,j-5) * real(-0.0007936507936507937,REAL64) &
                 + in(i+6,j-5) * real(-0.0007936507936507937,REAL64) &
                 + in(i+7,j-5) * real(-0.0007936507936507937,REAL64) &
                 + in(i-4,j-4) * real(-0.008928571428571428,REAL64) &
                 + in(i+1,j-4) * real(-0.0012755102040816326,REAL64) &
                 + in(i+2,j-4) * real(-0.0012755102040816326,REAL64) &
                 + in(i+3,j-4) * real(-0.0012755102040816326,REAL64) &
                 + in(i+4,j-4) * real(-0.0012755102040816326,REAL64) &
                 + in(i+5,j-4) * real(-0.0012755102040816326,REAL64) &
                 + in(i+6,j-4) * real(-0.0012755102040816326,REAL64) &
                 + in(i+7,j-4) * real(-0.0012755102040816326,REAL64) &
                 + in(i-3,j-3) * real(-0.011904761904761904,REAL64) &
                 + in(i+1,j-3) * real(-0.002380952380952381,REAL64) &
                 + in(i+2,j-3) * real(-0.002380952380952381,REAL64) &
                 + in(i+3,j-3) * real(-0.002380952380952381,REAL64) &
                 + in(i+4,j-3) * real(-0.002380952380952381,REAL64) &
                 + in(i+5,j-3) * real(-0.002380952380952381,REAL64) &
                 + in(i+6,j-3) * real(-0.002380952380952381,REAL64) &
                 + in(i+7,j-3) * real(-0.002380952380952381,REAL64) &
                 + in(i-2,j-2) * real(-0.017857142857142856,REAL64) &
                 + in(i+1,j-2) * real(-0.005952380952380952,REAL64) &
                 + in(i+2,j-2) * real(-0.005952380952380952,REAL64) &
                 + in(i+3,j-2) * real(-0.005952380952380952,REAL64) &
                 + in(i+4,j-2) * real(-0.005952380952380952,REAL64) &
                 + in(i+5,j-2) * real(-0.005952380952380952,REAL64) &
                 + in(i+6,j-2) * real(-0.005952380952380952,REAL64) &
                 + in(i+7,j-2) * real(-0.005952380952380952,REAL64) &
                 + in(i-1,j-1) * real(-0.03571428571428571,REAL64) &
                 + in(i+1,j-1) * real(-0.03571428571428571,REAL64) &
                 + in(i+2,j-1) * real(-0.03571428571428571,REAL64) &
                 + in(i+3,j-1) * real(-0.03571428571428571,REAL64) &
                 + in(i+4,j-1) * real(-0.03571428571428571,REAL64) &
                 + in(i+5,j-1) * real(-0.03571428571428571,REAL64) &
                 + in(i+6,j-1) * real(-0.03571428571428571,REAL64) &
                 + in(i+7,j-1) * real(-0.03571428571428571,REAL64) &
                 + in(i-7,j+1) * real(-0.0003924646781789639,REAL64) &
                 + in(i-6,j+1) * real(-0.0005411255411255411,REAL64) &
                 + in(i-5,j+1) * real(-0.0007936507936507937,REAL64) &
                 + in(i-4,j+1) * real(-0.0012755102040816326,REAL64) &
                 + in(i-3,j+1) * real(-0.002380952380952381,REAL64) &
                 + in(i-2,j+1) * real(-0.005952380952380952,REAL64) &
                 + in(i-1,j+1) * real(-0.03571428571428571,REAL64) &
                 + in(i+1,j+1) * real(0.03571428571428571,REAL64) &
                 + in(i+2,j+1) * real(0.005952380952380952,REAL64) &
                 + in(i+3,j+1) * real(0.002380952380952381,REAL64) &
                 + in(i+4,j+1) * real(0.0012755102040816326,REAL64) &
                 + in(i+5,j+1) * real(0.0007936507936507937,REAL64) &
                 + in(i+6,j+1) * real(0.0005411255411255411,REAL64) &
                 + in(i+7,j+1) * real(0.0003924646781789639,REAL64) &
                 + in(i-7,j+2) * real(-0.0003924646781789639,REAL64) &
                 + in(i-6,j+2) * real(-0.0005411255411255411,REAL64) &
                 + in(i-5,j+2) * real(-0.0007936507936507937,REAL64) &
                 + in(i-4,j+2) * real(-0.0012755102040816326,REAL64) &
                 + in(i-3,j+2) * real(-0.002380952380952381,REAL64) &
                 + in(i-2,j+2) * real(-0.005952380952380952,REAL64) &
                 + in(i-1,j+2) * real(-0.03571428571428571,REAL64) &
                 + in(i+1,j+2) * real(0.005952380952380952,REAL64) &
                 + in(i+2,j+2) * real(0.017857142857142856,REAL64) &
                 + in(i+3,j+2) * real(0.002380952380952381,REAL64) &
                 + in(i+4,j+2) * real(0.0012755102040816326,REAL64) &
                 + in(i+5,j+2) * real(0.0007936507936507937,REAL64) &
                 + in(i+6,j+2) * real(0.0005411255411255411,REAL64) &
                 + in(i+7,j+2) * real(0.0003924646781789639,REAL64) &
                 + in(i-7,j+3) * real(-0.0003924646781789639,REAL64) &
                 + in(i-6,j+3) * real(-0.0005411255411255411,REAL64) &
                 + in(i-5,j+3) * real(-0.0007936507936507937,REAL64) &
                 + in(i-4,j+3) * real(-0.0012755102040816326,REAL64) &
                 + in(i-3,j+3) * real(-0.002380952380952381,REAL64) &
                 + in(i-2,j+3) * real(-0.005952380952380952,REAL64) &
                 + in(i-1,j+3) * real(-0.03571428571428571,REAL64) &
                 + in(i+1,j+3) * real(0.002380952380952381,REAL64) &
                 + in(i+2,j+3) * real(0.002380952380952381,REAL64) &
                 + in(i+3,j+3) * real(0.011904761904761904,REAL64) &
                 + in(i+4,j+3) * real(0.0012755102040816326,REAL64) &
                 + in(i+5,j+3) * real(0.0007936507936507937,REAL64) &
                 + in(i+6,j+3) * real(0.0005411255411255411,REAL64) &
                 + in(i+7,j+3) * real(0.0003924646781789639,REAL64) &
                 + in(i-7,j+4) * real(-0.0003924646781789639,REAL64) &
                 + in(i-6,j+4) * real(-0.0005411255411255411,REAL64) &
                 + in(i-5,j+4) * real(-0.0007936507936507937,REAL64) &
                 + in(i-4,j+4) * real(-0.0012755102040816326,REAL64) &
                 + in(i-3,j+4) * real(-0.002380952380952381,REAL64) &
                 + in(i-2,j+4) * real(-0.005952380952380952,REAL64) &
                 + in(i-1,j+4) * real(-0.03571428571428571,REAL64) &
                 + in(i+1,j+4) * real(0.0012755102040816326,REAL64) &
                 + in(i+2,j+4) * real(0.0012755102040816326,REAL64) &
                 + in(i+3,j+4) * real(0.0012755102040816326,REAL64) &
                 + in(i+4,j+4) * real(0.008928571428571428,REAL64) &
                 + in(i+5,j+4) * real(0.0007936507936507937,REAL64) &
                 + in(i+6,j+4) * real(0.0005411255411255411,REAL64) &
                 + in(i+7,j+4) * real(0.0003924646781789639,REAL64) &
                 + in(i-7,j+5) * real(-0.0003924646781789639,REAL64) &
                 + in(i-6,j+5) * real(-0.0005411255411255411,REAL64) &
                 + in(i-5,j+5) * real(-0.0007936507936507937,REAL64) &
                 + in(i-4,j+5) * real(-0.0012755102040816326,REAL64) &
                 + in(i-3,j+5) * real(-0.002380952380952381,REAL64) &
                 + in(i-2,j+5) * real(-0.005952380952380952,REAL64) &
                 + in(i-1,j+5) * real(-0.03571428571428571,REAL64) &
                 + in(i+1,j+5) * real(0.0007936507936507937,REAL64) &
                 + in(i+2,j+5) * real(0.0007936507936507937,REAL64) &
                 + in(i+3,j+5) * real(0.0007936507936507937,REAL64) &
                 + in(i+4,j+5) * real(0.0007936507936507937,REAL64) &
                 + in(i+5,j+5) * real(0.007142857142857143,REAL64) &
                 + in(i+6,j+5) * real(0.0005411255411255411,REAL64) &
                 + in(i+7,j+5) * real(0.0003924646781789639,REAL64) &
                 + in(i-7,j+6) * real(-0.0003924646781789639,REAL64) &
                 + in(i-6,j+6) * real(-0.0005411255411255411,REAL64) &
                 + in(i-5,j+6) * real(-0.0007936507936507937,REAL64) &
                 + in(i-4,j+6) * real(-0.0012755102040816326,REAL64) &
                 + in(i-3,j+6) * real(-0.002380952380952381,REAL64) &
                 + in(i-2,j+6) * real(-0.005952380952380952,REAL64) &
                 + in(i-1,j+6) * real(-0.03571428571428571,REAL64) &
                 + in(i+1,j+6) * real(0.0005411255411255411,REAL64) &
                 + in(i+2,j+6) * real(0.0005411255411255411,REAL64) &
                 + in(i+3,j+6) * real(0.0005411255411255411,REAL64) &
                 + in(i+4,j+6) * real(0.0005411255411255411,REAL64) &
                 + in(i+5,j+6) * real(0.0005411255411255411,REAL64) &
                 + in(i+6,j+6) * real(0.005952380952380952,REAL64) &
                 + in(i+7,j+6) * real(0.0003924646781789639,REAL64) &
                 + in(i-7,j+7) * real(-0.0003924646781789639,REAL64) &
                 + in(i-6,j+7) * real(-0.0005411255411255411,REAL64) &
                 + in(i-5,j+7) * real(-0.0007936507936507937,REAL64) &
                 + in(i-4,j+7) * real(-0.0012755102040816326,REAL64) &
                 + in(i-3,j+7) * real(-0.002380952380952381,REAL64) &
                 + in(i-2,j+7) * real(-0.005952380952380952,REAL64) &
                 + in(i-1,j+7) * real(-0.03571428571428571,REAL64) &
                 + in(i+1,j+7) * real(0.0003924646781789639,REAL64) &
                 + in(i+2,j+7) * real(0.0003924646781789639,REAL64) &
                 + in(i+3,j+7) * real(0.0003924646781789639,REAL64) &
                 + in(i+4,j+7) * real(0.0003924646781789639,REAL64) &
                 + in(i+5,j+7) * real(0.0003924646781789639,REAL64) &
                 + in(i+6,j+7) * real(0.0003924646781789639,REAL64) &
                 + in(i+7,j+7) * real(0.00510204081632653,REAL64) &
                 + real(0.0,REAL64)
      end do
      !$omp end simd
    end do
    !$omp end taskloop
end subroutine

subroutine grid8(n, in, out)
use iso_fortran_env
implicit none
integer(kind=INT32), intent(in) :: n
real(kind=REAL64), intent(in) :: in(n,n)
real(kind=REAL64), intent(inout) :: out(n,n)
integer(kind=INT32) :: i,j
    !$omp taskloop
    do i=8,n-8-1
      !$omp simd
      do j=8,n-8-1
        out(i,j) = out(i,j) &
                 + in(i-8,j-8) * real(-0.00390625,REAL64) &
                 + in(i+1,j-8) * real(-0.00026041666666666666,REAL64) &
                 + in(i+2,j-8) * real(-0.00026041666666666666,REAL64) &
                 + in(i+3,j-8) * real(-0.00026041666666666666,REAL64) &
                 + in(i+4,j-8) * real(-0.00026041666666666666,REAL64) &
                 + in(i+5,j-8) * real(-0.00026041666666666666,REAL64) &
                 + in(i+6,j-8) * real(-0.00026041666666666666,REAL64) &
                 + in(i+7,j-8) * real(-0.00026041666666666666,REAL64) &
                 + in(i+8,j-8) * real(-0.00026041666666666666,REAL64) &
                 + in(i-7,j-7) * real(-0.004464285714285714,REAL64) &
                 + in(i+1,j-7) * real(-0.00034340659340659343,REAL64) &
                 + in(i+2,j-7) * real(-0.00034340659340659343,REAL64) &
                 + in(i+3,j-7) * real(-0.00034340659340659343,REAL64) &
                 + in(i+4,j-7) * real(-0.00034340659340659343,REAL64) &
                 + in(i+5,j-7) * real(-0.00034340659340659343,REAL64) &
                 + in(i+6,j-7) * real(-0.00034340659340659343,REAL64) &
                 + in(i+7,j-7) * real(-0.00034340659340659343,REAL64) &
                 + in(i+8,j-7) * real(-0.00034340659340659343,REAL64) &
                 + in(i-6,j-6) * real(-0.005208333333333333,REAL64) &
                 + in(i+1,j-6) * real(-0.0004734848484848485,REAL64) &
                 + in(i+2,j-6) * real(-0.0004734848484848485,REAL64) &
                 + in(i+3,j-6) * real(-0.0004734848484848485,REAL64) &
                 + in(i+4,j-6) * real(-0.0004734848484848485,REAL64) &
                 + in(i+5,j-6) * real(-0.0004734848484848485,REAL64) &
                 + in(i+6,j-6) * real(-0.0004734848484848485,REAL64) &
                 + in(i+7,j-6) * real(-0.0004734848484848485,REAL64) &
                 + in(i+8,j-6) * real(-0.0004734848484848485,REAL64) &
                 + in(i-5,j-5) * real(-0.00625,REAL64) &
                 + in(i+1,j-5) * real(-0.0006944444444444445,REAL64) &
                 + in(i+2,j-5) * real(-0.0006944444444444445,REAL64) &
                 + in(i+3,j-5) * real(-0.0006944444444444445,REAL64) &
                 + in(i+4,j-5) * real(-0.0006944444444444445,REAL64) &
                 + in(i+5,j-5) * real(-0.0006944444444444445,REAL64) &
                 + in(i+6,j-5) * real(-0.0006944444444444445,REAL64) &
                 + in(i+7,j-5) * real(-0.0006944444444444445,REAL64) &
                 + in(i+8,j-5) * real(-0.0006944444444444445,REAL64) &
                 + in(i-4,j-4) * real(-0.0078125,REAL64) &
                 + in(i+1,j-4) * real(-0.0011160714285714285,REAL64) &
                 + in(i+2,j-4) * real(-0.0011160714285714285,REAL64) &
                 + in(i+3,j-4) * real(-0.0011160714285714285,REAL64) &
                 + in(i+4,j-4) * real(-0.0011160714285714285,REAL64) &
                 + in(i+5,j-4) * real(-0.0011160714285714285,REAL64) &
                 + in(i+6,j-4) * real(-0.0011160714285714285,REAL64) &
                 + in(i+7,j-4) * real(-0.0011160714285714285,REAL64) &
                 + in(i+8,j-4) * real(-0.0011160714285714285,REAL64) &
                 + in(i-3,j-3) * real(-0.010416666666666666,REAL64) &
                 + in(i+1,j-3) * real(-0.0020833333333333333,REAL64) &
                 + in(i+2,j-3) * real(-0.0020833333333333333,REAL64) &
                 + in(i+3,j-3) * real(-0.0020833333333333333,REAL64) &
                 + in(i+4,j-3) * real(-0.0020833333333333333,REAL64) &
                 + in(i+5,j-3) * real(-0.0020833333333333333,REAL64) &
                 + in(i+6,j-3) * real(-0.0020833333333333333,REAL64) &
                 + in(i+7,j-3) * real(-0.0020833333333333333,REAL64) &
                 + in(i+8,j-3) * real(-0.0020833333333333333,REAL64) &
                 + in(i-2,j-2) * real(-0.015625,REAL64) &
                 + in(i+1,j-2) * real(-0.005208333333333333,REAL64) &
                 + in(i+2,j-2) * real(-0.005208333333333333,REAL64) &
                 + in(i+3,j-2) * real(-0.005208333333333333,REAL64) &
                 + in(i+4,j-2) * real(-0.005208333333333333,REAL64) &
                 + in(i+5,j-2) * real(-0.005208333333333333,REAL64) &
                 + in(i+6,j-2) * real(-0.005208333333333333,REAL64) &
                 + in(i+7,j-2) * real(-0.005208333333333333,REAL64) &
                 + in(i+8,j-2) * real(-0.005208333333333333,REAL64) &
                 + in(i-1,j-1) * real(-0.03125,REAL64) &
                 + in(i+1,j-1) * real(-0.03125,REAL64) &
                 + in(i+2,j-1) * real(-0.03125,REAL64) &
                 + in(i+3,j-1) * real(-0.03125,REAL64) &
                 + in(i+4,j-1) * real(-0.03125,REAL64) &
                 + in(i+5,j-1) * real(-0.03125,REAL64) &
                 + in(i+6,j-1) * real(-0.03125,REAL64) &
                 + in(i+7,j-1) * real(-0.03125,REAL64) &
                 + in(i+8,j-1) * real(-0.03125,REAL64) &
                 + in(i-8,j+1) * real(-0.00026041666666666666,REAL64) &
                 + in(i-7,j+1) * real(-0.00034340659340659343,REAL64) &
                 + in(i-6,j+1) * real(-0.0004734848484848485,REAL64) &
                 + in(i-5,j+1) * real(-0.0006944444444444445,REAL64) &
                 + in(i-4,j+1) * real(-0.0011160714285714285,REAL64) &
                 + in(i-3,j+1) * real(-0.0020833333333333333,REAL64) &
                 + in(i-2,j+1) * real(-0.005208333333333333,REAL64) &
                 + in(i-1,j+1) * real(-0.03125,REAL64) &
                 + in(i+1,j+1) * real(0.03125,REAL64) &
                 + in(i+2,j+1) * real(0.005208333333333333,REAL64) &
                 + in(i+3,j+1) * real(0.0020833333333333333,REAL64) &
                 + in(i+4,j+1) * real(0.0011160714285714285,REAL64) &
                 + in(i+5,j+1) * real(0.0006944444444444445,REAL64) &
                 + in(i+6,j+1) * real(0.0004734848484848485,REAL64) &
                 + in(i+7,j+1) * real(0.00034340659340659343,REAL64) &
                 + in(i+8,j+1) * real(0.00026041666666666666,REAL64) &
                 + in(i-8,j+2) * real(-0.00026041666666666666,REAL64) &
                 + in(i-7,j+2) * real(-0.00034340659340659343,REAL64) &
                 + in(i-6,j+2) * real(-0.0004734848484848485,REAL64) &
                 + in(i-5,j+2) * real(-0.0006944444444444445,REAL64) &
                 + in(i-4,j+2) * real(-0.0011160714285714285,REAL64) &
                 + in(i-3,j+2) * real(-0.0020833333333333333,REAL64) &
                 + in(i-2,j+2) * real(-0.005208333333333333,REAL64) &
                 + in(i-1,j+2) * real(-0.03125,REAL64) &
                 + in(i+1,j+2) * real(0.005208333333333333,REAL64) &
                 + in(i+2,j+2) * real(0.015625,REAL64) &
                 + in(i+3,j+2) * real(0.0020833333333333333,REAL64) &
                 + in(i+4,j+2) * real(0.0011160714285714285,REAL64) &
                 + in(i+5,j+2) * real(0.0006944444444444445,REAL64) &
                 + in(i+6,j+2) * real(0.0004734848484848485,REAL64) &
                 + in(i+7,j+2) * real(0.00034340659340659343,REAL64) &
                 + in(i+8,j+2) * real(0.00026041666666666666,REAL64) &
                 + in(i-8,j+3) * real(-0.00026041666666666666,REAL64) &
                 + in(i-7,j+3) * real(-0.00034340659340659343,REAL64) &
                 + in(i-6,j+3) * real(-0.0004734848484848485,REAL64) &
                 + in(i-5,j+3) * real(-0.0006944444444444445,REAL64) &
                 + in(i-4,j+3) * real(-0.0011160714285714285,REAL64) &
                 + in(i-3,j+3) * real(-0.0020833333333333333,REAL64) &
                 + in(i-2,j+3) * real(-0.005208333333333333,REAL64) &
                 + in(i-1,j+3) * real(-0.03125,REAL64) &
                 + in(i+1,j+3) * real(0.0020833333333333333,REAL64) &
                 + in(i+2,j+3) * real(0.0020833333333333333,REAL64) &
                 + in(i+3,j+3) * real(0.010416666666666666,REAL64) &
                 + in(i+4,j+3) * real(0.0011160714285714285,REAL64) &
                 + in(i+5,j+3) * real(0.0006944444444444445,REAL64) &
                 + in(i+6,j+3) * real(0.0004734848484848485,REAL64) &
                 + in(i+7,j+3) * real(0.00034340659340659343,REAL64) &
                 + in(i+8,j+3) * real(0.00026041666666666666,REAL64) &
                 + in(i-8,j+4) * real(-0.00026041666666666666,REAL64) &
                 + in(i-7,j+4) * real(-0.00034340659340659343,REAL64) &
                 + in(i-6,j+4) * real(-0.0004734848484848485,REAL64) &
                 + in(i-5,j+4) * real(-0.0006944444444444445,REAL64) &
                 + in(i-4,j+4) * real(-0.0011160714285714285,REAL64) &
                 + in(i-3,j+4) * real(-0.0020833333333333333,REAL64) &
                 + in(i-2,j+4) * real(-0.005208333333333333,REAL64) &
                 + in(i-1,j+4) * real(-0.03125,REAL64) &
                 + in(i+1,j+4) * real(0.0011160714285714285,REAL64) &
                 + in(i+2,j+4) * real(0.0011160714285714285,REAL64) &
                 + in(i+3,j+4) * real(0.0011160714285714285,REAL64) &
                 + in(i+4,j+4) * real(0.0078125,REAL64) &
                 + in(i+5,j+4) * real(0.0006944444444444445,REAL64) &
                 + in(i+6,j+4) * real(0.0004734848484848485,REAL64) &
                 + in(i+7,j+4) * real(0.00034340659340659343,REAL64) &
                 + in(i+8,j+4) * real(0.00026041666666666666,REAL64) &
                 + in(i-8,j+5) * real(-0.00026041666666666666,REAL64) &
                 + in(i-7,j+5) * real(-0.00034340659340659343,REAL64) &
                 + in(i-6,j+5) * real(-0.0004734848484848485,REAL64) &
                 + in(i-5,j+5) * real(-0.0006944444444444445,REAL64) &
                 + in(i-4,j+5) * real(-0.0011160714285714285,REAL64) &
                 + in(i-3,j+5) * real(-0.0020833333333333333,REAL64) &
                 + in(i-2,j+5) * real(-0.005208333333333333,REAL64) &
                 + in(i-1,j+5) * real(-0.03125,REAL64) &
                 + in(i+1,j+5) * real(0.0006944444444444445,REAL64) &
                 + in(i+2,j+5) * real(0.0006944444444444445,REAL64) &
                 + in(i+3,j+5) * real(0.0006944444444444445,REAL64) &
                 + in(i+4,j+5) * real(0.0006944444444444445,REAL64) &
                 + in(i+5,j+5) * real(0.00625,REAL64) &
                 + in(i+6,j+5) * real(0.0004734848484848485,REAL64) &
                 + in(i+7,j+5) * real(0.00034340659340659343,REAL64) &
                 + in(i+8,j+5) * real(0.00026041666666666666,REAL64) &
                 + in(i-8,j+6) * real(-0.00026041666666666666,REAL64) &
                 + in(i-7,j+6) * real(-0.00034340659340659343,REAL64) &
                 + in(i-6,j+6) * real(-0.0004734848484848485,REAL64) &
                 + in(i-5,j+6) * real(-0.0006944444444444445,REAL64) &
                 + in(i-4,j+6) * real(-0.0011160714285714285,REAL64) &
                 + in(i-3,j+6) * real(-0.0020833333333333333,REAL64) &
                 + in(i-2,j+6) * real(-0.005208333333333333,REAL64) &
                 + in(i-1,j+6) * real(-0.03125,REAL64) &
                 + in(i+1,j+6) * real(0.0004734848484848485,REAL64) &
                 + in(i+2,j+6) * real(0.0004734848484848485,REAL64) &
                 + in(i+3,j+6) * real(0.0004734848484848485,REAL64) &
                 + in(i+4,j+6) * real(0.0004734848484848485,REAL64) &
                 + in(i+5,j+6) * real(0.0004734848484848485,REAL64) &
                 + in(i+6,j+6) * real(0.005208333333333333,REAL64) &
                 + in(i+7,j+6) * real(0.00034340659340659343,REAL64) &
                 + in(i+8,j+6) * real(0.00026041666666666666,REAL64) &
                 + in(i-8,j+7) * real(-0.00026041666666666666,REAL64) &
                 + in(i-7,j+7) * real(-0.00034340659340659343,REAL64) &
                 + in(i-6,j+7) * real(-0.0004734848484848485,REAL64) &
                 + in(i-5,j+7) * real(-0.0006944444444444445,REAL64) &
                 + in(i-4,j+7) * real(-0.0011160714285714285,REAL64) &
                 + in(i-3,j+7) * real(-0.0020833333333333333,REAL64) &
                 + in(i-2,j+7) * real(-0.005208333333333333,REAL64) &
                 + in(i-1,j+7) * real(-0.03125,REAL64) &
                 + in(i+1,j+7) * real(0.00034340659340659343,REAL64) &
                 + in(i+2,j+7) * real(0.00034340659340659343,REAL64) &
                 + in(i+3,j+7) * real(0.00034340659340659343,REAL64) &
                 + in(i+4,j+7) * real(0.00034340659340659343,REAL64) &
                 + in(i+5,j+7) * real(0.00034340659340659343,REAL64) &
                 + in(i+6,j+7) * real(0.00034340659340659343,REAL64) &
                 + in(i+7,j+7) * real(0.004464285714285714,REAL64) &
                 + in(i+8,j+7) * real(0.00026041666666666666,REAL64) &
                 + in(i-8,j+8) * real(-0.00026041666666666666,REAL64) &
                 + in(i-7,j+8) * real(-0.00034340659340659343,REAL64) &
                 + in(i-6,j+8) * real(-0.0004734848484848485,REAL64) &
                 + in(i-5,j+8) * real(-0.0006944444444444445,REAL64) &
                 + in(i-4,j+8) * real(-0.0011160714285714285,REAL64) &
                 + in(i-3,j+8) * real(-0.0020833333333333333,REAL64) &
                 + in(i-2,j+8) * real(-0.005208333333333333,REAL64) &
                 + in(i-1,j+8) * real(-0.03125,REAL64) &
                 + in(i+1,j+8) * real(0.00026041666666666666,REAL64) &
                 + in(i+2,j+8) * real(0.00026041666666666666,REAL64) &
                 + in(i+3,j+8) * real(0.00026041666666666666,REAL64) &
                 + in(i+4,j+8) * real(0.00026041666666666666,REAL64) &
                 + in(i+5,j+8) * real(0.00026041666666666666,REAL64) &
                 + in(i+6,j+8) * real(0.00026041666666666666,REAL64) &
                 + in(i+7,j+8) * real(0.00026041666666666666,REAL64) &
                 + in(i+8,j+8) * real(0.00390625,REAL64) &
                 + real(0.0,REAL64)
      end do
      !$omp end simd
    end do
    !$omp end taskloop
end subroutine

subroutine grid9(n, in, out)
use iso_fortran_env
implicit none
integer(kind=INT32), intent(in) :: n
real(kind=REAL64), intent(in) :: in(n,n)
real(kind=REAL64), intent(inout) :: out(n,n)
integer(kind=INT32) :: i,j
    !$omp taskloop
    do i=9,n-9-1
      !$omp simd
      do j=9,n-9-1
        out(i,j) = out(i,j) &
                 + in(i-9,j-9) * real(-0.0030864197530864196,REAL64) &
                 + in(i+1,j-9) * real(-0.00018155410312273057,REAL64) &
                 + in(i+2,j-9) * real(-0.00018155410312273057,REAL64) &
                 + in(i+3,j-9) * real(-0.00018155410312273057,REAL64) &
                 + in(i+4,j-9) * real(-0.00018155410312273057,REAL64) &
                 + in(i+5,j-9) * real(-0.00018155410312273057,REAL64) &
                 + in(i+6,j-9) * real(-0.00018155410312273057,REAL64) &
                 + in(i+7,j-9) * real(-0.00018155410312273057,REAL64) &
                 + in(i+8,j-9) * real(-0.00018155410312273057,REAL64) &
                 + in(i+9,j-9) * real(-0.00018155410312273057,REAL64) &
                 + in(i-8,j-8) * real(-0.003472222222222222,REAL64) &
                 + in(i+1,j-8) * real(-0.0002314814814814815,REAL64) &
                 + in(i+2,j-8) * real(-0.0002314814814814815,REAL64) &
                 + in(i+3,j-8) * real(-0.0002314814814814815,REAL64) &
                 + in(i+4,j-8) * real(-0.0002314814814814815,REAL64) &
                 + in(i+5,j-8) * real(-0.0002314814814814815,REAL64) &
                 + in(i+6,j-8) * real(-0.0002314814814814815,REAL64) &
                 + in(i+7,j-8) * real(-0.0002314814814814815,REAL64) &
                 + in(i+8,j-8) * real(-0.0002314814814814815,REAL64) &
                 + in(i+9,j-8) * real(-0.0002314814814814815,REAL64) &
                 + in(i-7,j-7) * real(-0.003968253968253968,REAL64) &
                 + in(i+1,j-7) * real(-0.00030525030525030525,REAL64) &
                 + in(i+2,j-7) * real(-0.00030525030525030525,REAL64) &
                 + in(i+3,j-7) * real(-0.00030525030525030525,REAL64) &
                 + in(i+4,j-7) * real(-0.00030525030525030525,REAL64) &
                 + in(i+5,j-7) * real(-0.00030525030525030525,REAL64) &
                 + in(i+6,j-7) * real(-0.00030525030525030525,REAL64) &
                 + in(i+7,j-7) * real(-0.00030525030525030525,REAL64) &
                 + in(i+8,j-7) * real(-0.00030525030525030525,REAL64) &
                 + in(i+9,j-7) * real(-0.00030525030525030525,REAL64) &
                 + in(i-6,j-6) * real(-0.004629629629629629,REAL64) &
                 + in(i+1,j-6) * real(-0.00042087542087542086,REAL64) &
                 + in(i+2,j-6) * real(-0.00042087542087542086,REAL64) &
                 + in(i+3,j-6) * real(-0.00042087542087542086,REAL64) &
                 + in(i+4,j-6) * real(-0.00042087542087542086,REAL64) &
                 + in(i+5,j-6) * real(-0.00042087542087542086,REAL64) &
                 + in(i+6,j-6) * real(-0.00042087542087542086,REAL64) &
                 + in(i+7,j-6) * real(-0.00042087542087542086,REAL64) &
                 + in(i+8,j-6) * real(-0.00042087542087542086,REAL64) &
                 + in(i+9,j-6) * real(-0.00042087542087542086,REAL64) &
                 + in(i-5,j-5) * real(-0.005555555555555556,REAL64) &
                 + in(i+1,j-5) * real(-0.0006172839506172839,REAL64) &
                 + in(i+2,j-5) * real(-0.0006172839506172839,REAL64) &
                 + in(i+3,j-5) * real(-0.0006172839506172839,REAL64) &
                 + in(i+4,j-5) * real(-0.0006172839506172839,REAL64) &
                 + in(i+5,j-5) * real(-0.0006172839506172839,REAL64) &
                 + in(i+6,j-5) * real(-0.0006172839506172839,REAL64) &
                 + in(i+7,j-5) * real(-0.0006172839506172839,REAL64) &
                 + in(i+8,j-5) * real(-0.0006172839506172839,REAL64) &
                 + in(i+9,j-5) * real(-0.0006172839506172839,REAL64) &
                 + in(i-4,j-4) * real(-0.006944444444444444,REAL64) &
                 + in(i+1,j-4) * real(-0.000992063492063492,REAL64) &
                 + in(i+2,j-4) * real(-0.000992063492063492,REAL64) &
                 + in(i+3,j-4) * real(-0.000992063492063492,REAL64) &
                 + in(i+4,j-4) * real(-0.000992063492063492,REAL64) &
                 + in(i+5,j-4) * real(-0.000992063492063492,REAL64) &
                 + in(i+6,j-4) * real(-0.000992063492063492,REAL64) &
                 + in(i+7,j-4) * real(-0.000992063492063492,REAL64) &
                 + in(i+8,j-4) * real(-0.000992063492063492,REAL64) &
                 + in(i+9,j-4) * real(-0.000992063492063492,REAL64) &
                 + in(i-3,j-3) * real(-0.009259259259259259,REAL64) &
                 + in(i+1,j-3) * real(-0.001851851851851852,REAL64) &
                 + in(i+2,j-3) * real(-0.001851851851851852,REAL64) &
                 + in(i+3,j-3) * real(-0.001851851851851852,REAL64) &
                 + in(i+4,j-3) * real(-0.001851851851851852,REAL64) &
                 + in(i+5,j-3) * real(-0.001851851851851852,REAL64) &
                 + in(i+6,j-3) * real(-0.001851851851851852,REAL64) &
                 + in(i+7,j-3) * real(-0.001851851851851852,REAL64) &
                 + in(i+8,j-3) * real(-0.001851851851851852,REAL64) &
                 + in(i+9,j-3) * real(-0.001851851851851852,REAL64) &
                 + in(i-2,j-2) * real(-0.013888888888888888,REAL64) &
                 + in(i+1,j-2) * real(-0.004629629629629629,REAL64) &
                 + in(i+2,j-2) * real(-0.004629629629629629,REAL64) &
                 + in(i+3,j-2) * real(-0.004629629629629629,REAL64) &
                 + in(i+4,j-2) * real(-0.004629629629629629,REAL64) &
                 + in(i+5,j-2) * real(-0.004629629629629629,REAL64) &
                 + in(i+6,j-2) * real(-0.004629629629629629,REAL64) &
                 + in(i+7,j-2) * real(-0.004629629629629629,REAL64) &
                 + in(i+8,j-2) * real(-0.004629629629629629,REAL64) &
                 + in(i+9,j-2) * real(-0.004629629629629629,REAL64) &
                 + in(i-1,j-1) * real(-0.027777777777777776,REAL64) &
                 + in(i+1,j-1) * real(-0.027777777777777776,REAL64) &
                 + in(i+2,j-1) * real(-0.027777777777777776,REAL64) &
                 + in(i+3,j-1) * real(-0.027777777777777776,REAL64) &
                 + in(i+4,j-1) * real(-0.027777777777777776,REAL64) &
                 + in(i+5,j-1) * real(-0.027777777777777776,REAL64) &
                 + in(i+6,j-1) * real(-0.027777777777777776,REAL64) &
                 + in(i+7,j-1) * real(-0.027777777777777776,REAL64) &
                 + in(i+8,j-1) * real(-0.027777777777777776,REAL64) &
                 + in(i+9,j-1) * real(-0.027777777777777776,REAL64) &
                 + in(i-9,j+1) * real(-0.00018155410312273057,REAL64) &
                 + in(i-8,j+1) * real(-0.0002314814814814815,REAL64) &
                 + in(i-7,j+1) * real(-0.00030525030525030525,REAL64) &
                 + in(i-6,j+1) * real(-0.00042087542087542086,REAL64) &
                 + in(i-5,j+1) * real(-0.0006172839506172839,REAL64) &
                 + in(i-4,j+1) * real(-0.000992063492063492,REAL64) &
                 + in(i-3,j+1) * real(-0.001851851851851852,REAL64) &
                 + in(i-2,j+1) * real(-0.004629629629629629,REAL64) &
                 + in(i-1,j+1) * real(-0.027777777777777776,REAL64) &
                 + in(i+1,j+1) * real(0.027777777777777776,REAL64) &
                 + in(i+2,j+1) * real(0.004629629629629629,REAL64) &
                 + in(i+3,j+1) * real(0.001851851851851852,REAL64) &
                 + in(i+4,j+1) * real(0.000992063492063492,REAL64) &
                 + in(i+5,j+1) * real(0.0006172839506172839,REAL64) &
                 + in(i+6,j+1) * real(0.00042087542087542086,REAL64) &
                 + in(i+7,j+1) * real(0.00030525030525030525,REAL64) &
                 + in(i+8,j+1) * real(0.0002314814814814815,REAL64) &
                 + in(i+9,j+1) * real(0.00018155410312273057,REAL64) &
                 + in(i-9,j+2) * real(-0.00018155410312273057,REAL64) &
                 + in(i-8,j+2) * real(-0.0002314814814814815,REAL64) &
                 + in(i-7,j+2) * real(-0.00030525030525030525,REAL64) &
                 + in(i-6,j+2) * real(-0.00042087542087542086,REAL64) &
                 + in(i-5,j+2) * real(-0.0006172839506172839,REAL64) &
                 + in(i-4,j+2) * real(-0.000992063492063492,REAL64) &
                 + in(i-3,j+2) * real(-0.001851851851851852,REAL64) &
                 + in(i-2,j+2) * real(-0.004629629629629629,REAL64) &
                 + in(i-1,j+2) * real(-0.027777777777777776,REAL64) &
                 + in(i+1,j+2) * real(0.004629629629629629,REAL64) &
                 + in(i+2,j+2) * real(0.013888888888888888,REAL64) &
                 + in(i+3,j+2) * real(0.001851851851851852,REAL64) &
                 + in(i+4,j+2) * real(0.000992063492063492,REAL64) &
                 + in(i+5,j+2) * real(0.0006172839506172839,REAL64) &
                 + in(i+6,j+2) * real(0.00042087542087542086,REAL64) &
                 + in(i+7,j+2) * real(0.00030525030525030525,REAL64) &
                 + in(i+8,j+2) * real(0.0002314814814814815,REAL64) &
                 + in(i+9,j+2) * real(0.00018155410312273057,REAL64) &
                 + in(i-9,j+3) * real(-0.00018155410312273057,REAL64) &
                 + in(i-8,j+3) * real(-0.0002314814814814815,REAL64) &
                 + in(i-7,j+3) * real(-0.00030525030525030525,REAL64) &
                 + in(i-6,j+3) * real(-0.00042087542087542086,REAL64) &
                 + in(i-5,j+3) * real(-0.0006172839506172839,REAL64) &
                 + in(i-4,j+3) * real(-0.000992063492063492,REAL64) &
                 + in(i-3,j+3) * real(-0.001851851851851852,REAL64) &
                 + in(i-2,j+3) * real(-0.004629629629629629,REAL64) &
                 + in(i-1,j+3) * real(-0.027777777777777776,REAL64) &
                 + in(i+1,j+3) * real(0.001851851851851852,REAL64) &
                 + in(i+2,j+3) * real(0.001851851851851852,REAL64) &
                 + in(i+3,j+3) * real(0.009259259259259259,REAL64) &
                 + in(i+4,j+3) * real(0.000992063492063492,REAL64) &
                 + in(i+5,j+3) * real(0.0006172839506172839,REAL64) &
                 + in(i+6,j+3) * real(0.00042087542087542086,REAL64) &
                 + in(i+7,j+3) * real(0.00030525030525030525,REAL64) &
                 + in(i+8,j+3) * real(0.0002314814814814815,REAL64) &
                 + in(i+9,j+3) * real(0.00018155410312273057,REAL64) &
                 + in(i-9,j+4) * real(-0.00018155410312273057,REAL64) &
                 + in(i-8,j+4) * real(-0.0002314814814814815,REAL64) &
                 + in(i-7,j+4) * real(-0.00030525030525030525,REAL64) &
                 + in(i-6,j+4) * real(-0.00042087542087542086,REAL64) &
                 + in(i-5,j+4) * real(-0.0006172839506172839,REAL64) &
                 + in(i-4,j+4) * real(-0.000992063492063492,REAL64) &
                 + in(i-3,j+4) * real(-0.001851851851851852,REAL64) &
                 + in(i-2,j+4) * real(-0.004629629629629629,REAL64) &
                 + in(i-1,j+4) * real(-0.027777777777777776,REAL64) &
                 + in(i+1,j+4) * real(0.000992063492063492,REAL64) &
                 + in(i+2,j+4) * real(0.000992063492063492,REAL64) &
                 + in(i+3,j+4) * real(0.000992063492063492,REAL64) &
                 + in(i+4,j+4) * real(0.006944444444444444,REAL64) &
                 + in(i+5,j+4) * real(0.0006172839506172839,REAL64) &
                 + in(i+6,j+4) * real(0.00042087542087542086,REAL64) &
                 + in(i+7,j+4) * real(0.00030525030525030525,REAL64) &
                 + in(i+8,j+4) * real(0.0002314814814814815,REAL64) &
                 + in(i+9,j+4) * real(0.00018155410312273057,REAL64) &
                 + in(i-9,j+5) * real(-0.00018155410312273057,REAL64) &
                 + in(i-8,j+5) * real(-0.0002314814814814815,REAL64) &
                 + in(i-7,j+5) * real(-0.00030525030525030525,REAL64) &
                 + in(i-6,j+5) * real(-0.00042087542087542086,REAL64) &
                 + in(i-5,j+5) * real(-0.0006172839506172839,REAL64) &
                 + in(i-4,j+5) * real(-0.000992063492063492,REAL64) &
                 + in(i-3,j+5) * real(-0.001851851851851852,REAL64) &
                 + in(i-2,j+5) * real(-0.004629629629629629,REAL64) &
                 + in(i-1,j+5) * real(-0.027777777777777776,REAL64) &
                 + in(i+1,j+5) * real(0.0006172839506172839,REAL64) &
                 + in(i+2,j+5) * real(0.0006172839506172839,REAL64) &
                 + in(i+3,j+5) * real(0.0006172839506172839,REAL64) &
                 + in(i+4,j+5) * real(0.0006172839506172839,REAL64) &
                 + in(i+5,j+5) * real(0.005555555555555556,REAL64) &
                 + in(i+6,j+5) * real(0.00042087542087542086,REAL64) &
                 + in(i+7,j+5) * real(0.00030525030525030525,REAL64) &
                 + in(i+8,j+5) * real(0.0002314814814814815,REAL64) &
                 + in(i+9,j+5) * real(0.00018155410312273057,REAL64) &
                 + in(i-9,j+6) * real(-0.00018155410312273057,REAL64) &
                 + in(i-8,j+6) * real(-0.0002314814814814815,REAL64) &
                 + in(i-7,j+6) * real(-0.00030525030525030525,REAL64) &
                 + in(i-6,j+6) * real(-0.00042087542087542086,REAL64) &
                 + in(i-5,j+6) * real(-0.0006172839506172839,REAL64) &
                 + in(i-4,j+6) * real(-0.000992063492063492,REAL64) &
                 + in(i-3,j+6) * real(-0.001851851851851852,REAL64) &
                 + in(i-2,j+6) * real(-0.004629629629629629,REAL64) &
                 + in(i-1,j+6) * real(-0.027777777777777776,REAL64) &
                 + in(i+1,j+6) * real(0.00042087542087542086,REAL64) &
                 + in(i+2,j+6) * real(0.00042087542087542086,REAL64) &
                 + in(i+3,j+6) * real(0.00042087542087542086,REAL64) &
                 + in(i+4,j+6) * real(0.00042087542087542086,REAL64) &
                 + in(i+5,j+6) * real(0.00042087542087542086,REAL64) &
                 + in(i+6,j+6) * real(0.004629629629629629,REAL64) &
                 + in(i+7,j+6) * real(0.00030525030525030525,REAL64) &
                 + in(i+8,j+6) * real(0.0002314814814814815,REAL64) &
                 + in(i+9,j+6) * real(0.00018155410312273057,REAL64) &
                 + in(i-9,j+7) * real(-0.00018155410312273057,REAL64) &
                 + in(i-8,j+7) * real(-0.0002314814814814815,REAL64) &
                 + in(i-7,j+7) * real(-0.00030525030525030525,REAL64) &
                 + in(i-6,j+7) * real(-0.00042087542087542086,REAL64) &
                 + in(i-5,j+7) * real(-0.0006172839506172839,REAL64) &
                 + in(i-4,j+7) * real(-0.000992063492063492,REAL64) &
                 + in(i-3,j+7) * real(-0.001851851851851852,REAL64) &
                 + in(i-2,j+7) * real(-0.004629629629629629,REAL64) &
                 + in(i-1,j+7) * real(-0.027777777777777776,REAL64) &
                 + in(i+1,j+7) * real(0.00030525030525030525,REAL64) &
                 + in(i+2,j+7) * real(0.00030525030525030525,REAL64) &
                 + in(i+3,j+7) * real(0.00030525030525030525,REAL64) &
                 + in(i+4,j+7) * real(0.00030525030525030525,REAL64) &
                 + in(i+5,j+7) * real(0.00030525030525030525,REAL64) &
                 + in(i+6,j+7) * real(0.00030525030525030525,REAL64) &
                 + in(i+7,j+7) * real(0.003968253968253968,REAL64) &
                 + in(i+8,j+7) * real(0.0002314814814814815,REAL64) &
                 + in(i+9,j+7) * real(0.00018155410312273057,REAL64) &
                 + in(i-9,j+8) * real(-0.00018155410312273057,REAL64) &
                 + in(i-8,j+8) * real(-0.0002314814814814815,REAL64) &
                 + in(i-7,j+8) * real(-0.00030525030525030525,REAL64) &
                 + in(i-6,j+8) * real(-0.00042087542087542086,REAL64) &
                 + in(i-5,j+8) * real(-0.0006172839506172839,REAL64) &
                 + in(i-4,j+8) * real(-0.000992063492063492,REAL64) &
                 + in(i-3,j+8) * real(-0.001851851851851852,REAL64) &
                 + in(i-2,j+8) * real(-0.004629629629629629,REAL64) &
                 + in(i-1,j+8) * real(-0.027777777777777776,REAL64) &
                 + in(i+1,j+8) * real(0.0002314814814814815,REAL64) &
                 + in(i+2,j+8) * real(0.0002314814814814815,REAL64) &
                 + in(i+3,j+8) * real(0.0002314814814814815,REAL64) &
                 + in(i+4,j+8) * real(0.0002314814814814815,REAL64) &
                 + in(i+5,j+8) * real(0.0002314814814814815,REAL64) &
                 + in(i+6,j+8) * real(0.0002314814814814815,REAL64) &
                 + in(i+7,j+8) * real(0.0002314814814814815,REAL64) &
                 + in(i+8,j+8) * real(0.003472222222222222,REAL64) &
                 + in(i+9,j+8) * real(0.00018155410312273057,REAL64) &
                 + in(i-9,j+9) * real(-0.00018155410312273057,REAL64) &
                 + in(i-8,j+9) * real(-0.0002314814814814815,REAL64) &
                 + in(i-7,j+9) * real(-0.00030525030525030525,REAL64) &
                 + in(i-6,j+9) * real(-0.00042087542087542086,REAL64) &
                 + in(i-5,j+9) * real(-0.0006172839506172839,REAL64) &
                 + in(i-4,j+9) * real(-0.000992063492063492,REAL64) &
                 + in(i-3,j+9) * real(-0.001851851851851852,REAL64) &
                 + in(i-2,j+9) * real(-0.004629629629629629,REAL64) &
                 + in(i-1,j+9) * real(-0.027777777777777776,REAL64) &
                 + in(i+1,j+9) * real(0.00018155410312273057,REAL64) &
                 + in(i+2,j+9) * real(0.00018155410312273057,REAL64) &
                 + in(i+3,j+9) * real(0.00018155410312273057,REAL64) &
                 + in(i+4,j+9) * real(0.00018155410312273057,REAL64) &
                 + in(i+5,j+9) * real(0.00018155410312273057,REAL64) &
                 + in(i+6,j+9) * real(0.00018155410312273057,REAL64) &
                 + in(i+7,j+9) * real(0.00018155410312273057,REAL64) &
                 + in(i+8,j+9) * real(0.00018155410312273057,REAL64) &
                 + in(i+9,j+9) * real(0.0030864197530864196,REAL64) &
                 + real(0.0,REAL64)
      end do
      !$omp end simd
    end do
    !$omp end taskloop
end subroutine

