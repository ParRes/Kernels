! Copyright (c) 2015, Intel Corporation
! Copyright (c) 2021, Thomas Hayward-Schneider
! 
! Redistribution and use in source and binary forms, with or without
! modification, are permitted provided that the following conditions
! are met:
! 
! * Redistributions of source code must retain the above copyright
!       notice, this list of conditions and the following disclaimer.
! * Redistributions in binary form must reproduce the above
!       copyright notice, this list of conditions and the following
!       disclaimer in the documentation and/or other materials provided
!       with the distribution.
! * Neither the name of Intel Corporation nor the names of its
!       contributors may be used to endorse or promote products
!       derived from this software without specific prior written
!       permission.
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
! 
! HISTORY: - Written by Evangelos Georganas, August 2015.
!          - RvdW: Refactored to make the code PRK conforming, December 2015
!          - TWHS: Converted from C to Fortran

#define REL_X 0.5
#define REL_Y 0.5
#define DT 1.0
#define Q 1.0
#define MASS_INV 1.0
#define SUCCESS 1
#define FAILURE 0
#define epsilon 1.e-5

#ifndef _OPENMP
function prk_get_wtime() result(t)
  use iso_fortran_env
  implicit none
  real(kind=REAL64) ::  t
  integer(kind=INT64) :: c, r
  call system_clock(count = c, count_rate = r)
  t = real(c,REAL64) / real(r,REAL64)
end function prk_get_wtime
#endif

program pic
  use, intrinsic :: ISO_FORTRAN_ENV, only : REAL64, REAL32, INT64, INT32
#ifdef _OPENMP
  use omp_lib
#endif
  implicit none
#ifndef _OPENMP
  real(kind=REAL64) :: prk_get_wtime
#endif

  type particle_t
    real(kind=REAL64) :: x, y, v_x, v_y, q, x0, y0
    integer(kind=INT64) :: k, m
  end type particle_t

  type particles_t
    real(kind=REAL64), allocatable, dimension(:) :: x, y, v_x, v_y, q, x0, y0
    integer(kind=INT64), allocatable, dimension(:) :: k, m
  end type particles_t
  enum, bind(C)
    enumerator :: GEOMETRIC, SINUSOIDAL, LINEAR, PATCH
  end enum

  real(kind=REAL64), allocatable, dimension(:,:) :: Qgrid
  type(particles_t):: particles
  integer(kind=INT64) :: L, n, k, m, init_mode
  integer(kind=INT64) :: ip, iterations, iter
  real(kind=REAL64) :: rho
  real(kind=REAL64) :: t0, pic_time
  character(len=32) :: argtmp
  integer :: arglen, err

  ! ********************************************************************
  ! read and test input parameters
  ! ********************************************************************

  write(*,'(a25)') 'Parallel Research Kernels'
#ifdef _OPENMP
  write(*,'(a47)') 'Fortran OpenMP PIC'
#else
  write(*,'(a47)') 'Fortran Serial PIC'
#endif

  if (command_argument_count().lt.6) then
    write(*,'(a17,i1)') 'argument count = ', command_argument_count()
    print*,"Usage: %s <#simulation steps> <grid size> <#particles> <k (particle charge semi-increment)>"
    print*,"<m (vertical particle velocity)>"
    print*,"          <init mode> <init parameters>]"
    print*,"   init mode 'GEOMETRIC'  parameters: <attenuation factor>"
    print*,"             'SINUSOIDAL' parameters: none"
    print*,"             'LINEAR'     parameters: <negative slope> <constant offset>"
    print*,"             'PATCH'      parameters: <xleft> <xright>  <ybottom> <ytop>"
    stop 1
  endif


  call get_command_argument(1,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i32)') iterations
  if (iterations .lt. 1) then
    write(*,'(a,i5)') 'ERROR: iterations must be >= 1 : ', iterations
    stop 1
  endif

  call get_command_argument(2,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i64)') L

  call get_command_argument(3,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i64)') n

  call get_command_argument(4,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i64)') k

  call get_command_argument(5,argtmp,arglen,err)
  if (err.eq.0) read(argtmp,'(i64)') m

  call get_command_argument(6,argtmp,arglen,err)
  if(trim(argtmp) == 'GEOMETRIC') then
    init_mode = GEOMETRIC
    if (command_argument_count().lt.7) then
      print *, 'Not enough arguments provided'
      stop
    endif
    call get_command_argument(7,argtmp,arglen,err)
    if (err.eq.0) read(argtmp,*) rho
  else
    print *, 'Only "GEOMETRIC" implemented'
    stop 1
  endif

  print *, 'Grid size                      = ', L
  print *, 'Number of particles requested  = ', n
  print *, 'Number of time steps           = ', iterations
  print *, 'Initialization mode            = ', init_mode
  select case(init_mode)
  case(GEOMETRIC)
    print *, '  Attenuation factor           = ', rho
  end select
  print *, 'Particle charge semi-increment = ', k
  print *, 'Vertical velocity              = ', m


  allocate(Qgrid(0:L+1,0:L+1))

  call init_grid(Qgrid, L)
  call alloc_particles(particles, n + n/5) ! 20% buffer
  call init_particles(particles, n, L, rho, k, m, init_mode)

  print *, 'Number of particles placed     = ', n

  block
    real(kind=REAL64)   :: fx, fy, ax, ay, xval, yval, vx, vy, qval
    do iter = 0_INT64, iterations
    if(iter == 1) then
#ifdef _OPENMP
      t0 = omp_get_wtime()
#else
      t0 = prk_get_wtime()
#endif
    endif
      !$omp parallel do private(xval, yval, vx, vy, qval, fx, fy, ax, ay)
      do ip=1,n
        xval = particles%x(ip)
        yval = particles%y(ip)
        vx   = particles%v_x(ip)
        vy   = particles%v_y(ip)
        qval = particles%q(ip)

        fx = 0.0_REAL64
        fy = 0.0_REAL64
        call computeTotalForce(xval, yval, qval, Qgrid, fx, fy)
        ax = fx * MASS_INV
        ay = fy * MASS_INV

        particles%x(ip) = mod(xval + vx*DT + 0.5*ax*DT**2 + real(L, kind=REAL64), real(L, kind=REAL64))
        particles%y(ip) = mod(yval + vy*DT + 0.5*ay*DT**2 + real(L, kind=REAL64), real(L, kind=REAL64))

        particles%v_x(ip) = particles%x(ip) + ax * DT
        particles%v_y(ip) = particles%y(ip) + ay * DT
      enddo
      !$omp end parallel do
    enddo
  end block
#ifdef _OPENMP
  pic_time = omp_get_wtime() - t0
#else
  pic_time = prk_get_wtime() - t0
#endif

  block
    integer :: correctness
    type(particle_t) :: part
    correctness = SUCCESS
    do ip=1,n
      part = singlePart(particles, ip)
      correctness = correctness * verifyParticle(part, iterations, Qgrid, L)
    enddo
    if(correctness == SUCCESS) then
      print *, "Solution validates"
      print *, 'Rate (Mparts/s):', 1.0e-6 * real(n*iterations)/pic_time
    endif
  end block

  contains

  subroutine init_grid(Qgrid, L)
    real(kind=REAL64), allocatable, dimension(:,:), intent(inout) :: Qgrid
    integer(kind=INT64), intent(in)    :: L
    integer(kind=INT64) :: x, y
    do x=1,L
      do y=1,L
        if(mod(x,2_INT64)==0) then
          Qgrid(x,y) = Q
        else
          Qgrid(x,y) = -Q
        endif
      enddo
    enddo
  end subroutine init_grid

  subroutine alloc_particles(parts, nalloc)
    type(particles_t), intent(inout) :: parts
    integer(kind=INT64), intent(in) :: nalloc
    allocate(parts%x(nalloc), parts%y(nalloc), parts%v_x(nalloc), parts%v_y(nalloc), &
             parts%q(nalloc), parts%x0(nalloc), parts%y0(nalloc))
    allocate(parts%k(nalloc), parts%m(nalloc))
  end subroutine alloc_particles

  subroutine init_particles(particles, n, L, rho, k, m, init_mode)
    type(particles_t), intent(inout) :: particles
    integer(kind=INT64), intent(inout) :: n
    integer(kind=INT64), intent(in)    :: L, k, m, init_mode
    real(kind=REAL64), intent(in)      :: rho
    real(kind=REAL64)      :: A
    integer(kind=INT64)    :: x, y, ip
    integer                :: ip_cell, actual_particles

    select case(init_mode)
    case(GEOMETRIC)
      A = real(n, kind=REAL64) * ((1.0 - rho) / (1.0 - rho**L)) / real(L, kind=REAL64)
    end select
    !n_placed = 0_INT64
    !do x=1,L
    !  do y=1,L
    !    n_placed = n_placed + random_draw(A * rho**x)
    !  enddo
    !enddo

    ip=1
    do x=1,L
      do y=1,L
        select case(init_mode)
        case(GEOMETRIC)
          !actual_particles = random_draw(A * rho**x)
          actual_particles = int(A * rho**x)
        end select
        do ip_cell=1,actual_particles
          particles%x(ip) = real(x, kind=REAL64) + REL_X
          particles%y(ip) = real(y, kind=REAL64) + REL_Y
          particles%k(ip) = k
          particles%m(ip) = m

          ip = ip+1
        enddo
      enddo
    enddo
    n = ip - 1

    call finish_distribution(n, particles)
  end subroutine init_particles

  subroutine finish_distribution(n, particles)
    integer(kind=INT64), intent(in) :: n
    type(particles_t), intent(inout) :: particles
    integer(kind=INT64) :: ip, x, q_sign
    real(kind=REAL64) :: xval, yval, relx, rely, r1_sq, r2_sq, cos_theta, cos_phi, base_charge
    
    do ip=1,n
      xval  = particles%x(ip)
      yval  = particles%y(ip)
      relx  = xval - int(xval)
      rely  = yval - int(yval)
      x     = nint(xval)
      r1_sq = relx**2 + rely**2
      r2_sq = (1.0_REAL64 - relx)**2 + rely**2
      if (r1_sq > 0) then
        cos_theta = relx/sqrt(r1_sq)
      else
        cos_theta = 0.0_REAL64
      endif
      cos_phi   = (1.0_REAL64 - relx)/sqrt(r2_sq)
      base_charge = 1.0_REAL64 / (DT**2 * Q * (cos_theta/r1_sq + cos_phi/r2_sq))
      particles%v_x(ip) = 0.0_REAL64
      particles%v_y(ip) = real(particles%m(ip), kind=REAL64) / DT
      q_sign = 2 * mod(x,2_INT64) - 1
      particles%q(ip) = real(q_sign * (2 * particles%k(ip) + 1), kind=REAL64) * base_charge
      particles%x0(ip) = xval
      particles%y0(ip) = yval
    enddo
  end subroutine finish_distribution

  subroutine computeTotalForce(xpart, ypart, qpart, Qgrid, fx, fy)
    real(kind=REAL64), intent(in) :: xpart, ypart, qpart 
    real(kind=REAL64), allocatable, dimension(:,:), intent(in) :: Qgrid
    real(kind=REAL64), intent(out) :: fx, fy
    integer(kind=INT64) :: x, y
    real(kind=REAL64) :: tmp_fx, tmp_fy, rel_x, rel_y, tmp_res_x, tmp_res_y
    tmp_res_x = 0.0_REAL64
    tmp_res_y = 0.0_REAL64

    x = int(xpart, kind=INT64)
    y = int(ypart, kind=INT64)
    rel_x = xpart - real(x, kind=REAL64)
    rel_y = ypart - real(y, kind=REAL64)

    ! Force from top left charge
    call computeCoulomb(rel_x, rel_y, qpart, Qgrid(y,x), tmp_fx, tmp_fy)
    tmp_res_x = tmp_res_x + tmp_fx
    tmp_res_y = tmp_res_y + tmp_fy

    ! Force from bottom left charge
    call computeCoulomb(rel_x, (1.0_REAL64-rel_y), qpart, Qgrid(y+1,x), tmp_fx, tmp_fy)
    tmp_res_x = tmp_res_x + tmp_fx
    tmp_res_y = tmp_res_y + tmp_fy

    ! Force from top right charge
    call computeCoulomb((1.0_REAL64-rel_x), rel_y, qpart, Qgrid(y,x+1), tmp_fx, tmp_fy)
    tmp_res_x = tmp_res_x + tmp_fx
    tmp_res_y = tmp_res_y + tmp_fy

    ! Force from top right charge
    call computeCoulomb((1.0_REAL64-rel_x), (1.0_REAL64-rel_y), qpart, Qgrid(y+1,x+1), tmp_fx, tmp_fy)
    tmp_res_x = tmp_res_x + tmp_fx
    tmp_res_y = tmp_res_y + tmp_fy

    fx = tmp_res_x
    fy = tmp_res_y
  end subroutine computeTotalForce

  subroutine computeCoulomb(x_dist, y_dist, q1, q2, fx, fy)
    real(kind=REAL64), intent(in) :: x_dist, y_dist, q1, q2
    real(kind=REAL64), intent(out) :: fx, fy
    real(kind=REAL64) :: r, r2, f_coulomb

    r2 = x_dist**2 + y_dist**2
    r = sqrt(r2)

    f_coulomb = q1 * q2 / r2
    fx = f_coulomb * x_dist/r
    fy = f_coulomb * y_dist/r
  end subroutine computeCoulomb

  integer function verifyParticle(part, iterations, Qgrid, L)
    type(particle_t), intent(in) :: part
    integer(kind=REAL64), intent(in) :: iterations, L
    real(kind=REAL64), allocatable, dimension(:,:), intent(in) :: Qgrid
    integer(kind=INT64) :: x, y
    real(kind=REAL64) :: x_final, y_final, x_periodic, y_periodic, disp

    x = nint(part%x0)
    y = nint(part%y0)

    disp = real((iterations+1) * (2 * part%k + 1), kind=REAL64)
    if (part%q * Qgrid(x,y) > 0.0) then
      x_final = part%x0 + disp
    else
      x_final = part%x0 - disp
    endif
    y_final = part%y0 + part%m*(iterations+1)

    x_periodic = mod(x_final + real(iterations+1, kind=REAL64) * (2*part%k + 1)*L, real(L, kind=REAL64))
    y_periodic = mod(y_final + real(iterations+1, kind=REAL64) * abs(part%m)*L, real(L, kind=REAL64))

    if (abs(part%x - x_periodic) > epsilon .or. abs(part%y - y_periodic) > epsilon) then
      verifyParticle = FAILURE
    else
      verifyParticle = SUCCESS
    endif
  end function verifyParticle

  function singlePart(particles, ip) result(part)
    type(particles_t), intent(in) :: particles
    integer(kind=INT64), intent(in) :: ip
    type(particle_t) :: part
    part%x   = particles%x(ip)
    part%y   = particles%y(ip)
    part%v_x = particles%v_x(ip)
    part%v_y = particles%v_y(ip)
    part%q   = particles%q(ip)
    part%m   = particles%m(ip)
    part%k   = particles%k(ip)
    part%x0  = particles%x0(ip)
    part%y0  = particles%y0(ip)
  end function singlePart
end program pic
