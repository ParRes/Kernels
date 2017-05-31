
///
/// Copyright (c) 2013, Intel Corporation
///
/// Redistribution and use in source and binary forms, with or without
/// modification, are permitted provided that the following conditions
/// are met:
///
/// * Redistributions of source code must retain the above copyright
///       notice, this list of conditions and the following disclaimer.
/// * Redistributions in binary form must reproduce the above
///       copyright notice, this list of conditions and the following
///       disclaimer in the documentation and/or other materials provided
///       with the distribution.
/// * Neither the name of Intel Corporation nor the names of its
///       contributors may be used to endorse or promote products
///       derived from this software without specific prior written
///       permission.
///
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
/// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
/// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
/// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
/// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
/// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
/// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
/// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
/// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
/// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
/// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
/// POSSIBILITY OF SUCH DAMAGE.

//////////////////////////////////////////////////////////////////////
///
/// NAME:    Stencil
///
/// PURPOSE: This program tests the efficiency with which a space-invariant,
///          linear, symmetric filter (stencil) can be applied to a square
///          grid or image.
///
/// USAGE:   The program takes as input the linear
///          dimension of the grid, and the number of iterations on the grid
///
///                <progname> <iterations> <grid size>
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// FUNCTIONS CALLED:
///
///          Other than standard C functions, the following functions are used in
///          this program:
///          wtime()
///
/// HISTORY: - Written by Rob Van der Wijngaart, February 2009.
///          - RvdW: Removed unrolling pragmas for clarity;
///            added constant to array "in" at end of each iteration to force
///            refreshing of neighbor data in parallel versions; August 2013
///          - C++11-ification by Jeff Hammond, May 2017.
///          - radiust port by Jeff Hammond, May 2017.
///
//////////////////////////////////////////////////////////////////////

use std::env;
use std::time::Instant;

fn help() {
  println!("Usage: <# iterations> <grid size>");
}

fn main()
{
  println!("Parallel Research Kernels");
  println!("radiust stencil execution on 2D grid");

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  let args : Vec<String> = env::args().collect();

  let iterations : u32;
  let n : u32;

  if args.len() == 3 {
    iterations = match args[1].parse() {
      Ok(n) => { n },
      Err(_) => { help(); return; },
    };
    n = match args[2].parse() {
      Ok(n) => { n },
      Err(_) => { help(); return; },
    };
  } else {
    help();
    return;
  }

  if iterations < 1 {
    println!("ERROR: iterations must be >= 1");
  }
  if n < 1 {
    println!("ERROR: grid dimension must be positive: {}", n);
  }

  // this is disgusting - surely there is a better way...
  let r : u32 =
      if cfg!(radius = "1") { 1 } else
      if cfg!(radius = "2") { 2 } else
      if cfg!(radius = "3") { 3 } else
      if cfg!(radius = "4") { 4 } else
      if cfg!(radius = "5") { 5 } else
      if cfg!(radius = "6") { 6 } else
      { 0 };

  // grid stencil (star is the default)
  let grid : bool = if cfg!(grid) { true } else { false };

  if r < 1 {
    println!("ERROR: Stencil radius {} should be positive ", r);
    return;
  } else if (2 * r + 1) > n {
    println!("ERROR: Stencil radius {} exceeds grid size {}", r, n);
    return;
  }

  println!("Grid size            = {}", n);
  println!("Radius of stencil    = {}", r);
  if grid {
    println!("Type of stencil      = grid");
  } else {
    println!("Type of stencil      = star");
  }
  println!("Data type            = double precision");
  println!("Compact representation of stencil loop body");
  println!("Number of iterations = {}",iterations);

  //////////////////////////////////////////////////////////////////////
  // Allocate space for the input and do the work
  //////////////////////////////////////////////////////////////////////

  let nelems : usize = (n as usize)*(n as usize);
  let mut a : Vec<f64> = vec![0.0; nelems as usize];
  let mut b : Vec<f64> = vec![0.0; nelems as usize];

  // ws of points in the stencil
  let wdim : u32 = (2 * r as u32) + 1;
  let welems : u32 = wdim*wdim;
  let mut w : Vec<f64> = vec![0.0; welems as usize];
  for jj in 0..r+1 {
    for ii in 0..r+1 {
      let offset : usize = (ii as usize) * (welems as usize) + (jj as usize);
      w[offset] = 0.0;
    }
  }

  // fill the stencil ws to reflect a discrete divergence operator
  let stencil_size : u32;
  if grid {
    stencil_size = 4*r+1;
    for ii in 1..r+1 {
      w[r][r+ii] = w[r+ii][r] =  1./(2*ii*r);
      w[r][r-ii] = w[r-ii][r] = -1./(2*ii*r);
    }
  } else {
    stencil_size = (2*r+1)*(2*r+1);
    for jj in 1..r+1 {
      for ii in -jj+1..jj {
        w[r+ii][r+jj] =  1./(4*jj*(2*jj-1)*r);
        w[r+ii][r-jj] = -1./(4*jj*(2*jj-1)*r);
        w[r+jj][r+ii] =  1./(4*jj*(2*jj-1)*r);
        w[r-jj][r+ii] = -1./(4*jj*(2*jj-1)*r);
      }
      w[r+jj][r+jj]   =  1./(4*jj*r);
      w[r-jj][r-jj]   = -1./(4*jj*r);
    }
  }

/*
  // interior of grid with respect to stencil
  size_t active_points = static_cast<size_t>(n-2*r)*static_cast<size_t>(n-2*r);

  // initialize the input and output arrays
  for (auto i=0; i<n; i++) {
    for (auto j=0; j<n; j++) {
      in[i*n+j] = static_cast<double>(i+j);
    }
  }
  for (auto i=r; i<n-r; i++) {
    for (auto j=r; j<n-r; j++) {
      out[i*n+j] = 0.0;
    }
  }

  auto stencil_time = 0.0;

  for (auto iter = 0; iter<=iterations; iter++) {

    // start timer after a warmup iteration
    if (iter==1) stencil_time = prk::wtime();

    // Apply the stencil operator
    for (auto i=r; i<n-r; i++) {
      for (auto j=r; j<n-r; j++) {
        #ifdef STAR
            for (auto jj=-r; jj<=r; jj++) {
              out[i*n+j] += w[r][r+jj]*in[i*n+j+jj];
            }
            for (auto ii=-r; ii<0; ii++) {
              out[i*n+j] += w[r+ii][r]*in[(i+ii)*n+j];
            }
            for (auto ii=1; ii<=r; ii++) {
              out[i*n+j] += w[r+ii][r]*in[(i+ii)*n+j];
            }
        #else
            for (auto ii=-r; ii<=r; ii++) {
              for (auto jj=-r; jj<=r; jj++) {
                out[i*n+j] += w[r+ii][r+jj]*in[(i+ii)*n+j+jj];
              }
            }
        #endif
      }
    }

    // add constant to solution to force refresh of neighbor data, if any
    std::transform(in.begin(), in.end(), in.begin(), [](double c) { return c+=1.0; });

  }
  stencil_time = prk::wtime() - stencil_time;

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  // compute L1 norm in parallel
  double norm = 0.0;
  for (auto i=r; i<n-r; i++) {
    for (auto j=r; j<n-r; j++) {
      norm += std::fabs(out[i*n+j]);
    }
  }
  norm /= active_points;

  // verify correctness
  const double epsilon = 1.0e-8;
  double reference_norm = 2.*(iterations+1.);
  if (std::fabs(norm-reference_norm) > epsilon) {
    println!("ERROR: L1 norm = {} Reference L1 norm = {}", norm, reference_norm);
    return 1;
  } else {
    println!("Solution validates");
    if cfg!(VERBOSE) {
      println!("L1 norm = {} Reference L1 norm = {}", norm, reference_norm);
    }
    size_t flops = (2L*(size_t)stencil_size+1L) * active_points;
    auto avgtime = stencil_time/iterations;
    println!("Rate (MFlops/s): {} Avg time (s): {}", 1.0e-6 * static_cast<double>(flops)/avgtime, avgtime);
  }

*/
}
