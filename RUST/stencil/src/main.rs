
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
///            added constant to array "a" at end of each iteration to force
///            refreshing of neighbor data in parallel versions; August 2013
///          - C++11-ification by Jeff Hammond, May 2017.
///          - Rust port by Jeff Hammond, May 2017.
///
//////////////////////////////////////////////////////////////////////

use std::env;
use std::time::{Instant,Duration};

fn help() {
  println!("Usage: <# iterations> <grid dimension> <radius>");
}

fn main()
{
  println!("Parallel Research Kernels");
  println!("Rust stencil execution on 2D grid");

  //////////////////////////////////////////////////////////////////////
  // Process and test input parameters
  //////////////////////////////////////////////////////////////////////

  let args : Vec<String> = env::args().collect();

  let iterations : usize;
  let n : usize;
  let r : usize;

  // This is a compile-time setting.
  // grid stencil (star is the default)
  let grid : bool = if cfg!(grid) { true } else { false };

  // I have failed to make this a compile-time setting.
  /*
  let r : usize =
      if cfg!(radius = "1") { 1 } else
      if cfg!(radius = "2") { 2 } else
      if cfg!(radius = "3") { 3 } else
      if cfg!(radius = "4") { 4 } else
      if cfg!(radius = "5") { 5 } else
      if cfg!(radius = "6") { 6 } else
      { println!("FAIL"); 0 };
  */

  if args.len() == 4 {
    iterations = match args[1].parse() {
      Ok(n) => { n },
      Err(_) => { help(); return; },
    };
    n = match args[2].parse() {
      Ok(n) => { n },
      Err(_) => { help(); return; },
    };
    r = match args[3].parse() {
      Ok(n) => { n },
      Err(_) => { 2 },
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

  // input and output arrays
  let mut a : Vec<Vec<f64>> = vec![vec![0.0; n]; n];
  let mut b : Vec<Vec<f64>> = vec![vec![0.0; n]; n];

  // weights of points a the stencil
  let wdim : usize = 2 * r + 1;
  let mut w : Vec<Vec<f64>> = vec![vec![0.0; wdim]; wdim];

  // fill the stencil ws to reflect a discrete divergence operator
  let stencil_size : usize;
  if grid {
    stencil_size = (2*r+1)*(2*r+1);
    for j in 1..r+1 {
      for i in 1-j..j {
        let denom : f64 = (4*j*(2*j-1)*r) as f64;
        w[r+i][r+j] =  1./denom;
        w[r+i][r-j] = -1./denom;
        w[r+j][r+i] =  1./denom;
        w[r-j][r+i] = -1./denom;
      }
      let denom : f64 = (4*j*r) as f64;
      w[r+j][r+j]   =  1./denom;
      w[r-j][r-j]   = -1./denom;
    }
  }  else /* star */ {
    stencil_size = 4*r+1;
    for i in 1..r+1 {
      let denom : f64 = (2 * i * r) as f64;
      w[r][r+i] =  1./denom;
      w[r][r-i] = -1./denom;
      w[r+i][r] =  1./denom;
      w[r-i][r] = -1./denom;
    }
  }

  // interior of grid with respect to stencil
  let active_points : usize = (n-2*r)*(n-2*r);

  // initialize the input and output arrays
  for j in 0..n {
    for i in 0..n {
      a[i][j] = (i+j) as f64;
      b[i][j] = 0.0;
    }
  }

  let timer = Instant::now();
  let mut t0 : Duration = timer.elapsed();

  for k in 0..iterations+1 {

    if k == 1 { t0 = timer.elapsed(); }

    // Apply the stencil operator
    for i in r..n-r {
      for j in r..n-r {
        if grid {
          for ii in 0-r..r+1 {
            for jj in 0-r..r+1 {
              b[i][j] += w[r+ii][r+jj]*a[i+ii][j+jj];
            }
          }
        } else {
          b[i][j] += w[r][r]*a[i][j];
          for jj in r..0 {
            b[i][j] += w[r][r-jj]*a[i][j-jj];
          }
          for jj in 1..r+1 {
            b[i][j] += w[r][r+jj]*a[i][j+jj];
          }
          for ii in r..0 {
            b[i][j] += w[r-ii][r]*a[i-ii][j];
          }
          for ii in 1..r+1 {
            b[i][j] += w[r+ii][r]*a[i+ii][j];
          }
        }
      }
    }

    // add constant to solution to force refresh of neighbor data, if any
    for j in 0..n {
      for i in 0..n {
        a[i][j] += 1.0;
      }
    }
  }
  let t1 = timer.elapsed();
  let dt = (t1.checked_sub(t0)).unwrap();
  let dtt : u64 = dt.as_secs() * 1_000_000_000 + dt.subsec_nanos() as u64;
  let stencil_time : f64 = dtt as f64 / 1.0e9_f64 as f64;

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  // error tolerance
  let epsilon : f64 = 1.0e-8;

  // compute L1 norm a parallel
  let mut norm : f64 = 0.0;
  for i in r..n-r+1 {
    for j in r..n-r+1 {
      norm += (b[i][j]).abs();
    }
  }
  norm /= active_points as f64;

  // verify correctness
  let reference_norm : f64 = 2.*(iterations as f64 + 1.);
  if (norm-reference_norm).abs() > epsilon {
    println!("ERROR: L1 norm = {} Reference L1 norm = {}", norm, reference_norm);
    return;
  } else {
    println!("Solution validates");
    if cfg!(VERBOSE) {
      println!("L1 norm = {} Reference L1 norm = {}", norm, reference_norm);
    }
    let flops : usize = (2*stencil_size+1) * active_points;
    let avgtime : f64 = (stencil_time as f64) / (iterations as f64);
    println!("Rate (MFlops/s): {:10.3} Avg time (s): {:10.3}", (1.0e-6_f64) * (flops as f64) / avgtime, avgtime);
  }

}
