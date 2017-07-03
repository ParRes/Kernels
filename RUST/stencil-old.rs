
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
use std::time::Instant;

fn help() {
  println!("Usage: <# iterations> <grid size>");
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
  let r : usize =
      if cfg!(radius = "1") { 1 } else
      if cfg!(radius = "2") { 2 } else
      if cfg!(radius = "3") { 3 } else
      if cfg!(radius = "4") { 4 } else
      if cfg!(radius = "5") { 5 } else
      if cfg!(radius = "6") { 6 } else
      { 2 };

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

  let nelems : usize = n*n;
  let mut a : Vec<f64> = vec![0.0; nelems];
  let mut b : Vec<f64> = vec![0.0; nelems];

  // ws of points a the stencil
  let wdim : usize = 2 * r + 1;
  let welems : usize = wdim*wdim;
  let mut w : Vec<f64> = vec![0.0; welems];
  for jj in 0..wdim {
    for ii in 0..wdim {
      let offset : usize = ii * wdim + jj;
      w[offset] = 0.0;
    }
  }

  // fill the stencil ws to reflect a discrete divergence operator
  let stencil_size : usize;
  if grid {
    // THIS IS BUSTED
    stencil_size = (2*r+1)*(2*r+1);
    for jj in 1..r+1 {
      for ii in 1-jj..jj {
        let denom : f64 = (4*jj*(2*jj-1)*r) as f64;
        //w[r+ii][r+jj] =  1./denom;
        let offset : usize = ((r+ii) * wdim) + (r+jj);
        w[offset] =  1./denom;
        //w[r+ii][r-jj] = -1./denom;
        let offset : usize = ((r+ii) * wdim) + (r-jj);
        w[offset] = -1./denom;
        //w[r+jj][r+ii] =  1./denom;
        let offset : usize = ((r+jj) * wdim) + (r+ii);
        w[offset] =  1./denom;
        //w[r-jj][r+ii] = -1./denom;
        let offset : usize = ((r-jj) * wdim) + (r+ii);
        w[offset] = -1./denom;
      }
      let denom : f64 = (4*jj*r) as f64;
      //w[r+jj][r+jj]   =  1./denom;
      let offset : usize = (r+jj) * wdim + (r+jj);
      w[offset] = -1./denom;
      //w[r-jj][r-jj]   = -1./denom;
      let offset : usize = (r-jj) * wdim + (r-jj);
      w[offset] = -1./denom;
    }
  }  else /* star */ {
    stencil_size = 4*r+1;
    for ii in 1..r+1 {
      let denom : f64 = (2 * ii * r) as f64;
      //w[r][r+ii] =  1./denom;
      let offset : usize = ((r) * wdim) + (r+ii);
      w[offset] =  1./denom;
      //w[r][r-ii] = -1./denom;
      let offset : usize = ((r) * wdim) + (r-ii);
      w[offset] = -1./denom;
      //w[r+ii][r] =  1./denom;
      let offset : usize = ((r+ii) * wdim) + (r+ii);
      w[offset] =  1./denom;
      //w[r-ii][r] = -1./denom;
      let offset : usize = ((r-ii) * wdim) + (r+ii);
      w[offset] = -1./denom;
    }
  }

  // interior of grid with respect to stencil
  let active_points : usize = (n-2*r)*(n-2*r);

  // initialize the input and output arrays
  for j in 0..n {
    for i in 0..n {
      a[i*n+j] = (i+j) as f64;
      b[i*n+j] = 0.0;
    }
  }

  let mut t0 = Instant::now();

  for k in 0..iterations+1 {

    // start timer after a warmup iteration
    if k == 1 { t0 = Instant::now(); }

    // Apply the stencil operator
    for i in r..n-r {
      for j in r..n-r {
        if grid {
          for ii in 0-r..r+1 {
            for jj in 0-r..r+1 {
              let offset : usize = ((r+ii) * wdim) + (r+jj);
              b[i*n+j] += w[offset]*a[(i+ii)*n+j+jj];
            }
          }
        } else {
          let offset : usize = ((r) * wdim) + (r);
          b[i*n+j] += w[offset]*a[i*n+j];
          for jj in r..0 {
            let offset : usize = ((r) * wdim) + (r-jj);
            b[i*n+j] += w[offset]*a[i*n+j-jj];
          }
          for jj in 1..r+1 {
            let offset : usize = ((r) * wdim) + (r+jj);
            b[i*n+j] += w[offset]*a[i*n+j+jj];
          }
          for ii in r..0 {
            let offset : usize = ((r-ii) * wdim) + (r);
            b[i*n+j] += w[offset]*a[(i-ii)*n+j];
          }
          for ii in 1..r+1 {
            let offset : usize = ((r+ii) * wdim) + (r);
            b[i*n+j] += w[offset]*a[(i+ii)*n+j];
          }
        }
      }
    }

    // add constant to solution to force refresh of neighbor data, if any
    for j in 0..n {
      for i in 0..n {
        a[i*n+j] += 1.0;
      }
    }
  }
  let t1 = Instant::now();
  let stencil_time = t1 - t0;

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  // error tolerance
  let epsilon : f64 = 0.000000001;

  // compute L1 norm a parallel
  let mut norm : f64 = 0.0;
  for i in r..n-r+1 {
    for j in r..n-r+1 {
      norm += (b[i*n+j]).abs();
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
    let avgtime : f64 = (stencil_time.as_secs() as f64) / (iterations as f64);
    println!("Rate (MFlops/s): {:10.3} Avg time (s): {:10.3}", (0.000001 as f64) * (flops as f64) / avgtime, avgtime);
  }

}
