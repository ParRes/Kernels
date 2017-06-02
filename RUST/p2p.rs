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
/// NAME:    Pipeline
///
/// PURPOSE: This program tests the efficiency with which point-to-point
///          synchronization can be carried out. It does so by executing
///          a pipelined algorithm on an m*n grid. The first array dimension
///          is distributed among the threads (stripwise decomposition).
///
/// USAGE:   The program takes as input the
///          dimensions of the grid, and the number of iterations on the grid
///
///                <progname> <iterations> <m> <n>
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// FUNCTIONS CALLED:
///
///          Other than standard C functions, the following
///          functions are used in this program:
///
///          wtime()
///
/// HISTORY: - Written by Rob Van der Wijngaart, February 2009.
///          - C99-ification by Jeff Hammond, February 2016.
///          - C++11-ification by Jeff Hammond, May 2017.
///          - Rust port by Jeff Hammond, May 2017.
///
//////////////////////////////////////////////////////////////////////

use std::env;
use std::time::{Instant,Duration};

fn help() {
  println!("Usage: <# iterations> <matrix order> [tile size]");
}

fn main()
{
  println!("Parallel Research Kernels version");
  println!("Rust pipeline execution on 2D grid");

  //////////////////////////////////////////////////////////////////////
  // Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  let args : Vec<String> = env::args().collect();

  let iterations : u32;
  let m : usize;
  let n : usize;

  if args.len() == 4 {
    iterations = match args[1].parse() {
      Ok(n) => { n },
      Err(_) => { help(); return; },
    };
    m = match args[2].parse() {
      Ok(n) => { n },
      Err(_) => { help(); return; },
    };
    n = match args[3].parse() {
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
  if m < 1 || n < 1 {
    println!("ERROR: grid dimensions must be positive: {}, {}", m, n);
  }

  println!("Grid sizes                = {}, {}", m, n);
  println!("Number of iterations      = {}", iterations);

  //////////////////////////////////////////////////////////////////////
  // Allocate space for the input and do the work
  //////////////////////////////////////////////////////////////////////

  let nelems : usize = m*n;
  let mut vector : Vec<f64> = vec![0.0; nelems];

  // set boundary values (bottom and left side of grid)
  for j in 0..n {
    vector[0*n+j] = j as f64;
  }
  for i in 0..m {
    vector[i*n+0] = i as f64;
  }

  let timer = Instant::now();
  let mut t0 : Duration = timer.elapsed();

  for k in 0..iterations+1 {

    if k == 1 { t0 = timer.elapsed(); }

    for i in 1..m {
      for j in 1..n {
        vector[i*n+j] = vector[(i-1)*n+j] + vector[i*n+(j-1)] - vector[(i-1)*n+(j-1)];
      }
    }

    // copy top right corner value to bottom left corner to create dependency; we
    // need a barrier to make sure the latest value is used. This also guarantees
    // that the flags for the next iteration (if any) are not getting clobbered
    vector[0*n+0] = -vector[(m-1)*n+(n-1)];

  }
  let t1 = timer.elapsed();
  let dt = (t1.checked_sub(t0)).unwrap();
  let dtt : u64 = dt.as_secs() * 1_000_000_000 + dt.subsec_nanos() as u64;
  let pipeline_time : f64 = dtt as f64 / 1.0e9_f64 as f64;

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results.
  //////////////////////////////////////////////////////////////////////

  // error tolerance
  let epsilon : f64 = 1.0e-8;

  // verify correctness, using top right value
  let corner_val : f64 = (((iterations+1) as usize)*(n + m as usize - 2 as usize)) as f64;
  if ( (vector[(m-1)*n+(n-1)] - corner_val).abs() / corner_val) > epsilon {
    println!("ERROR: checksum {} does not match verification value {} ", vector[(m-1)*n+(n-1)], corner_val);
    return;
  }

  if cfg!(VERBOSE) {
    println!("Solution validates; verification value = {}", corner_val);
  } else {
    println!("Solution validates");
  }

  let avgtime : f64 = (pipeline_time as f64) / (iterations as f64);
  let bytes : usize = 2 * (m-1) * (n-1);
  println!("Rate (MB/s): {:10.3} Avg time (s): {:10.3}", (1.0e-6_f64) * (bytes as f64) / avgtime, avgtime);
}
