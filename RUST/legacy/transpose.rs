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
/// NAME:    transpose
///
/// PURPOSE: This program measures the time for the transpose of a
///          column-major stored matrix into a row-major stored matrix.
///
/// USAGE:   Program input is the matrix order and the number of times to
///          repeat the operation:
///
///          transpose <matrix_size> <# iterations> [tile size]
///
///          An optional parameter specifies the tile size used to divide the
///          individual matrix blocks for improved cache and TLB performance.
///
///          The output consists of diagnostics to make sure the
///          transpose worked and timing statistics.
///
/// HISTORY: Written by  Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, February 2016 and May 2017.
///
//////////////////////////////////////////////////////////////////////

use std::env;
use std::mem;
use std::time::{Instant,Duration};

fn help() {
  println!("Usage: <# iterations> <matrix order> [tile size]");
}

fn main()
{
  println!("Parallel Research Kernels");
  println!("Rust Matrix transpose: B = A^T");

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  let args : Vec<String> = env::args().collect();

  let iterations : u32;
  let order      : usize;
  let tilesize   : usize;

  match args.len() {
    3 => {
      iterations = match args[1].parse() {
        Ok(n) => { n },
        Err(_) => { help(); return; },
      };
      order = match args[2].parse() {
        Ok(n) => { n },
        Err(_) => { help(); return; },
      };
      tilesize = 32;
    },
    4 => {
      iterations = match args[1].parse() {
        Ok(n) => { n },
        Err(_) => { help(); return; },
      };
      order = match args[2].parse() {
        Ok(n) => { n },
        Err(_) => { help(); return; },
      };
      tilesize = match args[3].parse() {
        Ok(n) => { n },
        Err(_) => { help(); return; },
      };
    },
    _ => {
      help();
      return;
    }
  }

  if iterations < 1 {
    println!("ERROR: iterations must be >= 1");
  }
  if tilesize > order {
    println!("ERROR: tilesize cannot be > order");
  }

  println!("Matrix order          = {}", order);
  if tilesize < order {
      println!("Tile size             = {}", tilesize);
  } else {
      println!("Untiled");
  }
  println!("Number of iterations  = {}", iterations);

  //////////////////////////////////////////////////////////////////////
  // Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

  let nelems : usize = order*order;
  let mut a : Vec<f64> = vec![0.0; nelems];
  let mut b : Vec<f64> = vec![0.0; nelems];

  for i in 0..order {
    for j in 0..order {
      a[i*order+j] = (i*order+j) as f64;
    }
  }

  let timer = Instant::now();
  let mut t0 : Duration = timer.elapsed();

  for k in 0..iterations+1 {

    if k == 1 { t0 = timer.elapsed(); }

    for i in 0..order {
      for j in 0..order {
        b[j*order+i] += a[i*order+j];
        a[i*order+j] += 1.0;
      }
    }

  }
  let t1 = timer.elapsed();
  let dt = (t1.checked_sub(t0)).unwrap();
  let dtt : u64 = dt.as_secs() * 1_000_000_000 + dt.subsec_nanos() as u64;
  let transpose_time : f64 = dtt as f64 * 1.0e-9;

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  let addit : usize = ((iterations as usize + 1) * (iterations as usize)) / 2;
  let mut abserr : f64 = 0.0;
  for i in 0..order {
    for j in 0..order {
      let ij = i*order+j;
      let ji = j*order+i;
      let reference : f64 = (ij*(iterations as usize + 1)+addit) as f64;
      abserr += (b[ji] - reference).abs();
    }
  }

  if cfg!(VERBOSE) {
    println!("Sum of absolute differences: {:30.15}", abserr);
  }

  let epsilon : f64 = 1.0e-8;
  if abserr < epsilon {
    println!("Solution validates");
    let avgtime : f64 = (transpose_time as f64) / (iterations as f64);
    let bytes : usize = 2 * nelems * mem::size_of::<f64>();
    println!("Rate (MB/s): {:10.3} Avg time (s): {:10.3}", (1.0e0-6_f64) * (bytes as f64) / avgtime, avgtime);
  } else {
    println!("ERROR: Aggregate squared error {:30.15} exceeds threshold {:30.15}", abserr, epsilon);
    return;
  }
}


