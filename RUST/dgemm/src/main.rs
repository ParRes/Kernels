//
// Copyright (c) 2013, Intel Corporation
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above
//       copyright notice, this list of conditions and the following
//       disclaimer in the documentation and/or other materials provided
//       with the distribution.
// * Neither the name of Intel Corporation nor the names of its
//       contributors may be used to endorse or promote products
//       derived from this software without specific prior written
//       permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

//////////////////////////////////////////////
//
// NAME:    transpose
//
// PURPOSE: This program measures the time for the transpose of a
//          column-major stored matrix into a row-major stored matrix.
//
// USAGE:   Program input is the matrix order and the number of times to
//          repeat the operation:
//
//          transpose <matrix_size> <# iterations> [tile size]
//
//          An optional parameter specifies the tile size used to divide the
//          individual matrix blocks for improved cache and TLB performance.
//
//          The output consists of diagnostics to make sure the
//          transpose worked and timing statistics.
//
// HISTORY: Written by  Rob Van der Wijngaart, February 2009.
//          Converted to C++11 by Jeff Hammond, February 2016 and May 2017.
//
///////////////////////////////////////////////

extern crate blas;
extern crate cblas;
extern crate blas_src;

use std::env;
use std::time::{Instant,Duration};

//use blas::*;
use cblas::*;

fn prk_dgemm(order : usize, a : &mut Vec<f64>, b : &mut Vec<f64>, c : &mut Vec<f64>)
{
  for i in 0..order {
    for k in 0..order {
      for j in 0..order {
        c[i*order+j] += a[i*order+k] * b[k*order+j];
      }
    }
  }
}

fn help() {
  println!("Usage: <# iterations> <matrix order>");
}

fn main()
{
  println!("Parallel Research Kernels");
  println!("Rust Dense matrix-matrix multiplication: C += A x B");

  ///////////////////////////////////////////////
  // Read and test input parameters
  ///////////////////////////////////////////////

  let args : Vec<String> = env::args().collect();

  let iterations : u32;
  let order      : usize;

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
    },
    _ => {
      help();
      return;
    }
  }

  if iterations < 1 {
    println!("ERROR: iterations must be >= 1");
  }

  println!("Number of iterations  = {}", iterations);
  println!("Matrix order          = {}", order);

  ///////////////////////////////////////////////
  // Allocate space for the input and transpose matrix
  ///////////////////////////////////////////////

  let nelems : usize = order*order;
  let mut a : Vec<f64> = vec![0.0; nelems];
  let mut b : Vec<f64> = vec![0.0; nelems];
  let mut c : Vec<f64> = vec![0.0; nelems];

  for i in 0..order {
    for j in 0..order {
      a[i*order+j] = i as f64;
      b[i*order+j] = i as f64;
    }
  }

  let timer = Instant::now();
  let mut t0 : Duration = timer.elapsed();

  for k in 0..iterations+1 {

    if k == 1 { t0 = timer.elapsed(); }

    //prk_dgemm(order, &mut a, &mut b, &mut c);
    let m : i32 = order as i32;
    let n : i32 = order as i32;
    let k : i32 = order as i32;
    unsafe {
        dgemm(Layout::RowMajor, Transpose::None, Transpose::None,
              m, n, k, 1.0, &a, m, &b, k, 1.0, &mut c, m);
    }

  }
  let t1 = timer.elapsed();
  let dt = (t1.checked_sub(t0)).unwrap();
  let dtt : u64 = dt.as_secs() * 1_000_000_000 + dt.subsec_nanos() as u64;
  let dgemm_time : f64 = dtt as f64 * 1.0e-9;

  ///////////////////////////////////////////////
  // Analyze and output results
  ///////////////////////////////////////////////

  let forder : f64 = order as f64;
  let reference : f64 = 0.25 * (forder*forder*forder) * (forder-1.0)*(forder-1.0) * (iterations as f64 + 1.0);
  let mut checksum : f64 = 0.0;
  for i in 0..order {
    for j in 0..order {
      checksum += c[i*order+j];
    }
  }

  if cfg!(VERBOSE) {
    println!("Sum of absolute differences: {:30.15}", checksum);
  }

  let epsilon : f64 = 1.0e-8;
  let residuum : f64 = (checksum - reference)/reference;
  if residuum < epsilon {
    println!("Solution validates");
    let avgtime : f64 = (dgemm_time as f64) / (iterations as f64);
    let uorder : usize = order as usize;
    let nflops : usize = 2_usize * uorder * uorder * uorder;
    println!("Rate (MB/s): {:10.3} Avg time (s): {:10.3}", (1.0e-6_f64) * (nflops as f64) / avgtime, avgtime);
  } else {
    println!("ERROR: Aggregate squared error {:30.15} exceeds threshold {:30.15}", residuum, epsilon);
    return;
  }
}


