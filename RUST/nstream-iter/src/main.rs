//
// Copyright (c) 2020, Intel Corporation
// Copyright (c) 2020, Thomas Hayward-Schneider
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

///////////////////////////////////////////////
//
// NAME:    nstream
//
// PURPOSE: To compute memory bandwidth when adding a vector of a given
//          number of double precision values to the scalar multiple of
//          another vector of the same length, and storing the result in
//          a third vector.
//
// USAGE:   The program takes as input the number
//          of iterations to loop over the triad vectors, the length of the
//          vectors, and the offset between vectors
//
//          <progname> <# iterations> <vector length> <offset>
//
//          The output consists of diagnostics to make sure the
//          algorithm worked, and of timing statistics.
//
// NOTES:   Bandwidth is determined as the number of words read, plus the
//          number of words written, times the size of the words, divided
//          by the execution time. For a vector length of N, the total
//          number of words read and written is 4*N*sizeof(double).
//
// HISTORY: This code is loosely based on the Stream benchmark by John
//          McCalpin, but does not follow all the Stream rules. Hence,
//          reported results should not be associated with Stream in
//          external publications
//
//          Converted to C++11 by Jeff Hammond, November 2017.
//
///////////////////////////////////////////////

use std::env;
use std::mem;
//use std::num; // abs?
use std::time::{Instant,Duration};

fn help() {
  println!("Usage: <# iterations> <vector length>");
}

fn main()
{
  println!("Parallel Research Kernels");
  println!("Rust STREAM triad: A = B + scalar * C");

  ///////////////////////////////////////////////
  // Read and test input parameters
  ///////////////////////////////////////////////

  let args : Vec<String> = env::args().collect();

  let iterations : u32;
  let length     : usize;

  match args.len() {
    3 => {
      iterations = match args[1].parse() {
        Ok(n) => { n },
        Err(_) => { help(); return; },
      };
      length = match args[2].parse() {
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
  println!("vector length         = {}", length);

  ///////////////////////////////////////////////
  // Allocate space and perform the computation
  ///////////////////////////////////////////////

  let mut a : Vec<f64> = vec![0.0; length];
  let b : Vec<f64> = vec![2.0; length];
  let c : Vec<f64> = vec![2.0; length];

  let timer = Instant::now();
  let mut t0 : Duration = timer.elapsed();

  let scalar : f64 = 3.0;

  for _k in 0..iterations+1 {

    if _k == 1 { t0 = timer.elapsed(); }

    for ((ax, &bx), &cx) in a.iter_mut().zip(b.iter()).zip(c.iter()) {
      *ax += bx + scalar * cx;
    }

  }
  let t1 = timer.elapsed();
  let dt = (t1.checked_sub(t0)).unwrap();
  let dtt : u64 = dt.as_secs() * 1_000_000_000 + dt.subsec_nanos() as u64;
  let nstream_time : f64 = dtt as f64 * 1.0e-9;

  ///////////////////////////////////////////////
  // Analyze and output results
  ///////////////////////////////////////////////

  let mut ar : f64 = 0.0;
  let br : f64 = 2.0;
  let cr : f64 = 2.0;
  for _k in 0..iterations+1 {
      ar += br + scalar * cr;
  }

  ar *= length as f64;

  let mut asum = 0.0;
  for i in 0..length {
      let absa : f64 = a[i].abs();
      asum += absa;
  }

  let err : f64 = (ar-asum)/asum;
  let abserr : f64 = err.abs();
  let epsilon : f64 = 1.0e-8;
  if abserr < epsilon {
    println!("Solution validates");
    let avgtime : f64 = (nstream_time as f64) / (iterations as f64);
    let nbytes : usize = 4 * length * mem::size_of::<f64>();
    println!("Rate (MB/s): {:10.3} Avg time (s): {:10.3}", (1.0e-6_f64) * (nbytes as f64) / avgtime, avgtime);
  } else {
    println!("Failed Validation on output array");
    println!("       Expected checksum: {}", ar);
    println!("       Observed checksum: {}", asum);
    println!("ERROR: solution did not validate");
  }
  return;
}


