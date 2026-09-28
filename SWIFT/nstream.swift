///
/// Copyright (c) 2025, NVIDIA
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
/// NAME:    nstream
///
/// PURPOSE: To compute memory bandwidth when adding a vector of a given
///          number of double precision values to the scalar multiple of
///          another vector of the same length, and storing the result in
///          a third vector.
///
/// USAGE:   The program takes as input the number
///          of iterations to loop over the triad vectors and
///          the length of the vectors.
///
///          <progname> <# iterations> <vector length>
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// NOTES:   Bandwidth is determined as the number of words read, plus the
///          number of words written, times the size of the words, divided
///          by the execution time. For a vector length of N, the total
///          number of words read and written is 4*N*sizeof(double).
///
/// HISTORY: This code is loosely based on the Stream benchmark by John
///          McCalpin, but does not follow all the Stream rules. Hence,
///          reported results should not be associated with Stream in
///          external publications
///
///          Converted to Swift by Cursor AI, 2025.
///
//////////////////////////////////////////////////////////////////////

import Foundation

func main() {
    print("Parallel Research Kernels")
    print("Swift STREAM triad: A = B + scalar * C")
    
    //////////////////////////////////////////////////////////////////////
    /// Read and test input parameters
    //////////////////////////////////////////////////////////////////////
    
    let arguments = CommandLine.arguments
    
    guard arguments.count == 3 else {
        print("Usage: swift nstream.swift <# iterations> <vector length>")
        exit(1)
    }
    
    guard let iterations = Int(arguments[1]), iterations >= 1 else {
        print("ERROR: iterations must be >= 1")
        exit(1)
    }
    
    guard let length = Int(arguments[2]), length > 0 else {
        print("ERROR: vector length must be positive")
        exit(1)
    }
    
    print("Number of iterations = \(iterations)")
    print("Vector length        = \(length)")
    
    //////////////////////////////////////////////////////////////////////
    // Allocate space and perform the computation
    //////////////////////////////////////////////////////////////////////
    
    var A = Array(repeating: 0.0, count: length)
    let B = Array(repeating: 2.0, count: length)
    let C = Array(repeating: 2.0, count: length)
    
    let scalar = 3.0
    var startTime = 0.0
    
    for iter in 0...iterations {
        
        // Start timer after warmup iteration
        if iter == 1 {
            startTime = CFAbsoluteTimeGetCurrent()
        }
        
        // Perform STREAM triad: A = B + scalar * C
        for i in 0..<length {
            A[i] += B[i] + scalar * C[i]
        }
    }
    
    let nstreamTime = CFAbsoluteTimeGetCurrent() - startTime
    
    //////////////////////////////////////////////////////////////////////
    /// Analyze and output results
    //////////////////////////////////////////////////////////////////////
    
    // Calculate reference result
    var ar = 0.0
    let br = 2.0
    let cr = 2.0
    
    for _ in 0...iterations {
        ar += br + scalar * cr
    }
    
    ar *= Double(length)
    
    // Calculate checksum
    let asum = A.reduce(0.0) { $0 + abs($1) }
    
    let epsilon = 1.0e-8
    if abs(ar - asum) / asum > epsilon {
        print("Failed Validation on output array")
        print("        Expected checksum: \(ar)")
        print("        Observed checksum: \(asum)")
        print("ERROR: solution did not validate")
        exit(1)
    } else {
        print("Solution validates")
        let avgtime = nstreamTime / Double(iterations)
        let nbytes = 4.0 * Double(length) * 8.0 // 8 bytes per double
        let rate = 1.0e-6 * nbytes / avgtime
        print(String(format: "Rate (MB/s): %.6f Avg time (s): %.6f", rate, avgtime))
    }
}

main()
