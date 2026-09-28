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
/// NAME:    dgemm
///
/// PURPOSE: This program tests the efficiency with which a dense matrix
///          dense multiplication is carried out
///
/// USAGE:   The program takes as input the matrix order,
///          the number of times the matrix-matrix multiplication
///          is carried out.
///
///          <progname> <# iterations> <matrix order>
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// HISTORY: Written by Rob Van der Wijngaart, February 2009.
///          Converted to Swift by Cursor AI, 2025.
///
//////////////////////////////////////////////////////////////////////

import Foundation

func main() {
    print("Parallel Research Kernels")
    print("Swift Dense matrix-matrix multiplication: C += A x B")
    
    //////////////////////////////////////////////////////////////////////
    /// Read and test input parameters
    //////////////////////////////////////////////////////////////////////
    
    let arguments = CommandLine.arguments
    
    guard arguments.count == 3 else {
        print("Usage: swift dgemm.swift <# iterations> <matrix order>")
        exit(1)
    }
    
    guard let iterations = Int(arguments[1]), iterations >= 1 else {
        print("ERROR: iterations must be >= 1")
        exit(1)
    }
    
    guard let order = Int(arguments[2]), order > 0 else {
        print("ERROR: matrix order must be positive")
        exit(1)
    }
    
    print("Number of iterations = \(iterations)")
    print("Matrix order         = \(order)")
    
    //////////////////////////////////////////////////////////////////////
    // Allocate space for matrices
    //////////////////////////////////////////////////////////////////////
    
    var A = Array(repeating: 0.0, count: order * order)
    var B = Array(repeating: 0.0, count: order * order)
    var C = Array(repeating: 0.0, count: order * order)
    
    // Initialize matrices A and B
    for i in 0..<order {
        for j in 0..<order {
            A[i * order + j] = Double(i)
            B[i * order + j] = Double(i)
        }
    }
    
    var startTime = 0.0
    
    for iter in 0...iterations {
        
        // Start timer after warmup iteration
        if iter == 1 {
            startTime = CFAbsoluteTimeGetCurrent()
        }
        
        // Perform matrix multiplication: C += A * B
        for i in 0..<order {
            for k in 0..<order {
                for j in 0..<order {
                    C[i * order + j] += A[i * order + k] * B[k * order + j]
                }
            }
        }
    }
    
    let dgemmTime = CFAbsoluteTimeGetCurrent() - startTime
    
    //////////////////////////////////////////////////////////////////////
    /// Analyze and output results
    //////////////////////////////////////////////////////////////////////
    
    // Calculate average time
    let dgemmAve = dgemmTime / Double(iterations)
    
    // Calculate checksum
    let checksum = C.reduce(0.0, +)
    
    // Calculate reference checksum
    let refChecksum = 0.25 * Double(order * order * order) * Double(order - 1) * Double(order - 1) * Double(iterations + 1)
    
    let epsilon = 1.0e-8
    if abs(checksum - refChecksum) / refChecksum < epsilon {
        print("Solution validates")
        let nflops = 2.0 * Double(order * order * order)
        print("nflops: \(nflops)")
        print(String(format: "Rate: %.6f Avg time (s): %.6f", 1.0e-6 * nflops / dgemmAve, dgemmAve))
    } else {
        print("ERROR: Checksum = \(checksum), Reference checksum = \(refChecksum)")
        print("ERROR: solution did not validate")
        exit(1)
    }
}

main()
