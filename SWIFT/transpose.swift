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
/// NAME:    transpose
///
/// PURPOSE: This program measures the time for the transpose of a
///          column-major stored matrix into a row-major stored matrix.
///
/// USAGE:   Program input is the matrix order and the number of times to
///          repeat the operation:
///
///          <progname> <# iterations> <matrix order>
///
///          The output consists of diagnostics to make sure the
///          transpose worked and timing statistics.
///
/// HISTORY: Written by  Rob Van der Wijngaart, February 2009.
///          Converted to Swift by Cursor AI, 2025.
///
//////////////////////////////////////////////////////////////////////

import Foundation

func main() {
    print("Parallel Research Kernels")
    print("Swift Matrix transpose: B = A^T")
    
    //////////////////////////////////////////////////////////////////////
    /// Read and test input parameters
    //////////////////////////////////////////////////////////////////////
    
    let arguments = CommandLine.arguments
    
    guard arguments.count == 3 else {
        print("Usage: swift transpose.swift <# iterations> <matrix order>")
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
    // Allocate space for the input and transpose matrix
    //////////////////////////////////////////////////////////////////////
    
    // Initialize matrices as 1D arrays for better performance
    var A = Array(repeating: 0.0, count: order * order)
    var B = Array(repeating: 0.0, count: order * order)
    
    // Initialize matrix A with sequence values
    for i in 0..<order {
        for j in 0..<order {
            A[i * order + j] = Double(i * order + j)
        }
    }
    
    var startTime = 0.0
    
    for iter in 0...iterations {
        
        // Start timer after warmup iteration
        if iter == 1 {
            startTime = CFAbsoluteTimeGetCurrent()
        }
        
        // Perform matrix transpose: B[i][j] += A[j][i]; A[j][i] += 1.0
        for i in 0..<order {
            for j in 0..<order {
                B[i * order + j] += A[j * order + i]
                A[j * order + i] += 1.0
            }
        }
    }
    
    let transTime = CFAbsoluteTimeGetCurrent() - startTime
    
    //////////////////////////////////////////////////////////////////////
    /// Analyze and output results
    //////////////////////////////////////////////////////////////////////
    
    // Calculate additive term for validation
    let addit = Double(iterations * (iterations + 1)) / 2.0
    var abserr = 0.0
    
    for i in 0..<order {
        for j in 0..<order {
            let ij = i * order + j
            let ji = j * order + i
            let reference = Double(ij) * Double(iterations + 1) + addit
            abserr += abs(B[ji] - reference)
        }
    }
    
    let epsilon = 1.0e-8
    let nbytes = 2.0 * Double(order * order) * 8.0 // 8 bytes per double, read and write
    
    if abserr < epsilon {
        print("Solution validates")
        let avgtime = transTime / Double(iterations)
        let rate = 1.0e-6 * nbytes / avgtime
        print(String(format: "Rate (MB/s): %.6f Avg time (s): %.6f", rate, avgtime))
    } else {
        print("ERROR: Aggregate error \(abserr) exceeds threshold \(epsilon)")
        print("ERROR: solution did not validate")
        exit(1)
    }
}

main()
