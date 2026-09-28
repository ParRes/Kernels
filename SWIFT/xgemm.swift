///
/// Copyright (c) 2020, Intel Corporation
/// Copyright (c) 2023, NVIDIA
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
/// NAME:    xgemm
///
/// PURPOSE: This program tests the efficiency with which a dense matrix
///          dense multiplication is carried out using multiple precision types
///
/// USAGE:   The program takes as input the matrix order,
///          the number of times the matrix-matrix multiplication
///          is carried out
///
///          <progname> <# iterations> <matrix order>
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// HISTORY: Written by Rob Van der Wijngaart, February 2009.
///          Converted to Swift by AI Assistant, December 2024.
///
//////////////////////////////////////////////////////////////////////

import Foundation

// Generic matrix multiplication function
func matrixMultiply<T: BinaryFloatingPoint>(
    A: [T], B: [T], C: inout [T], 
    order: Int, alpha: T, beta: T
) {
    for i in 0..<order {
        for j in 0..<order {
            var temp: T = 0
            for k in 0..<order {
                temp += A[i * order + k] * B[k * order + j]
            }
            C[i * order + j] = alpha * temp + beta * C[i * order + j]
        }
    }
}

// Generic function to run benchmark for a specific type
func runBenchmark<T: BinaryFloatingPoint>(
    type: T.Type, iterations: Int, order: Int
) {
    print("Testing precision: \(getPrecisionName(type))")
    
    var dgemmTime: Double = 0
    let nelems = order * order
    
    // Initialize matrices
    var A = Array<T>(repeating: T(0), count: nelems)
    var B = Array<T>(repeating: T(0), count: nelems)
    var C = Array<T>(repeating: T(0), count: nelems)
    
    for i in 0..<order {
        for j in 0..<order {
            A[i * order + j] = T(i)
            B[i * order + j] = T(i)
            C[i * order + j] = T(0)
        }
    }
    
    var startTime: Double = 0
    
    // Benchmark loop
    for k in 0...iterations {
        if k == 1 {
            startTime = CFAbsoluteTimeGetCurrent()
        }
        
        let alpha: T = T(1)
        let beta: T = T(1)
        
        matrixMultiply(A: A, B: B, C: &C, order: order, alpha: alpha, beta: beta)
    }
    
    dgemmTime = CFAbsoluteTimeGetCurrent() - startTime
    let dgemmAve = dgemmTime / Double(iterations)
    
    //////////////////////////////////////////////////////////////////////
    // Analyze and output results
    //////////////////////////////////////////////////////////////////////
    
    let checksum = C.reduce(T(0), +)
    let forder = Double(order)
    let refChecksumDouble = 0.25 * forder * forder * forder * (forder - 1.0) * (forder - 1.0) * Double(iterations + 1)
    // Convert via the closest representable value rather than requiring an exact
    // (lossless) conversion: T(exactly:) fails for almost every non-trivial value
    // in lower-precision types like Float16, which made validation spuriously
    // report "overflow" even when the reference value was well within range.
    let refChecksumFinite = refChecksumDouble <= Double(T.greatestFiniteMagnitude)
    let refChecksum: T = refChecksumFinite ? T(refChecksumDouble) : T.infinity

    let epsilon: Double = getSizeBasedEpsilon(type)
    let residuum = refChecksumFinite ? abs(Double(checksum) - Double(refChecksum)) / Double(refChecksum) : Double.infinity

    if residuum < epsilon {
        print("Solution validates")
        let nflops = 2.0 * forder * forder * forder
        let rate = 1.0e-6 * nflops / dgemmAve
        print("\(getPrecisionName(type)) Rate (MF/s): \(String(format: "%.6f", rate)); Avg time (s): \(String(format: "%.6f", dgemmAve))")
    } else {
        if !refChecksumFinite {
            print("ERROR: Reference checksum overflow for \(getPrecisionName(type)) precision with order \(order)")
        } else {
            print(String(format: "ERROR: Checksum = %.6f; Reference = %.6f; Residuum = %.6e",
                         Double(checksum), Double(refChecksum), residuum))
        }
    }
}

// Helper function to get precision name
func getPrecisionName<T>(_ type: T.Type) -> String {
    switch type {
    case is Float16.Type:
        return "FP16"
    case is Float.Type:
        return "FP32"
    case is Double.Type:
        return "FP64"
    default:
        return "Unknown"
    }
}

// Helper function to get appropriate epsilon based on precision
func getSizeBasedEpsilon<T>(_ type: T.Type) -> Double {
    switch type {
    case is Float16.Type:
        return 1.0e-3  // Relaxed for 16-bit
    case is Float.Type:
        return 1.0e-4  // Relaxed for 32-bit: naive O(n^3) accumulation in Float32
                       // accumulates enough rounding error at larger orders that
                       // 1e-6 spuriously fails (e.g. residuum ~5e-5 at order 500)
    case is Double.Type:
        return 1.0e-8  // Strict for 64-bit
    default:
        return 1.0e-6
    }
}

func main() {
    print("Parallel Research Kernels")
    print("Swift Dense matrix-matrix multiplication: C += A x B (Multi-precision)")
    
    let arguments = CommandLine.arguments
    guard arguments.count >= 3 else {
        print("Usage: \(arguments[0]) <# iterations> <matrix order>")
        exit(1)
    }
    
    guard let iterations = Int(arguments[1]), iterations >= 1 else {
        print("ERROR: iterations must be >= 1")
        exit(1)
    }
    
    guard let order = Int(arguments[2]), order > 0 else {
        print("ERROR: Matrix Order must be greater than 0")
        exit(1)
    }
    
    if order > 2000 {
        print("WARNING: matrix order > 2000 - Float precision reference checksum may overflow")
    }

    print("Number of iterations = \(iterations)")
    print("Matrix order         = \(order)")
    
    // Test all supported precision types
    runBenchmark(type: Float16.self, iterations: iterations, order: order)
    runBenchmark(type: Float.self, iterations: iterations, order: order)
    runBenchmark(type: Double.self, iterations: iterations, order: order)
}

main()
