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
/// NAME:    xgemm-accelerate
///
/// PURPOSE: This program tests the efficiency with which a dense matrix
///          dense multiplication is carried out using Apple's Accelerate
///          framework with multiple precision types
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
///          Converted to Swift with Accelerate by AI Assistant, December 2024.
///
//////////////////////////////////////////////////////////////////////

import Foundation
import Accelerate

// Matrix multiplication using Accelerate for Float32
func accelerateMatrixMultiplyFloat(
    A: [Float], B: [Float], C: inout [Float], 
    order: Int, alpha: Float, beta: Float
) {
    cblas_sgemm(
        CblasRowMajor,      // Layout
        CblasNoTrans,       // TransA
        CblasNoTrans,       // TransB
        Int32(order),       // M
        Int32(order),       // N
        Int32(order),       // K
        alpha,              // alpha
        A, Int32(order),    // A, lda
        B, Int32(order),    // B, ldb
        beta,               // beta
        &C, Int32(order)    // C, ldc
    )
}

// Matrix multiplication using Accelerate for Float64
func accelerateMatrixMultiplyDouble(
    A: [Double], B: [Double], C: inout [Double], 
    order: Int, alpha: Double, beta: Double
) {
    cblas_dgemm(
        CblasRowMajor,      // Layout
        CblasNoTrans,       // TransA
        CblasNoTrans,       // TransB
        Int32(order),       // M
        Int32(order),       // N
        Int32(order),       // K
        alpha,              // alpha
        A, Int32(order),    // A, lda
        B, Int32(order),    // B, ldb
        beta,               // beta
        &C, Int32(order)    // C, ldc
    )
}

// Fallback naive implementation for Float16 (Accelerate doesn't support it directly)
func naiveMatrixMultiply<T: BinaryFloatingPoint>(
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

// Generic function to run benchmark for Float16
func runBenchmarkFloat16(iterations: Int, order: Int) {
    print("Testing precision: FP16 (naive implementation)")
    
    var dgemmTime: Double = 0
    let nelems = order * order
    
    // Initialize matrices
    var A = Array<Float16>(repeating: Float16(0), count: nelems)
    var B = Array<Float16>(repeating: Float16(0), count: nelems)
    var C = Array<Float16>(repeating: Float16(0), count: nelems)
    
    for i in 0..<order {
        for j in 0..<order {
            A[i * order + j] = Float16(i)
            B[i * order + j] = Float16(i)
            C[i * order + j] = Float16(0)
        }
    }
    
    var startTime: Double = 0
    
    // Benchmark loop
    for k in 0...iterations {
        if k == 1 {
            startTime = CFAbsoluteTimeGetCurrent()
        }
        
        let alpha: Float16 = Float16(1)
        let beta: Float16 = Float16(1)
        
        naiveMatrixMultiply(A: A, B: B, C: &C, order: order, alpha: alpha, beta: beta)
    }
    
    dgemmTime = CFAbsoluteTimeGetCurrent() - startTime
    let dgemmAve = dgemmTime / Double(iterations)
    
    //////////////////////////////////////////////////////////////////////
    // Analyze and output results
    //////////////////////////////////////////////////////////////////////
    
    let checksum = C.reduce(Float16(0), +)
    let forder = Double(order)
    let refChecksum = Float16(0.25 * forder * forder * forder * (forder - 1.0) * (forder - 1.0) * Double(iterations + 1))
    
    let epsilon: Double = 1.0e-3  // Relaxed for 16-bit
    let residuum = abs(Double(checksum) - Double(refChecksum)) / Double(refChecksum)
    
    if residuum < epsilon {
        print("Solution validates")
        let nflops = 2.0 * forder * forder * forder
        let rate = 1.0e-6 * nflops / dgemmAve
        print(String(format: "FP16 Rate (MF/s): %.6f; Avg time (s): %.6f", rate, dgemmAve))
    } else {
        print(String(format: "ERROR: Checksum = %.6f; Reference = %.6f; Residuum = %.6e", 
                     Double(checksum), Double(refChecksum), residuum))
    }
}

// Function to run benchmark for Float32 with Accelerate
func runBenchmarkFloat32(iterations: Int, order: Int) {
    print("Testing precision: FP32 (Accelerate cblas_sgemm)")
    
    var dgemmTime: Double = 0
    let nelems = order * order
    
    // Initialize matrices
    var A = Array<Float>(repeating: Float(0), count: nelems)
    var B = Array<Float>(repeating: Float(0), count: nelems)
    var C = Array<Float>(repeating: Float(0), count: nelems)
    
    for i in 0..<order {
        for j in 0..<order {
            A[i * order + j] = Float(i)
            B[i * order + j] = Float(i)
            C[i * order + j] = Float(0)
        }
    }
    
    var startTime: Double = 0
    
    // Benchmark loop
    for k in 0...iterations {
        if k == 1 {
            startTime = CFAbsoluteTimeGetCurrent()
        }
        
        let alpha: Float = 1.0
        let beta: Float = 1.0
        
        accelerateMatrixMultiplyFloat(A: A, B: B, C: &C, order: order, alpha: alpha, beta: beta)
    }
    
    dgemmTime = CFAbsoluteTimeGetCurrent() - startTime
    let dgemmAve = dgemmTime / Double(iterations)
    
    //////////////////////////////////////////////////////////////////////
    // Analyze and output results
    //////////////////////////////////////////////////////////////////////
    
    let checksum = C.reduce(Float(0), +)
    let forder = Double(order)
    let refChecksum = Float(0.25 * forder * forder * forder * (forder - 1.0) * (forder - 1.0) * Double(iterations + 1))
    
    let epsilon: Double = 1.0e-6  // Standard for 32-bit
    let residuum = abs(Double(checksum) - Double(refChecksum)) / Double(refChecksum)
    
    if residuum < epsilon {
        print("Solution validates")
        let nflops = 2.0 * forder * forder * forder
        let rate = 1.0e-6 * nflops / dgemmAve
        print(String(format: "FP32 Rate (MF/s): %.6f; Avg time (s): %.6f", rate, dgemmAve))
    } else {
        print(String(format: "ERROR: Checksum = %.6f; Reference = %.6f; Residuum = %.6e", 
                     Double(checksum), Double(refChecksum), residuum))
    }
}

// Function to run benchmark for Float64 with Accelerate
func runBenchmarkFloat64(iterations: Int, order: Int) {
    print("Testing precision: FP64 (Accelerate cblas_dgemm)")
    
    var dgemmTime: Double = 0
    let nelems = order * order
    
    // Initialize matrices
    var A = Array<Double>(repeating: Double(0), count: nelems)
    var B = Array<Double>(repeating: Double(0), count: nelems)
    var C = Array<Double>(repeating: Double(0), count: nelems)
    
    for i in 0..<order {
        for j in 0..<order {
            A[i * order + j] = Double(i)
            B[i * order + j] = Double(i)
            C[i * order + j] = Double(0)
        }
    }
    
    var startTime: Double = 0
    
    // Benchmark loop
    for k in 0...iterations {
        if k == 1 {
            startTime = CFAbsoluteTimeGetCurrent()
        }
        
        let alpha: Double = 1.0
        let beta: Double = 1.0
        
        accelerateMatrixMultiplyDouble(A: A, B: B, C: &C, order: order, alpha: alpha, beta: beta)
    }
    
    dgemmTime = CFAbsoluteTimeGetCurrent() - startTime
    let dgemmAve = dgemmTime / Double(iterations)
    
    //////////////////////////////////////////////////////////////////////
    // Analyze and output results
    //////////////////////////////////////////////////////////////////////
    
    let checksum = C.reduce(Double(0), +)
    let forder = Double(order)
    let refChecksum = Double(0.25 * forder * forder * forder * (forder - 1.0) * (forder - 1.0) * Double(iterations + 1))
    
    let epsilon: Double = 1.0e-8  // Strict for 64-bit
    let residuum = abs(checksum - refChecksum) / refChecksum
    
    if residuum < epsilon {
        print("Solution validates")
        let nflops = 2.0 * forder * forder * forder
        let rate = 1.0e-6 * nflops / dgemmAve
        print(String(format: "FP64 Rate (MF/s): %.6f; Avg time (s): %.6f", rate, dgemmAve))
    } else {
        print(String(format: "ERROR: Checksum = %.6f; Reference = %.6f; Residuum = %.6e", 
                     checksum, refChecksum, residuum))
    }
}

func main() {
    print("Parallel Research Kernels")
    print("Swift Dense matrix-matrix multiplication: C += A x B (Multi-precision Accelerate)")
    
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
    
    guard order <= 2000 else {
        print("ERROR: matrix dimension too large - overflow risk")
        exit(1)
    }
    
    print("Number of iterations = \(iterations)")
    print("Matrix order         = \(order)")
    
    // Test all supported precision types
    if #available(macOS 11.0, *) {
        runBenchmarkFloat16(iterations: iterations, order: order)
    } else {
        print("Float16 not available on this macOS version")
    }
    
    runBenchmarkFloat32(iterations: iterations, order: order)
    runBenchmarkFloat64(iterations: iterations, order: order)
}

main()
