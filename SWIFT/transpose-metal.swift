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
/// NAME:    transpose-metal
///
/// PURPOSE: This program measures the time for the transpose of a
///          column-major stored matrix into a row-major stored matrix
///          using Metal GPU compute.
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
///          Converted to Swift with Metal by Cursor AI, 2025.
///
//////////////////////////////////////////////////////////////////////

import Foundation
import Metal

let metalSource = """
#include <metal_stdlib>
using namespace metal;

kernel void transpose_kernel(device float* A [[buffer(0)]],
                           device float* B [[buffer(1)]],
                           constant uint& order [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= order || gid.y >= order) return;
    
    uint i = gid.y;
    uint j = gid.x;
    
    B[i * order + j] += A[j * order + i];
    A[j * order + i] += 1.0;
}
"""

func main() {
    print("Parallel Research Kernels")
    print("Swift Matrix transpose: B = A^T (Metal GPU)")
    
    //////////////////////////////////////////////////////////////////////
    /// Read and test input parameters
    //////////////////////////////////////////////////////////////////////
    
    let arguments = CommandLine.arguments
    
    guard arguments.count == 3 else {
        print("Usage: swift transpose-metal.swift <# iterations> <matrix order>")
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
    // Setup Metal
    //////////////////////////////////////////////////////////////////////
    
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("ERROR: Metal is not supported on this device")
        exit(1)
    }
    
    guard let commandQueue = device.makeCommandQueue() else {
        print("ERROR: Failed to create command queue")
        exit(1)
    }
    
    guard let library = try? device.makeLibrary(source: metalSource, options: nil) else {
        print("ERROR: Failed to create Metal library")
        exit(1)
    }
    
    guard let function = library.makeFunction(name: "transpose_kernel") else {
        print("ERROR: Failed to find kernel function")
        exit(1)
    }
    
    guard let computePipelineState = try? device.makeComputePipelineState(function: function) else {
        print("ERROR: Failed to create compute pipeline state")
        exit(1)
    }
    
    //////////////////////////////////////////////////////////////////////
    // Allocate space for the input and transpose matrix
    //////////////////////////////////////////////////////////////////////
    
    // Initialize matrices as 1D arrays using Float32
    var A = Array(repeating: Float(0.0), count: order * order)
    let B = Array(repeating: Float(0.0), count: order * order)
    
    // Initialize matrix A with sequence values
    for i in 0..<order {
        for j in 0..<order {
            A[i * order + j] = Float(i * order + j)
        }
    }
    
    var orderConstant = UInt32(order)
    
    // Create Metal buffers
    guard let bufferA = device.makeBuffer(bytes: A, length: order * order * MemoryLayout<Float>.size, options: [.storageModeShared]) else {
        print("ERROR: Failed to create buffer A")
        exit(1)
    }
    
    guard let bufferB = device.makeBuffer(bytes: B, length: order * order * MemoryLayout<Float>.size, options: [.storageModeShared]) else {
        print("ERROR: Failed to create buffer B")
        exit(1)
    }
    
    guard let bufferOrder = device.makeBuffer(bytes: &orderConstant, length: MemoryLayout<UInt32>.size, options: [.storageModeShared]) else {
        print("ERROR: Failed to create order buffer")
        exit(1)
    }
    
    //////////////////////////////////////////////////////////////////////
    // Execute computation
    //////////////////////////////////////////////////////////////////////
    
    var startTime = 0.0
    
    for iter in 0...iterations {
        
        // Start timer after warmup iteration
        if iter == 1 {
            startTime = CFAbsoluteTimeGetCurrent()
        }
        
        // Create command buffer
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            print("ERROR: Failed to create command buffer")
            exit(1)
        }
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            print("ERROR: Failed to create compute encoder")
            exit(1)
        }
        
        // Set compute pipeline and buffers
        computeEncoder.setComputePipelineState(computePipelineState)
        computeEncoder.setBuffer(bufferA, offset: 0, index: 0)
        computeEncoder.setBuffer(bufferB, offset: 0, index: 1)
        computeEncoder.setBuffer(bufferOrder, offset: 0, index: 2)
        
        // Calculate thread group sizes for 2D dispatch
        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (order + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: (order + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
            depth: 1
        )
        
        // Dispatch threads
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        // Execute
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    let transTime = CFAbsoluteTimeGetCurrent() - startTime
    
    //////////////////////////////////////////////////////////////////////
    /// Analyze and output results
    //////////////////////////////////////////////////////////////////////
    
    // Copy results back from GPU
    let resultPointerB = bufferB.contents().bindMemory(to: Float.self, capacity: order * order)
    let resultsB = Array(UnsafeBufferPointer(start: resultPointerB, count: order * order))
    
    // Calculate additive term for validation
    let addit = Double(iterations * (iterations + 1)) / 2.0
    var abserr = 0.0
    
    for i in 0..<order {
        for j in 0..<order {
            let ij = i * order + j
            let ji = j * order + i
            let reference = Double(ij) * Double(iterations + 1) + addit
            abserr += abs(Double(resultsB[ji]) - reference)
        }
    }
    
    let epsilon = 1.0e-6  // Relaxed for Float32
    let nbytes = 2.0 * Double(order * order) * 4.0 // 4 bytes per float, read and write
    
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
