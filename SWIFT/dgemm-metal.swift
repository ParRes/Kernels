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
/// NAME:    dgemm-metal
///
/// PURPOSE: This program tests the efficiency with which a dense matrix
///          dense multiplication is carried out using Metal GPU compute
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
///          Converted to Swift with Metal by Cursor AI, 2025.
///
//////////////////////////////////////////////////////////////////////

import Foundation
import Metal

let metalSource = """
#include <metal_stdlib>
using namespace metal;

kernel void dgemm_kernel(device float* A [[buffer(0)]],
                        device float* B [[buffer(1)]],
                        device float* C [[buffer(2)]],
                        constant uint& order [[buffer(3)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= order || gid.y >= order) return;
    
    uint i = gid.y;
    uint j = gid.x;
    
    float sum = 0.0;
    for (uint k = 0; k < order; k++) {
        sum += A[i * order + k] * B[k * order + j];
    }
    C[i * order + j] += sum;
}
"""

func main() {
    print("Parallel Research Kernels")
    print("Swift Dense matrix-matrix multiplication: C += A x B (Metal GPU)")
    
    //////////////////////////////////////////////////////////////////////
    /// Read and test input parameters
    //////////////////////////////////////////////////////////////////////
    
    let arguments = CommandLine.arguments
    
    guard arguments.count == 3 else {
        print("Usage: swift dgemm-metal.swift <# iterations> <matrix order>")
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
    
    guard let function = library.makeFunction(name: "dgemm_kernel") else {
        print("ERROR: Failed to find kernel function")
        exit(1)
    }
    
    guard let computePipelineState = try? device.makeComputePipelineState(function: function) else {
        print("ERROR: Failed to create compute pipeline state")
        exit(1)
    }
    
    //////////////////////////////////////////////////////////////////////
    // Allocate space for matrices
    //////////////////////////////////////////////////////////////////////
    
    var A = Array(repeating: Float(0.0), count: order * order)
    var B = Array(repeating: Float(0.0), count: order * order)
    let C = Array(repeating: Float(0.0), count: order * order)
    
    // Initialize matrices A and B
    for i in 0..<order {
        for j in 0..<order {
            A[i * order + j] = Float(i)
            B[i * order + j] = Float(i)
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
    
    guard let bufferC = device.makeBuffer(bytes: C, length: order * order * MemoryLayout<Float>.size, options: [.storageModeShared]) else {
        print("ERROR: Failed to create buffer C")
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
        computeEncoder.setBuffer(bufferC, offset: 0, index: 2)
        computeEncoder.setBuffer(bufferOrder, offset: 0, index: 3)
        
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
    
    let dgemmTime = CFAbsoluteTimeGetCurrent() - startTime
    
    //////////////////////////////////////////////////////////////////////
    /// Analyze and output results
    //////////////////////////////////////////////////////////////////////
    
    // Calculate average time
    let dgemmAve = dgemmTime / Double(iterations)
    
    // Copy results back from GPU
    let resultPointer = bufferC.contents().bindMemory(to: Float.self, capacity: order * order)
    let results = Array(UnsafeBufferPointer(start: resultPointer, count: order * order))
    
    // Calculate checksum
    let checksum = results.reduce(0.0) { Double($0) + Double($1) }
    
    // Calculate reference checksum
    let refChecksum = 0.25 * Double(order * order * order) * Double(order - 1) * Double(order - 1) * Double(iterations + 1)
    
    let epsilon = 1.0e-6  // Relaxed for Float32
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
