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
/// NAME:    xgemm-metal
///
/// PURPOSE: This program tests the efficiency with which a dense matrix
///          dense multiplication is carried out using Apple's Metal
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
///          Converted to Swift with Metal by AI Assistant, December 2024.
///
//////////////////////////////////////////////////////////////////////

import Foundation
import Metal

let metalSource16 = """
#include <metal_stdlib>
using namespace metal;

kernel void gemm16_kernel(const device half* A [[buffer(0)]],
                         const device half* B [[buffer(1)]],
                         device half* C [[buffer(2)]],
                         constant uint& N [[buffer(3)]],
                         constant half& alpha [[buffer(4)]],
                         constant half& beta [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= N || col >= N) return;
    
    half sum = 0.0h;
    for (uint k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}
"""

let metalSource32 = """
#include <metal_stdlib>
using namespace metal;

kernel void gemm32_kernel(const device float* A [[buffer(0)]],
                         const device float* B [[buffer(1)]],
                         device float* C [[buffer(2)]],
                         constant uint& N [[buffer(3)]],
                         constant float& alpha [[buffer(4)]],
                         constant float& beta [[buffer(5)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= N || col >= N) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < N; k++) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
}
"""

// Function to run benchmark for Float16 with Metal
@available(macOS 11.0, *)
func runBenchmarkFloat16Metal(iterations: Int, order: Int) {
    print("Testing precision: FP16 (Metal GPU)")
    
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("ERROR: Metal is not supported on this device")
        return
    }
    
    guard let commandQueue = device.makeCommandQueue() else {
        print("ERROR: Failed to create command queue")
        return
    }
    
    guard let library = try? device.makeLibrary(source: metalSource16, options: nil) else {
        print("ERROR: Failed to create Metal library")
        return
    }
    
    guard let function = library.makeFunction(name: "gemm16_kernel") else {
        print("ERROR: Failed to find kernel function")
        return
    }
    
    guard let computePipelineState = try? device.makeComputePipelineState(function: function) else {
        print("ERROR: Failed to create compute pipeline state")
        return
    }
    
    var dgemmTime: Double = 0
    let nelems = order * order
    
    // Initialize matrices on CPU
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
    
    // Create Metal buffers
    guard let bufferA = device.makeBuffer(bytes: A, length: nelems * MemoryLayout<Float16>.size, options: [.storageModeShared]) else {
        print("ERROR: Failed to create A buffer")
        return
    }
    guard let bufferB = device.makeBuffer(bytes: B, length: nelems * MemoryLayout<Float16>.size, options: [.storageModeShared]) else {
        print("ERROR: Failed to create B buffer")
        return
    }
    let bufferC = device.makeBuffer(bytes: C, length: nelems * MemoryLayout<Float16>.size, options: [.storageModeShared])!
    
    var orderConstant = UInt32(order)
    var alphaConstant = Float16(1.0)
    var betaConstant = Float16(1.0)
    
    let bufferOrder = device.makeBuffer(bytes: &orderConstant, length: MemoryLayout<UInt32>.size, options: [.storageModeShared])!
    let bufferAlpha = device.makeBuffer(bytes: &alphaConstant, length: MemoryLayout<Float16>.size, options: [.storageModeShared])!
    let bufferBeta = device.makeBuffer(bytes: &betaConstant, length: MemoryLayout<Float16>.size, options: [.storageModeShared])!
    
    var startTime: Double = 0
    
    // Benchmark loop
    for k in 0...iterations {
        if k == 1 {
            startTime = CFAbsoluteTimeGetCurrent()
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            print("ERROR: Failed to create command buffer")
            return
        }
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            print("ERROR: Failed to create compute encoder")
            return
        }
        
        computeEncoder.setComputePipelineState(computePipelineState)
        computeEncoder.setBuffer(bufferA, offset: 0, index: 0)
        computeEncoder.setBuffer(bufferB, offset: 0, index: 1)
        computeEncoder.setBuffer(bufferC, offset: 0, index: 2)
        computeEncoder.setBuffer(bufferOrder, offset: 0, index: 3)
        computeEncoder.setBuffer(bufferAlpha, offset: 0, index: 4)
        computeEncoder.setBuffer(bufferBeta, offset: 0, index: 5)
        
        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (order + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: (order + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    dgemmTime = CFAbsoluteTimeGetCurrent() - startTime
    let dgemmAve = dgemmTime / Double(iterations)
    
    //////////////////////////////////////////////////////////////////////
    // Analyze and output results
    //////////////////////////////////////////////////////////////////////
    
    let resultPointer = bufferC.contents().bindMemory(to: Float16.self, capacity: nelems)
    let results = Array(UnsafeBufferPointer(start: resultPointer, count: nelems))
    
    let checksum = results.reduce(Float16(0), +)
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

// Function to run benchmark for Float32 with Metal
func runBenchmarkFloat32Metal(iterations: Int, order: Int) {
    print("Testing precision: FP32 (Metal GPU)")
    
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("ERROR: Metal is not supported on this device")
        return
    }
    
    guard let commandQueue = device.makeCommandQueue() else {
        print("ERROR: Failed to create command queue")
        return
    }
    
    guard let library = try? device.makeLibrary(source: metalSource32, options: nil) else {
        print("ERROR: Failed to create Metal library")
        return
    }
    
    guard let function = library.makeFunction(name: "gemm32_kernel") else {
        print("ERROR: Failed to find kernel function")
        return
    }
    
    guard let computePipelineState = try? device.makeComputePipelineState(function: function) else {
        print("ERROR: Failed to create compute pipeline state")
        return
    }
    
    var dgemmTime: Double = 0
    let nelems = order * order
    
    // Initialize matrices on CPU
    var A = Array<Float32>(repeating: Float32(0), count: nelems)
    var B = Array<Float32>(repeating: Float32(0), count: nelems)
    var C = Array<Float32>(repeating: Float32(0), count: nelems)
    
    for i in 0..<order {
        for j in 0..<order {
            A[i * order + j] = Float32(i)
            B[i * order + j] = Float32(i)
            C[i * order + j] = Float32(0)
        }
    }
    
    // Create Metal buffers
    guard let bufferA = device.makeBuffer(bytes: A, length: nelems * MemoryLayout<Float32>.size, options: [.storageModeShared]) else {
        print("ERROR: Failed to create A buffer")
        return
    }
    guard let bufferB = device.makeBuffer(bytes: B, length: nelems * MemoryLayout<Float32>.size, options: [.storageModeShared]) else {
        print("ERROR: Failed to create B buffer")
        return
    }
    let bufferC = device.makeBuffer(bytes: C, length: nelems * MemoryLayout<Float32>.size, options: [.storageModeShared])!
    
    var orderConstant = UInt32(order)
    var alphaConstant = Float32(1.0)
    var betaConstant = Float32(1.0)
    
    let bufferOrder = device.makeBuffer(bytes: &orderConstant, length: MemoryLayout<UInt32>.size, options: [.storageModeShared])!
    let bufferAlpha = device.makeBuffer(bytes: &alphaConstant, length: MemoryLayout<Float32>.size, options: [.storageModeShared])!
    let bufferBeta = device.makeBuffer(bytes: &betaConstant, length: MemoryLayout<Float32>.size, options: [.storageModeShared])!
    
    var startTime: Double = 0
    
    // Benchmark loop
    for k in 0...iterations {
        if k == 1 {
            startTime = CFAbsoluteTimeGetCurrent()
        }
        
        guard let commandBuffer = commandQueue.makeCommandBuffer() else {
            print("ERROR: Failed to create command buffer")
            return
        }
        
        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else {
            print("ERROR: Failed to create compute encoder")
            return
        }
        
        computeEncoder.setComputePipelineState(computePipelineState)
        computeEncoder.setBuffer(bufferA, offset: 0, index: 0)
        computeEncoder.setBuffer(bufferB, offset: 0, index: 1)
        computeEncoder.setBuffer(bufferC, offset: 0, index: 2)
        computeEncoder.setBuffer(bufferOrder, offset: 0, index: 3)
        computeEncoder.setBuffer(bufferAlpha, offset: 0, index: 4)
        computeEncoder.setBuffer(bufferBeta, offset: 0, index: 5)
        
        let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
        let threadgroupsPerGrid = MTLSize(
            width: (order + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width,
            height: (order + threadsPerThreadgroup.height - 1) / threadsPerThreadgroup.height,
            depth: 1
        )
        
        computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
    
    dgemmTime = CFAbsoluteTimeGetCurrent() - startTime
    let dgemmAve = dgemmTime / Double(iterations)
    
    //////////////////////////////////////////////////////////////////////
    // Analyze and output results
    //////////////////////////////////////////////////////////////////////
    
    let resultPointer = bufferC.contents().bindMemory(to: Float32.self, capacity: nelems)
    let results = Array(UnsafeBufferPointer(start: resultPointer, count: nelems))
    
    let checksum = results.reduce(Float32(0), +)
    let forder = Double(order)
    let refChecksum = Float32(0.25 * forder * forder * forder * (forder - 1.0) * (forder - 1.0) * Double(iterations + 1))
    
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

func main() {
    print("Parallel Research Kernels")
    print("Swift Dense matrix-matrix multiplication: C += A x B (Multi-precision Metal GPU)")
    
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
    
    // Test supported precision types
    if #available(macOS 11.0, *) {
        runBenchmarkFloat16Metal(iterations: iterations, order: order)
    } else {
        print("Float16 not available on this macOS version")
    }
    
    runBenchmarkFloat32Metal(iterations: iterations, order: order)
    
    // Note: Metal doesn't have native FP64 support on most hardware
    print("Note: FP64 not supported in Metal on most Apple GPUs")
}

main()

