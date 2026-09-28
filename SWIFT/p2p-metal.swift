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
/// NAME:    p2p-metal
///
/// PURPOSE: This program tests the efficiency with which point-to-point
///          synchronization can be carried out. It does so by executing
///          a pipelined algorithm on an m*n grid using Metal GPU compute.
///
/// USAGE:   The program takes as input the
///          dimensions of the grid, and the number of iterations on the grid
///
///          <progname> <iterations> <m> <n>
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

kernel void p2p_kernel(device float* grid [[buffer(0)]],
                      constant uint& n [[buffer(1)]],
                      constant uint& diagonal [[buffer(2)]],
                      uint thread_id [[thread_position_in_grid]]) {
    uint j = thread_id + 1; // thread_id maps to j coordinate (1-based)
    uint i = diagonal;
    
    // Check bounds for diagonal sweep (following OpenCL logic exactly)
    if (j >= max(2u, i - n + 2) && j <= min(i, n)) {
        uint x = i - j + 2 - 1;  // Convert to 0-based x coordinate
        uint y = j - 1;          // Convert to 0-based y coordinate
        
        // Additional bounds check
        if (x >= 1 && x < n && y >= 1 && y < n) {
            grid[x * n + y] = grid[(x-1) * n + y] 
                            + grid[x * n + (y-1)] 
                            - grid[(x-1) * n + (y-1)];
        }
    }
}
"""

func main() {
    print("Parallel Research Kernels")
    print("Swift pipeline execution on 2D grid (Metal GPU)")
    
    //////////////////////////////////////////////////////////////////////
    /// Read and test input parameters
    //////////////////////////////////////////////////////////////////////
    
    let arguments = CommandLine.arguments
    
    guard arguments.count == 4 else {
        print("Usage: swift p2p-metal.swift <# iterations> <first array dimension> <second array dimension>")
        exit(1)
    }
    
    guard let iterations = Int(arguments[1]), iterations >= 1 else {
        print("ERROR: iterations must be >= 1")
        exit(1)
    }
    
    guard let m = Int(arguments[2]), m >= 1 else {
        print("ERROR: array dimension must be >= 1")
        exit(1)
    }
    
    guard let n = Int(arguments[3]), n >= 1 else {
        print("ERROR: array dimension must be >= 1")
        exit(1)
    }
    
    print("Grid sizes               = \(m) * \(n)")
    print("Number of iterations     = \(iterations)")
    
    // For the OpenCL algorithm, we need to use the larger dimension
    let gridSize = max(m, n)
    
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
    
    guard let function = library.makeFunction(name: "p2p_kernel") else {
        print("ERROR: Failed to find kernel function")
        exit(1)
    }
    
    guard let computePipelineState = try? device.makeComputePipelineState(function: function) else {
        print("ERROR: Failed to create compute pipeline state")
        exit(1)
    }
    
    //////////////////////////////////////////////////////////////////////
    // Allocate space and initialize grid
    //////////////////////////////////////////////////////////////////////
    
    var grid = Array(repeating: Float(0.0), count: m * n)
    
    // Initialize grid boundaries
    for j in 0..<n {
        grid[0 * n + j] = Float(j)
    }
    for i in 0..<m {
        grid[i * n + 0] = Float(i)
    }
    
    var nConstant = UInt32(gridSize)
    
    // Create Metal buffers
    guard let bufferGrid = device.makeBuffer(bytes: grid, length: m * n * MemoryLayout<Float>.size, options: [.storageModeShared]) else {
        print("ERROR: Failed to create grid buffer")
        exit(1)
    }
    
    guard let bufferN = device.makeBuffer(bytes: &nConstant, length: MemoryLayout<UInt32>.size, options: [.storageModeShared]) else {
        print("ERROR: Failed to create n buffer")
        exit(1)
    }
    
    //////////////////////////////////////////////////////////////////////
    // Execute computation
    //////////////////////////////////////////////////////////////////////
    
    var startTime = 0.0
    
    for k in 0...iterations {
        
        // Start timer after warmup iteration
        if k == 1 {
            startTime = CFAbsoluteTimeGetCurrent()
        }
        
        // Execute pipeline algorithm using diagonal sweep pattern (like OpenCL version)
        for i in 2...(2*gridSize-2) {
            var diagonalConstant = UInt32(i)
            
            guard let bufferDiagonal = device.makeBuffer(bytes: &diagonalConstant, length: MemoryLayout<UInt32>.size, options: [.storageModeShared]) else {
                print("ERROR: Failed to create diagonal buffer")
                exit(1)
            }
            
            // Create command buffer for this diagonal
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
            computeEncoder.setBuffer(bufferGrid, offset: 0, index: 0)
            computeEncoder.setBuffer(bufferN, offset: 0, index: 1)
            computeEncoder.setBuffer(bufferDiagonal, offset: 0, index: 2)
            
            // Calculate thread group sizes for 1D dispatch
            let threadsPerThreadgroup = MTLSize(width: min(computePipelineState.maxTotalThreadsPerThreadgroup, gridSize), height: 1, depth: 1)
            let threadgroupsPerGrid = MTLSize(width: (gridSize + threadsPerThreadgroup.width - 1) / threadsPerThreadgroup.width, height: 1, depth: 1)
            
            // Dispatch threads
            computeEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadsPerThreadgroup)
            computeEncoder.endEncoding()
            
            // Execute and wait for completion (synchronization barrier)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        
        // Copy top right corner value to bottom left corner to create dependency
        let resultPointer = bufferGrid.contents().bindMemory(to: Float.self, capacity: m * n)
        let topRightValue = -resultPointer[(m-1) * n + (n-1)]
        resultPointer[0 * n + 0] = topRightValue
    }
    
    let pipelineTime = CFAbsoluteTimeGetCurrent() - startTime
    
    //////////////////////////////////////////////////////////////////////
    /// Analyze and output results
    //////////////////////////////////////////////////////////////////////
    
    // Copy results back from GPU
    let resultPointer = bufferGrid.contents().bindMemory(to: Float.self, capacity: m * n)
    let results = Array(UnsafeBufferPointer(start: resultPointer, count: m * n))
    
    let epsilon = 1.0e-6  // Relaxed for Float32
    
    // Verify correctness, using top right value  
    let cornerVal = Double((iterations + 1) * (2 * gridSize - 2))
    let observedCornerVal = Double(results[(m-1) * n + (n-1)])
    
    print("DEBUG: gridSize=\(gridSize), m=\(m), n=\(n)")
    print("DEBUG: Expected corner value: \(cornerVal)")
    print("DEBUG: Observed corner value: \(observedCornerVal)")
    print("DEBUG: Grid corner values: [\(results[0]), \(results[n-1]), \(results[(m-1)*n]), \(results[(m-1)*n+(n-1)])]")
    
    if abs(observedCornerVal - cornerVal) / cornerVal < epsilon {
        print("Solution validates")
        let avgtime = pipelineTime / Double(iterations)
        let rate = 1.0e-6 * 2.0 * Double(m-1) * Double(n-1) / avgtime
        print(String(format: "Rate (MFlops/s): %.6f; Avg time (s): %.6f", rate, avgtime))
    } else {
        print("ERROR: checksum \(observedCornerVal) does not match verification value \(cornerVal)")
        print("ERROR: solution did not validate")
        exit(1)
    }
}

main()
