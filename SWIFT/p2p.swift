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
/// NAME:    Pipeline
///
/// PURPOSE: This program tests the efficiency with which point-to-point
///          synchronization can be carried out. It does so by executing
///          a pipelined algorithm on an m*n grid. The first array dimension
///          is distributed among the threads (stripwise decomposition).
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
///          Converted to Swift by Cursor AI, 2025.
///
//////////////////////////////////////////////////////////////////////

import Foundation

func main() {
    print("Parallel Research Kernels")
    print("Swift pipeline execution on 2D grid")
    
    //////////////////////////////////////////////////////////////////////
    /// Read and test input parameters
    //////////////////////////////////////////////////////////////////////
    
    let arguments = CommandLine.arguments
    
    guard arguments.count == 4 else {
        print("Usage: swift p2p.swift <# iterations> <first array dimension> <second array dimension>")
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
    
    //////////////////////////////////////////////////////////////////////
    // Allocate space and initialize grid
    //////////////////////////////////////////////////////////////////////
    
    var grid = Array(repeating: Array(repeating: 0.0, count: n), count: m)
    
    // Initialize grid boundaries
    for j in 0..<n {
        grid[0][j] = Double(j)
    }
    for i in 0..<m {
        grid[i][0] = Double(i)
    }
    
    var startTime = 0.0
    
    for k in 0...iterations {
        
        // Start timer after warmup iteration
        if k == 1 {
            startTime = CFAbsoluteTimeGetCurrent()
        }
        
        // Execute pipeline algorithm
        for i in 1..<m {
            for j in 1..<n {
                grid[i][j] = grid[i-1][j] + grid[i][j-1] - grid[i-1][j-1]
            }
        }
        
        // Copy top right corner value to bottom left corner to create dependency
        grid[0][0] = -grid[m-1][n-1]
    }
    
    let pipelineTime = CFAbsoluteTimeGetCurrent() - startTime
    
    //////////////////////////////////////////////////////////////////////
    /// Analyze and output results
    //////////////////////////////////////////////////////////////////////
    
    let epsilon = 1.0e-8
    
    // Verify correctness, using top right value
    let cornerVal = Double((iterations + 1) * (n + m - 2))
    if abs(grid[m-1][n-1] - cornerVal) / cornerVal < epsilon {
        print("Solution validates")
        let avgtime = pipelineTime / Double(iterations)
        let rate = 1.0e-6 * 2.0 * Double(m-1) * Double(n-1) / avgtime
        print(String(format: "Rate (MFlops/s): %.6f; Avg time (s): %.6f", rate, avgtime))
    } else {
        print("ERROR: checksum \(grid[m-1][n-1]) does not match verification value \(cornerVal)")
        print("ERROR: solution did not validate")
        exit(1)
    }
}

main()
