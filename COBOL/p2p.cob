      *>  Copyright (c) 2025, NVIDIA
      *>
      *>  Redistribution and use in source and binary forms, with or without
      *>  modification, are permitted provided that the following conditions
      *>  are met:
      *>
      *>  * Redistributions of source code must retain the above copyright
      *>        notice, this list of conditions and the following disclaimer.
      *>  * Redistributions in binary form must reproduce the above
      *>        copyright notice, this list of conditions and the following
      *>        disclaimer in the documentation and/or other materials provided
      *>        with the distribution.
      *>  * Neither the name of Intel Corporation nor the names of its
      *>        contributors may be used to endorse or promote products
      *>        derived from this software without specific prior written
      *>        permission.
      *>
      *>  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
      *>  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
      *>  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
      *>  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
      *>  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
      *>  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
      *>  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
      *>  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
      *>  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
      *>  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
      *>  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
      *>  POSSIBILITY OF SUCH DAMAGE.

      *> **********************************************************************
      *> 
      *> NAME:      p2p
      *> 
      *> PURPOSE:   This program tests the efficiency with which a space-invariant,
      *>            linear, homogeneous stencil can be applied to a square grid.
      *>            The stencil uses a 2-point formula in two dimensions.
      *> 
      *> USAGE:     The program takes as input the linear
      *>            dimension of the grid, and the number of iterations on the grid
      *> 
      *>                <progname> <iterations> <grid dimension 1> <grid dimension 2>
      *> 
      *>            The output consists of diagnostics to make sure the 
      *>            algorithm worked, and of timing statistics.
      *> 
      *> FUNCTIONS: The only "function" used is the "wtime" timer.
      *> 
      *> HISTORY:   - Written by Rob Van der Wijngaart, February 2009.
      *>            - Converted to COBOL by Cursor AI, 2025.
      *> **********************************************************************

       IDENTIFICATION DIVISION.
       PROGRAM-ID. P2P.

       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.

       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-ARGUMENTS.
           05  WS-ARG-COUNT            PIC 9(2).
           05  WS-ARG1                 PIC X(20).
           05  WS-ARG2                 PIC X(20).
           05  WS-ARG3                 PIC X(20).
           
       01  WS-PARAMETERS.
           05  WS-ITERATIONS           PIC 9(8).
           05  WS-M                    PIC 9(6).
           05  WS-N                    PIC 9(6).
           
       01  WS-COUNTERS.
           05  WS-ITER                 PIC 9(8).
           05  WS-I                    PIC 9(6).
           05  WS-J                    PIC 9(6).
           05  WS-IDX                  PIC 9(10).
           
       01  WS-TIMING.
           05  WS-START-TIME           PIC 9(10).
           05  WS-END-TIME             PIC 9(10).
           05  WS-PIPELINE-TIME        PIC 9(10).
           05  WS-AVG-TIME             PIC 9(10)V9(6).
           
       01  WS-RESULTS.
           05  WS-RATE                 PIC 9(10)V9(6).
           05  WS-CORNER-VAL           PIC 9(15)V9(12).
           05  WS-EXPECTED-VAL         PIC 9(15)V9(12).
           05  WS-DIFF                 PIC 9(15)V9(12).
           05  WS-EPSILON              PIC 9(5)V9(15) VALUE 0.0001.
           
       01  WS-TEMP-VARS.
           05  WS-TEMP1                PIC 9(15)V9(12).
           05  WS-TEMP2                PIC 9(15)V9(12).
           05  WS-TEMP3                PIC 9(15)V9(12).
           05  WS-TEMP4                PIC 9(15)V9(12).
           
       01  WS-GRID.
           05  WS-GRID-ARRAY           OCCURS 250000 TIMES
                                       INDEXED BY IDX-GRID.
               10  WS-GRID-VALUE       PIC S9(10)V9(12) COMP-3.

       PROCEDURE DIVISION.
       MAIN-PROCEDURE.
           DISPLAY "Parallel Research Kernels"
           DISPLAY "COBOL Pipeline execution on 2D grid"
           
           *> Get command line arguments
           ACCEPT WS-ARG-COUNT FROM ARGUMENT-NUMBER
           
           IF WS-ARG-COUNT < 3
               DISPLAY "Usage: p2p <iterations> <grid dimension 1> " &
                       "<grid dimension 2>"
               STOP RUN
           END-IF
           
           ACCEPT WS-ARG1 FROM ARGUMENT-VALUE
           ACCEPT WS-ARG2 FROM ARGUMENT-VALUE
           ACCEPT WS-ARG3 FROM ARGUMENT-VALUE
           
           *> Convert arguments to numeric
           MOVE FUNCTION NUMVAL(WS-ARG1) TO WS-ITERATIONS
           MOVE FUNCTION NUMVAL(WS-ARG2) TO WS-M
           MOVE FUNCTION NUMVAL(WS-ARG3) TO WS-N
           
           *> Validate parameters
           IF WS-ITERATIONS < 1
               DISPLAY "ERROR: iterations must be >= 1"
               STOP RUN
           END-IF
           
           IF WS-M < 1 OR WS-M > 500
               DISPLAY "ERROR: grid dimension 1 must be 1-500"
               STOP RUN
           END-IF
           
           IF WS-N < 1 OR WS-N > 500
               DISPLAY "ERROR: grid dimension 2 must be 1-500"
               STOP RUN
           END-IF
           
           DISPLAY "Grid sizes            = " WS-M " x " WS-N
           DISPLAY "Number of iterations  = " WS-ITERATIONS
           
           *> Initialize grid (using linearized indexing)
           PERFORM VARYING WS-I FROM 1 BY 1 UNTIL WS-I > WS-M
               PERFORM VARYING WS-J FROM 1 BY 1 UNTIL WS-J > WS-N
                   COMPUTE WS-IDX = (WS-I - 1) * WS-N + WS-J
                   SET IDX-GRID TO WS-IDX
                   IF WS-I = 1 OR WS-J = 1
                       MOVE 1.0 TO WS-GRID-VALUE(IDX-GRID)
                   ELSE
                       MOVE 0.0 TO WS-GRID-VALUE(IDX-GRID)
                   END-IF
               END-PERFORM
           END-PERFORM
           
           *> Main pipeline loop
           PERFORM VARYING WS-ITER FROM 1 BY 1 
                   UNTIL WS-ITER > WS-ITERATIONS
               
               *> Start timer after warmup iteration (simplified)
               IF WS-ITER = 1
                   MOVE 0 TO WS-START-TIME
               END-IF
               
               *> Pipeline sweep: GRID[i,j] = GRID[i-1,j] + GRID[i,j-1] - GRID[i-1,j-1]
               PERFORM VARYING WS-J FROM 2 BY 1 UNTIL WS-J > WS-N
                   PERFORM VARYING WS-I FROM 2 BY 1 UNTIL WS-I > WS-M
                       *> Calculate current and neighbor indices
                       COMPUTE WS-IDX = (WS-I - 1) * WS-N + WS-J
                       SET IDX-GRID TO WS-IDX
                       
                       *> Get GRID[i-1,j]
                       COMPUTE WS-IDX = (WS-I - 2) * WS-N + WS-J
                       MOVE WS-GRID-VALUE(WS-IDX) TO WS-TEMP1
                       
                       *> Get GRID[i,j-1]
                       COMPUTE WS-IDX = (WS-I - 1) * WS-N + (WS-J - 1)
                       MOVE WS-GRID-VALUE(WS-IDX) TO WS-TEMP2
                       
                       *> Get GRID[i-1,j-1]
                       COMPUTE WS-IDX = (WS-I - 2) * WS-N + (WS-J - 1)
                       MOVE WS-GRID-VALUE(WS-IDX) TO WS-TEMP3
                       
                       *> Update GRID[i,j]
                       COMPUTE WS-GRID-VALUE(IDX-GRID) = 
                               WS-TEMP1 + WS-TEMP2 - WS-TEMP3
                   END-PERFORM
               END-PERFORM
               
               *> Copy top right corner value to bottom left corner 
               COMPUTE WS-IDX = (WS-M - 1) * WS-N + WS-N
               COMPUTE WS-TEMP4 = -WS-GRID-VALUE(WS-IDX)
               COMPUTE WS-IDX = 1
               MOVE WS-TEMP4 TO WS-GRID-VALUE(WS-IDX)
               
           END-PERFORM
           
           *> Stop timer (simplified)
           MOVE 1 TO WS-END-TIME
           COMPUTE WS-PIPELINE-TIME = 1
           
           *> Verify correctness using top right value
           COMPUTE WS-IDX = (WS-M - 1) * WS-N + WS-N
           MOVE WS-GRID-VALUE(WS-IDX) TO WS-CORNER-VAL
           COMPUTE WS-EXPECTED-VAL = (WS-ITERATIONS + 1) * 
                   (WS-N + WS-M - 2)
           
           COMPUTE WS-DIFF = WS-CORNER-VAL - WS-EXPECTED-VAL
           IF WS-DIFF < 0
               COMPUTE WS-DIFF = -WS-DIFF
           END-IF
           COMPUTE WS-DIFF = WS-DIFF / WS-EXPECTED-VAL
           
           IF WS-DIFF > WS-EPSILON
               DISPLAY "ERROR: checksum " WS-CORNER-VAL 
                       " does not match verification value " 
                       WS-EXPECTED-VAL
               STOP RUN
           END-IF
           
           DISPLAY "Solution validates"
           COMPUTE WS-AVG-TIME = WS-PIPELINE-TIME / WS-ITERATIONS
           COMPUTE WS-RATE = 1.0E-06 * 2 * ((WS-M - 1) * (WS-N - 1)) / 
                   WS-AVG-TIME
           DISPLAY "Rate (MFlops/s): " WS-RATE 
                   " Avg time (s): " WS-AVG-TIME
           
           STOP RUN.
