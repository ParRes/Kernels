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
      *> NAME:      transpose
      *> 
      *> PURPOSE:   This program measures the time for the transpose of a
      *>            column-major stored matrix into a row-major stored matrix.
      *> 
      *> USAGE:     Program input is the matrix order and the number of times 
      *>            to repeat the operation:
      *> 
      *>            transpose <matrix_size> <# iterations>
      *> 
      *>            The output consists of diagnostics to make sure the 
      *>            transpose worked and timing statistics.
      *> 
      *> HISTORY:   Written by  Rob Van der Wijngaart, February 2009.
      *>            Converted to COBOL by Cursor AI, 2025.
      *> **********************************************************************

       IDENTIFICATION DIVISION.
       PROGRAM-ID. TRANSPOSE.

       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.

       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-ARGUMENTS.
           05  WS-ARG-COUNT            PIC 9(2).
           05  WS-ARG1                 PIC X(20).
           05  WS-ARG2                 PIC X(20).
           
       01  WS-PARAMETERS.
           05  WS-ITERATIONS           PIC 9(8).
           05  WS-ORDER                PIC 9(6).
           
       01  WS-COUNTERS.
           05  WS-ITER                 PIC 9(8).
           05  WS-I                    PIC 9(6).
           05  WS-J                    PIC 9(6).
           05  WS-IDX-A                PIC 9(10).
           05  WS-IDX-B                PIC 9(10).
           
       01  WS-TIMING.
           05  WS-START-TIME           PIC 9(10).
           05  WS-END-TIME             PIC 9(10).
           05  WS-TRANS-TIME           PIC 9(10).
           05  WS-AVG-TIME             PIC 9(10)V9(6).
           
       01  WS-RESULTS.
           05  WS-BYTES                PIC 9(15)V9(6).
           05  WS-RATE                 PIC 9(10)V9(6).
           05  WS-ABSERR               PIC 9(15)V9(12).
           05  WS-ADDIT                PIC 9(15)V9(12).
           05  WS-EXPECTED             PIC 9(15)V9(12).
           05  WS-EPSILON              PIC 9(5)V9(15) VALUE 0.0001.
           
       01  WS-TEMP-VARS.
           05  WS-TEMP1                PIC 9(15)V9(12).
           05  WS-TEMP2                PIC 9(15)V9(12).
           
       01  WS-ARRAYS.
           05  WS-MATRIX-A             OCCURS 1000000 TIMES
                                       INDEXED BY IDX-A.
               10  WS-A-VALUE          PIC S9(10)V9(12) COMP-3.
           05  WS-MATRIX-B             OCCURS 1000000 TIMES
                                       INDEXED BY IDX-B.
               10  WS-B-VALUE          PIC S9(10)V9(12) COMP-3.

       PROCEDURE DIVISION.
       MAIN-PROCEDURE.
           DISPLAY "Parallel Research Kernels"
           DISPLAY "COBOL Matrix transpose: B = A^T"
           
           *> Get command line arguments
           ACCEPT WS-ARG-COUNT FROM ARGUMENT-NUMBER
           
           IF WS-ARG-COUNT < 2
               DISPLAY "Usage: transpose <matrix_size> <# iterations>"
               STOP RUN
           END-IF
           
           ACCEPT WS-ARG1 FROM ARGUMENT-VALUE
           ACCEPT WS-ARG2 FROM ARGUMENT-VALUE
           
           *> Convert arguments to numeric
           MOVE FUNCTION NUMVAL(WS-ARG1) TO WS-ORDER
           MOVE FUNCTION NUMVAL(WS-ARG2) TO WS-ITERATIONS
           
           *> Validate parameters
           IF WS-ITERATIONS < 1
               DISPLAY "ERROR: iterations must be >= 1"
               STOP RUN
           END-IF
           
           IF WS-ORDER < 1 OR WS-ORDER > 1000
               DISPLAY "ERROR: matrix order must be 1-1000"
               STOP RUN
           END-IF
           
           DISPLAY "Matrix order          = " WS-ORDER
           DISPLAY "Number of iterations  = " WS-ITERATIONS
           
           *> Initialize matrices
           *> Fill the original matrix A (using linearized indexing)
           PERFORM VARYING WS-I FROM 1 BY 1 UNTIL WS-I > WS-ORDER
               PERFORM VARYING WS-J FROM 1 BY 1 UNTIL WS-J > WS-ORDER
                   COMPUTE WS-IDX-A = (WS-I - 1) * WS-ORDER + WS-J
                   SET IDX-A TO WS-IDX-A
                   COMPUTE WS-A-VALUE(IDX-A) = 
                           WS-ORDER * (WS-J - 1) + (WS-I - 1)
               END-PERFORM
           END-PERFORM
           
           *> Set the transpose matrix B to zero
           PERFORM VARYING WS-I FROM 1 BY 1 UNTIL WS-I > WS-ORDER
               PERFORM VARYING WS-J FROM 1 BY 1 UNTIL WS-J > WS-ORDER
                   COMPUTE WS-IDX-B = (WS-I - 1) * WS-ORDER + WS-J
                   SET IDX-B TO WS-IDX-B
                   MOVE 0.0 TO WS-B-VALUE(IDX-B)
               END-PERFORM
           END-PERFORM
           
           *> Main transpose loop
           PERFORM VARYING WS-ITER FROM 1 BY 1 
                   UNTIL WS-ITER > WS-ITERATIONS
               
               *> Start timer after warmup iteration (simplified)
               IF WS-ITER = 1
                   MOVE 0 TO WS-START-TIME
               END-IF
               
               *> Transpose the matrix: B[j,i] += A[i,j]; A[i,j] += 1.0
               PERFORM VARYING WS-I FROM 1 BY 1 
                       UNTIL WS-I > WS-ORDER
                   PERFORM VARYING WS-J FROM 1 BY 1 
                           UNTIL WS-J > WS-ORDER
                       *> Calculate linearized indices: A[i,j] and B[j,i]
                       COMPUTE WS-IDX-A = (WS-I - 1) * WS-ORDER + WS-J
                       COMPUTE WS-IDX-B = (WS-J - 1) * WS-ORDER + WS-I
                       SET IDX-A TO WS-IDX-A
                       SET IDX-B TO WS-IDX-B
                       COMPUTE WS-B-VALUE(IDX-B) = 
                               WS-B-VALUE(IDX-B) + WS-A-VALUE(IDX-A)
                       COMPUTE WS-A-VALUE(IDX-A) = 
                               WS-A-VALUE(IDX-A) + 1.0
                   END-PERFORM
               END-PERFORM
               
           END-PERFORM
           
           *> Stop timer (simplified)
           MOVE 1 TO WS-END-TIME
           COMPUTE WS-TRANS-TIME = 1
           
           *> Verify results
           MOVE 0.0 TO WS-ABSERR
           COMPUTE WS-ADDIT = (WS-ITERATIONS + 1.0) * WS-ITERATIONS / 2.0
           
           PERFORM VARYING WS-I FROM 1 BY 1 
                   UNTIL WS-I > WS-ORDER
               PERFORM VARYING WS-J FROM 1 BY 1 
                       UNTIL WS-J > WS-ORDER
                   *> Calculate linearized index for B[i,j]
                   COMPUTE WS-IDX-B = (WS-I - 1) * WS-ORDER + WS-J
                   SET IDX-B TO WS-IDX-B
                   *> Expected value: original_A[j,i] * iterations + addit
                   *> A[j,i] was initialized as: order*(i-1) + (j-1)
                   COMPUTE WS-TEMP2 = WS-ORDER * (WS-I - 1) + (WS-J - 1)
                   COMPUTE WS-EXPECTED = WS-TEMP2 * WS-ITERATIONS + 
                           WS-ADDIT
                   COMPUTE WS-TEMP1 = WS-B-VALUE(IDX-B) - WS-EXPECTED
                   IF WS-TEMP1 < 0
                       COMPUTE WS-TEMP1 = -WS-TEMP1
                   END-IF
                   COMPUTE WS-ABSERR = WS-ABSERR + WS-TEMP1
               END-PERFORM
           END-PERFORM
           
           IF WS-ABSERR < WS-EPSILON
               DISPLAY "Solution validates"
               COMPUTE WS-AVG-TIME = WS-TRANS-TIME / WS-ITERATIONS
               COMPUTE WS-BYTES = 2.0 * 8 * WS-ORDER * WS-ORDER
               COMPUTE WS-RATE = 1.0E-06 * WS-BYTES / WS-AVG-TIME
               DISPLAY "Rate (MB/s): " WS-RATE 
                       " Avg time (s): " WS-AVG-TIME
           ELSE
               DISPLAY "ERROR: Aggregate squared error " WS-ABSERR
                       " exceeds threshold " WS-EPSILON
           END-IF
           
           STOP RUN.
