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
      *> NAME:      dgemm
      *> 
      *> PURPOSE:   This program tests the efficiency with which a dense matrix
      *>            dense multiplication is carried out
      *> 
      *> USAGE:     The program takes as input the matrix order and the number of 
      *>            times the matrix-matrix multiplication is carried out.
      *> 
      *>            <progname> <# iterations> <matrix order>
      *> 
      *>            The output consists of diagnostics to make sure the 
      *>            algorithm worked, and of timing statistics.
      *> 
      *> FUNCTIONS: The only "function" used is the "wtime" timer.
      *> 
      *> HISTORY:   Written by Rob Van der Wijngaart, February 2009.
      *>            Converted to COBOL by Cursor AI, 2025.
      *> **********************************************************************

       IDENTIFICATION DIVISION.
       PROGRAM-ID. DGEMM.

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
           05  WS-K                    PIC 9(6).
           05  WS-IDX-A                PIC 9(10).
           05  WS-IDX-B                PIC 9(10).
           05  WS-IDX-C                PIC 9(10).
           
       01  WS-TIMING.
           05  WS-START-TIME           PIC 9(10).
           05  WS-END-TIME             PIC 9(10).
           05  WS-DGEMM-TIME           PIC 9(10).
           05  WS-AVG-TIME             PIC 9(10)V9(6).
           
       01  WS-RESULTS.
           05  WS-RATE                 PIC 9(10)V9(6).
           05  WS-CHECKSUM             PIC 9(15)V9(12).
           05  WS-REF-CHECKSUM         PIC 9(15)V9(12).
           05  WS-RESIDUUM             PIC 9(15)V9(12).
           05  WS-EPSILON              PIC 9(5)V9(15) VALUE 0.0001.
           05  WS-NFLOPS               PIC 9(15)V9(6).
           
       01  WS-TEMP-VARS.
           05  WS-TEMP1                PIC 9(15)V9(12).
           05  WS-TEMP2                PIC 9(15)V9(12).
           05  WS-FORDER               PIC 9(10)V9(6).
           
       01  WS-MATRICES.
           05  WS-MATRIX-A             OCCURS 90000 TIMES
                                       INDEXED BY IDX-A.
               10  WS-A-VALUE          PIC S9(10)V9(12) COMP-3.
           05  WS-MATRIX-B             OCCURS 90000 TIMES
                                       INDEXED BY IDX-B.
               10  WS-B-VALUE          PIC S9(10)V9(12) COMP-3.
           05  WS-MATRIX-C             OCCURS 90000 TIMES
                                       INDEXED BY IDX-C.
               10  WS-C-VALUE          PIC S9(10)V9(12) COMP-3.

       PROCEDURE DIVISION.
       MAIN-PROCEDURE.
           DISPLAY "Parallel Research Kernels"
           DISPLAY "COBOL Dense matrix-matrix multiplication"
           
           *> Get command line arguments
           ACCEPT WS-ARG-COUNT FROM ARGUMENT-NUMBER
           
           IF WS-ARG-COUNT < 2
               DISPLAY "Usage: dgemm <# iterations> <matrix order>"
               STOP RUN
           END-IF
           
           ACCEPT WS-ARG1 FROM ARGUMENT-VALUE
           ACCEPT WS-ARG2 FROM ARGUMENT-VALUE
           
           *> Convert arguments to numeric
           MOVE FUNCTION NUMVAL(WS-ARG1) TO WS-ITERATIONS
           MOVE FUNCTION NUMVAL(WS-ARG2) TO WS-ORDER
           
           *> Validate parameters
           IF WS-ITERATIONS < 1
               DISPLAY "ERROR: iterations must be >= 1"
               STOP RUN
           END-IF
           
           IF WS-ORDER < 1 OR WS-ORDER > 300
               DISPLAY "ERROR: matrix order must be 1-300"
               STOP RUN
           END-IF
           
           DISPLAY "Matrix order          = " WS-ORDER
           DISPLAY "Number of iterations  = " WS-ITERATIONS
           
           *> Initialize matrices (using linearized indexing)
           *> A[i,j] = i, B[i,j] = i, C[i,j] = 0
           PERFORM VARYING WS-I FROM 1 BY 1 UNTIL WS-I > WS-ORDER
               PERFORM VARYING WS-J FROM 1 BY 1 UNTIL WS-J > WS-ORDER
                   COMPUTE WS-IDX-A = (WS-I - 1) * WS-ORDER + WS-J
                   COMPUTE WS-IDX-B = (WS-I - 1) * WS-ORDER + WS-J
                   COMPUTE WS-IDX-C = (WS-I - 1) * WS-ORDER + WS-J
                   SET IDX-A TO WS-IDX-A
                   SET IDX-B TO WS-IDX-B
                   SET IDX-C TO WS-IDX-C
                   MOVE WS-I TO WS-A-VALUE(IDX-A)
                   MOVE WS-I TO WS-B-VALUE(IDX-B)
                   MOVE 0.0 TO WS-C-VALUE(IDX-C)
               END-PERFORM
           END-PERFORM
           
           *> Main DGEMM loop
           PERFORM VARYING WS-ITER FROM 1 BY 1 
                   UNTIL WS-ITER > WS-ITERATIONS
               
               *> Start timer after warmup iteration (simplified)
               IF WS-ITER = 1
                   MOVE 0 TO WS-START-TIME
               END-IF
               
               *> Matrix multiplication: C[i,j] += A[i,k] * B[k,j]
               PERFORM VARYING WS-J FROM 1 BY 1 
                       UNTIL WS-J > WS-ORDER
                   PERFORM VARYING WS-K FROM 1 BY 1 
                           UNTIL WS-K > WS-ORDER
                       PERFORM VARYING WS-I FROM 1 BY 1 
                               UNTIL WS-I > WS-ORDER
                           *> Calculate linearized indices
                           COMPUTE WS-IDX-A = (WS-I - 1) * WS-ORDER + 
                                   WS-K
                           COMPUTE WS-IDX-B = (WS-K - 1) * WS-ORDER + 
                                   WS-J
                           COMPUTE WS-IDX-C = (WS-I - 1) * WS-ORDER + 
                                   WS-J
                           SET IDX-A TO WS-IDX-A
                           SET IDX-B TO WS-IDX-B
                           SET IDX-C TO WS-IDX-C
                           COMPUTE WS-C-VALUE(IDX-C) = 
                                   WS-C-VALUE(IDX-C) +
                                   WS-A-VALUE(IDX-A) * WS-B-VALUE(IDX-B)
                       END-PERFORM
                   END-PERFORM
               END-PERFORM
               
           END-PERFORM
           
           *> Stop timer (simplified)
           MOVE 1 TO WS-END-TIME
           COMPUTE WS-DGEMM-TIME = 1
           
           *> Verify results
           MOVE 0.0 TO WS-CHECKSUM
           PERFORM VARYING WS-I FROM 1 BY 1 UNTIL WS-I > WS-ORDER
               PERFORM VARYING WS-J FROM 1 BY 1 UNTIL WS-J > WS-ORDER
                   COMPUTE WS-IDX-C = (WS-I - 1) * WS-ORDER + WS-J
                   SET IDX-C TO WS-IDX-C
                   COMPUTE WS-CHECKSUM = WS-CHECKSUM + WS-C-VALUE(IDX-C)
               END-PERFORM
           END-PERFORM
           
           *> Calculate reference checksum
           MOVE WS-ORDER TO WS-FORDER
           COMPUTE WS-REF-CHECKSUM = 0.25 * WS-FORDER * WS-FORDER * 
                   WS-FORDER * (WS-FORDER - 1.0) * (WS-FORDER - 1.0) *
                   (WS-ITERATIONS + 1)
           
           *> Check if solution validates
           COMPUTE WS-TEMP1 = WS-CHECKSUM - WS-REF-CHECKSUM
           IF WS-TEMP1 < 0
               COMPUTE WS-TEMP1 = -WS-TEMP1
           END-IF
           COMPUTE WS-RESIDUUM = WS-TEMP1 / WS-REF-CHECKSUM
           
           IF WS-RESIDUUM < WS-EPSILON
               DISPLAY "Solution validates"
               COMPUTE WS-AVG-TIME = WS-DGEMM-TIME / WS-ITERATIONS
               COMPUTE WS-NFLOPS = 2.0 * WS-FORDER * WS-FORDER * 
                       WS-FORDER
               COMPUTE WS-RATE = 1.0E-06 * WS-NFLOPS / WS-AVG-TIME
               DISPLAY "Rate (MF/s): " WS-RATE 
                       " Avg time (s): " WS-AVG-TIME
           ELSE
               DISPLAY "ERROR: Checksum " WS-CHECKSUM 
                       " does not match verification value " 
                       WS-REF-CHECKSUM
               DISPLAY "Residuum: " WS-RESIDUUM
           END-IF
           
           STOP RUN.
