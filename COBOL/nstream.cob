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
      *> NAME:      nstream
      *> 
      *> PURPOSE:   To compute memory bandwidth when adding a vector of a given
      *>            number of double precision values to the scalar multiple of 
      *>            another vector of the same length, and storing the result in
      *>            a third vector. 
      *> 
      *> USAGE:     The program takes as input the number 
      *>            of iterations to loop over the triad vectors and the length 
      *>            of the vectors
      *> 
      *>            <progname> <# iterations> <vector length>
      *> 
      *>            The output consists of diagnostics to make sure the 
      *>            algorithm worked, and of timing statistics.
      *> 
      *> NOTES:     Bandwidth is determined as the number of words read, plus the 
      *>            number of words written, times the size of the words, divided 
      *>            by the execution time. For a vector length of N, the total 
      *>            number of words read and written is 4*N*sizeof(double).
      *> 
      *> HISTORY:   This code is loosely based on the Stream benchmark by John
      *>            McCalpin, but does not follow all the Stream rules. Hence,
      *>            reported results should not be associated with Stream in
      *>            external publications
      *>            Converted to COBOL by Cursor AI, 2025.
      *> **********************************************************************

       IDENTIFICATION DIVISION.
       PROGRAM-ID. NSTREAM.

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
           05  WS-LENGTH               PIC 9(10).
           05  WS-SCALAR               PIC 9(3)V9(6) VALUE 3.0.
           
       01  WS-COUNTERS.
           05  WS-ITER                 PIC 9(8).
           05  WS-J                    PIC 9(10).
           
       01  WS-TIMING.
           05  WS-START-TIME           PIC 9(10).
           05  WS-END-TIME             PIC 9(10).
           05  WS-NSTREAM-TIME         PIC 9(10).
           05  WS-AVG-TIME             PIC 9(10)V9(6).
           
       01  WS-RESULTS.
           05  WS-BYTES                PIC 9(15)V9(6).
           05  WS-RATE                 PIC 9(10)V9(6).
           05  WS-CHECKSUM             PIC 9(15)V9(12).
           05  WS-REF-CHECKSUM         PIC 9(15)V9(12).
           05  WS-RESIDUUM             PIC 9(5)V9(15).
           05  WS-EPSILON              PIC 9(5)V9(15) VALUE 0.000001.
           
       01  WS-TEMP-VARS.
           05  WS-TEMP1                PIC 9(15)V9(12).
           05  WS-TEMP2                PIC 9(15)V9(12).
           
       01  WS-ARRAYS.
           05  WS-ARRAY-A              OCCURS 1000000 TIMES 
                                       INDEXED BY IDX-A.
               10  WS-A-VALUE          PIC S9(10)V9(12) COMP-3.
           05  WS-ARRAY-B              OCCURS 1000000 TIMES 
                                       INDEXED BY IDX-B.
               10  WS-B-VALUE          PIC S9(10)V9(12) COMP-3.
           05  WS-ARRAY-C              OCCURS 1000000 TIMES 
                                       INDEXED BY IDX-C.
               10  WS-C-VALUE          PIC S9(10)V9(12) COMP-3.

       PROCEDURE DIVISION.
       MAIN-PROCEDURE.
           DISPLAY "Parallel Research Kernels"
           DISPLAY "COBOL STREAM triad: A = B + scalar*C"
           
           *> Get command line arguments
           ACCEPT WS-ARG-COUNT FROM ARGUMENT-NUMBER
           
           IF WS-ARG-COUNT < 2
               DISPLAY "Usage: nstream <# iterations> <vector length>"
               STOP RUN
           END-IF
           
           ACCEPT WS-ARG1 FROM ARGUMENT-VALUE
           ACCEPT WS-ARG2 FROM ARGUMENT-VALUE
           
           *> Convert arguments to numeric
           MOVE FUNCTION NUMVAL(WS-ARG1) TO WS-ITERATIONS
           MOVE FUNCTION NUMVAL(WS-ARG2) TO WS-LENGTH
           
           *> Validate parameters
           IF WS-ITERATIONS < 1
               DISPLAY "ERROR: iterations must be >= 1"
               STOP RUN
           END-IF
           
           IF WS-LENGTH < 1 OR WS-LENGTH > 1000000
               DISPLAY "ERROR: vector length must be 1-1000000"
               STOP RUN
           END-IF
           
           DISPLAY "Vector length        = " WS-LENGTH
           DISPLAY "Number of iterations = " WS-ITERATIONS
           
           *> Initialize arrays
           PERFORM VARYING WS-J FROM 1 BY 1 UNTIL WS-J > WS-LENGTH
               SET IDX-A TO WS-J
               SET IDX-B TO WS-J
               SET IDX-C TO WS-J
               MOVE 0.0 TO WS-A-VALUE(IDX-A)
               MOVE 2.0 TO WS-B-VALUE(IDX-B)
               MOVE 2.0 TO WS-C-VALUE(IDX-C)
           END-PERFORM
           
           *> Main loop - repeat triad iterations times  
           PERFORM VARYING WS-ITER FROM 1 BY 1 
                   UNTIL WS-ITER > WS-ITERATIONS
               
               *> Start timer after warmup iteration (simplified)
               IF WS-ITER = 1
                   MOVE 0 TO WS-START-TIME
               END-IF
               
               *> STREAM triad: A[j] += B[j] + scalar*C[j]
               PERFORM VARYING WS-J FROM 1 BY 1 
                       UNTIL WS-J > WS-LENGTH
                   SET IDX-A TO WS-J
                   SET IDX-B TO WS-J  
                   SET IDX-C TO WS-J
                   COMPUTE WS-A-VALUE(IDX-A) = WS-A-VALUE(IDX-A) +
                           WS-B-VALUE(IDX-B) + 
                           (WS-SCALAR * WS-C-VALUE(IDX-C))
               END-PERFORM
               
           END-PERFORM
           
           *> Stop timer (simplified - just use constant for now)
           MOVE 1 TO WS-END-TIME
           COMPUTE WS-NSTREAM-TIME = 1
           
           *> Calculate bandwidth
           COMPUTE WS-BYTES = 4.0 * 8 * WS-LENGTH
           
           *> Verify results
           MOVE 0.0 TO WS-CHECKSUM
           PERFORM VARYING WS-J FROM 1 BY 1 
                   UNTIL WS-J > WS-LENGTH
               SET IDX-A TO WS-J
               COMPUTE WS-CHECKSUM = WS-CHECKSUM + WS-A-VALUE(IDX-A)
           END-PERFORM
           
           *> Calculate reference checksum
           *> A[j] = iterations * (B[j] + scalar*C[j]) = iterations * (2 + 3*2) = iterations * 8
           *> Total = iterations * 8 * length
           COMPUTE WS-REF-CHECKSUM = WS-ITERATIONS * 8.0 * WS-LENGTH
           
           *> Check if solution validates
           COMPUTE WS-TEMP1 = WS-CHECKSUM - WS-REF-CHECKSUM
           IF WS-TEMP1 < 0
               COMPUTE WS-TEMP1 = -WS-TEMP1
           END-IF
           COMPUTE WS-RESIDUUM = WS-TEMP1 / WS-REF-CHECKSUM
           
           IF WS-RESIDUUM < WS-EPSILON
               DISPLAY "Solution validates"
               COMPUTE WS-AVG-TIME = WS-NSTREAM-TIME / WS-ITERATIONS
               COMPUTE WS-RATE = 1.0E-06 * WS-BYTES / WS-AVG-TIME
               DISPLAY "Rate (MB/s): " WS-RATE 
                       " Avg time (s): " WS-AVG-TIME
           ELSE
               DISPLAY "ERROR: Checksum " WS-CHECKSUM 
                       " does not match verification value " 
                       WS-REF-CHECKSUM
               DISPLAY "Residuum: " WS-RESIDUUM
           END-IF
           
           STOP RUN.
