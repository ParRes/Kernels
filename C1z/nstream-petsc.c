///
/// Copyright (c) 2019, Intel Corporation
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
/// NAME:    nstream
///
/// PURPOSE: To compute memory bandwidth when adding a vector of a given
///          number of double precision values to the scalar multiple of
///          another vector of the same length, and storing the result in
///          a third vector.
///
/// USAGE:   The program takes as input the number
///          of iterations to loop over the triad vectors and
///          the length of the vectors.
///
///          <progname> <# iterations> <vector length>
///
///          The output consists of diagnostics to make sure the
///          algorithm worked, and of timing statistics.
///
/// NOTES:   Bandwidth is determined as the number of words read, plus the
///          number of words written, times the size of the words, divided
///          by the execution time. For a vector length of N, the total
///          number of words read and written is 4*N*sizeof(double).
///
///
/// HISTORY: This code is loosely based on the Stream benchmark by John
///          McCalpin, but does not follow all the Stream rules. Hence,
///          reported results should not be associated with Stream in
///          external publications
///
///          Converted to C++11 by Jeff Hammond, November 2017.
///          Converted to C11 by Jeff Hammond, February 2019.
///
//////////////////////////////////////////////////////////////////////

#include "prk_util.h"
#include "prk_petsc.h"

static char help[] = "FIXME.\n\n";

int main(int argc, char * argv[])
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);
  if (ierr) {
      PetscPrintf(PETSC_COMM_WORLD,"PetscInitialize failed\n");
      abort();
      return -1;
  }

  PetscPrintf(PETSC_COMM_WORLD,"Parallel Research Kernels version %d\n", PRKVERSION );
  PetscPrintf(PETSC_COMM_WORLD,"C11/PETSc STREAM triad: A = B + scalar * C\n");

  if (argc == 1) {
    PetscPrintf(PETSC_COMM_WORLD,"Example arguments:\n");
    PetscPrintf(PETSC_COMM_WORLD,"  ./nstream-petsc -i <iterations> -n <vector length> \\\n");
    PetscPrintf(PETSC_COMM_WORLD,"                 [-vec_type {standard,seq,mpi,shared,node}] \\\n");
    PetscPrintf(PETSC_COMM_WORLD,"                 [-log_view]\n");
  }

  //////////////////////////////////////////////////////////////////////
  /// Read and test input parameters
  //////////////////////////////////////////////////////////////////////

  PetscBool set = PETSC_FALSE;

  PetscInt iterations = -1;
  ierr = PetscOptionsGetInt(NULL,NULL,"-i",&iterations,&set); CHKERRQ(ierr);

  if (set != PETSC_TRUE || iterations < 1) {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: iterations must be >= 1\n");
    PetscPrintf(PETSC_COMM_WORLD,"HELP:  Set with -i <iterations>\n");
    PetscFinalize();
    return 1;
  }

  PetscInt length = -1;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&length,&set); CHKERRQ(ierr);
  if (set != PETSC_TRUE || length <= 0) {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: Vector length must be greater than 0\n");
    PetscPrintf(PETSC_COMM_WORLD,"HELP:  Set with -n <vector length>\n");
    PetscFinalize();
    return 1;
  }

  int np = 1;
#ifdef PRK_PETSC_USE_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &np);
#endif
  PetscPrintf(PETSC_COMM_WORLD,"Number of processes  = %d\n", np);
  PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %d\n", iterations);
  PetscPrintf(PETSC_COMM_WORLD,"Vector length        = %zu\n", length);

  //////////////////////////////////////////////////////////////////////
  // Allocate space and perform the computation
  //////////////////////////////////////////////////////////////////////

  double nstream_time = 0.0;

  PetscReal zero  = 0;
  PetscReal one   = 1;
  PetscReal two   = 2;
  PetscReal three = 3;

  Vec A;
  Vec B;
  Vec C;

  ierr = VecCreate(PETSC_COMM_WORLD, &A); CHKERRQ(ierr);
  ierr = VecSetFromOptions(A); CHKERRQ(ierr);
  ierr = VecSetSizes(A,PETSC_DECIDE,length); CHKERRQ(ierr);

  ierr = VecDuplicate(A,&B); CHKERRQ(ierr);
  ierr = VecDuplicate(A,&C); CHKERRQ(ierr);

  ierr = VecSet(A,zero); CHKERRQ(ierr);
  ierr = VecSet(B,two);  CHKERRQ(ierr);
  ierr = VecSet(C,two);  CHKERRQ(ierr);

  PetscLogEvent prk_event;
  PetscLogEventRegister("PRK nstream",0,&prk_event);

  for (int iter = 0; iter<=iterations; iter++) {

    if (iter==1) {
        ierr = PetscBarrier(NULL); CHKERRQ(ierr);
        nstream_time = prk_wtime();
        ierr = PetscLogEventBegin(prk_event,0,0,0,0); CHKERRQ(ierr);
    }

    // A += B + three * C
    // z = alpha x + beta y + gamma z
    // z:=A gamma:=1
    // x:=B alpha:=1
    // y:=C beta:=three
    ierr = VecAXPBYPCZ(A, one, three, one, B, C); CHKERRQ(ierr);
  }

  ierr = PetscBarrier(NULL); CHKERRQ(ierr);
  nstream_time = prk_wtime() - nstream_time;
  ierr = PetscLogEventEnd(prk_event,0,0,0,0); CHKERRQ(ierr);

  //////////////////////////////////////////////////////////////////////
  /// Analyze and output results
  //////////////////////////////////////////////////////////////////////

  PetscReal ar = 0;
  PetscReal br = 2;
  PetscReal cr = 2;
  for (int i=0; i<=iterations; i++) {
      ar += br + three * cr;
  }

  ar *= length;

  PetscReal asum = 0;
  ierr = VecNorm(A, NORM_1, &asum);

  double epsilon=1.e-8;
  if (fabs(ar-asum)/asum > epsilon) {
      PetscPrintf(PETSC_COMM_WORLD,"Failed Validation on output array\n"
             "       Expected checksum: %lf\n"
             "       Observed checksum: %lf\n"
             "ERROR: solution did not validate\n", ar, asum);
      ierr = VecView(A,PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  } else {
      PetscPrintf(PETSC_COMM_WORLD,"Solution validates\n");
      double avgtime = nstream_time/iterations;
      double nbytes = 4.0 * length * sizeof(double);
      PetscPrintf(PETSC_COMM_WORLD,"Rate (MB/s): %lf Avg time (s): %lf\n", 1.e-6*nbytes/avgtime, avgtime);
  }

  ierr = VecDestroy(&A); CHKERRQ(ierr);
  ierr = VecDestroy(&B); CHKERRQ(ierr);
  ierr = VecDestroy(&C); CHKERRQ(ierr);

  PetscFinalize();

  return 0;
}


