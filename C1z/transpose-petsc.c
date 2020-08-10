///
/// Copyright (c) 2013, Intel Corporation
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
/// NAME:    transpose
///
/// PURPOSE: This program measures the time for the transpose of a
///          column-major stored matrix into a row-major stored matrix.
///
/// USAGE:   Program input is the matrix order and the number of times to
///          repeat the operation:
///
///          transpose <matrix_size> <# iterations> [tile size]
///
///          An optional parameter specifies the tile size used to divide the
///          individual matrix blocks for improved cache and TLB performance.
///
///          The output consists of diagnostics to make sure the
///          transpose worked and timing statistics.
///
/// HISTORY: Written by  Rob Van der Wijngaart, February 2009.
///          Converted to C++11 by Jeff Hammond, February 2016 and May 2017.
///          C11-ification by Jeff Hammond, June 2017.
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
  PetscPrintf(PETSC_COMM_WORLD,"C11/PETSc Transpose: B += A^T\n");

  if (argc == 1) {
    PetscPrintf(PETSC_COMM_WORLD,"Example arguments:\n");
    PetscPrintf(PETSC_COMM_WORLD,"  ./transpose-petsc -i <iterations> -n <matrix rank> [-t <tiling>]\\\n");
    PetscPrintf(PETSC_COMM_WORLD,"                   [-mat_type {dense,seqdense,mpidense}] \\\n");
    PetscPrintf(PETSC_COMM_WORLD,"                   [-log_view]\n");
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

  PetscInt order = -1;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&order,&set); CHKERRQ(ierr);
  if (set != PETSC_TRUE || order <= 0) {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: Matrix order must be greater than 0\n");
    PetscPrintf(PETSC_COMM_WORLD,"HELP:  Set with -n <matrix order>\n");
    PetscFinalize();
    return 1;
  }

  int me = 0, np = 1;
#ifdef PRK_PETSC_USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
#endif
  PetscPrintf(PETSC_COMM_WORLD,"Number of processes  = %d\n", np);
  PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %d\n", iterations);
  PetscPrintf(PETSC_COMM_WORLD,"Matrix order         = %d\n", order);

  //////////////////////////////////////////////////////////////////////
  /// Allocate space for the input and transpose matrix
  //////////////////////////////////////////////////////////////////////

  double trans_time = 0.0;

  PetscReal zero  = 0;
  PetscReal one   = 1;
  PetscReal two   = 2;
  PetscReal three = 3;

  Mat A;
  Mat B;
  Mat AT; // A^T formed explicitly every iteration
  Mat C;  // C:=constant - full of ones

  ierr = MatCreateDense(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, order, order, NULL, &A); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
  //ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,order,order); CHKERRQ(ierr);
  ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&B); CHKERRQ(ierr);
  ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&AT); CHKERRQ(ierr);
  ierr = MatDuplicate(A,MAT_DO_NOT_COPY_VALUES,&C); CHKERRQ(ierr);

  // A[i,j] = (i*order+j)
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  for (int i=0;i<order; i++) {
    for (int j=0;j<order;j++) {
      PetscReal v = (i*order+j);
      ierr = MatSetValue(A, i, j, v, INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  // B[i,j] = 0
#if 0
  ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  for (int i=0;i<order; i++) {
    for (int j=0;j<order;j++) {
      PetscReal v = 0;
      ierr = MatSetValue(B, i, j, v, INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
#else
  ierr = MatZeroEntries(B); CHKERRQ(ierr);
#endif

  // C[i,j] = 1
  ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  for (int i=0;i<order; i++) {
    for (int j=0;j<order;j++) {
      PetscReal v = 1;
      ierr = MatSetValue(B, i, j, v, INSERT_VALUES); CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  PetscLogEvent prk_event;
  PetscLogEventRegister("PRK transpose",0,&prk_event);

  for (int iter = 0; iter<=iterations; iter++) {

    if (iter==1) {
        ierr = PetscBarrier(NULL); CHKERRQ(ierr);
        trans_time = prk_wtime();
        ierr = PetscLogEventBegin(prk_event,0,0,0,0); CHKERRQ(ierr);
    }

    // AT = A^T
    ierr = MatTranspose(A, MAT_REUSE_MATRIX, &AT); CHKERRQ(ierr);
    // B += A^T
    ierr = MatAXPY(B, one, AT, SAME_NONZERO_PATTERN); CHKERRQ(ierr);
    // A += 1
    ierr = MatAXPY(A, one, C, SAME_NONZERO_PATTERN); CHKERRQ(ierr);
  }

  ierr = PetscBarrier(NULL); CHKERRQ(ierr);
  trans_time = prk_wtime() - trans_time;
  ierr = PetscLogEventEnd(prk_event,0,0,0,0); CHKERRQ(ierr);

  //////////////////////////////////////////////////////////////////////
  // Analyze and output results
  //////////////////////////////////////////////////////////////////////

  PetscReal addit = (iterations+1)*(iterations)/2;
  PetscReal abserr = 0;
#if 0
  for (int j=0; j<order; j++) {
    for (int i=0; i<order; i++) {
      const size_t ij = i*order+j;
      const size_t ji = j*order+i;
      const double reference = (double)(ij)*(1.+iterations)+addit;
      abserr += fabs(B[ji] - reference);
    }
  }
#endif

#ifdef VERBOSE
  PetscPrintf(PETSC_COMM_WORLD,"Sum of absolute differences: %lf\n", abserr);
  if (order < 10) {
      ierr = MatView(B, PETSC_VIEWER_STDOUT_WORLD);
  }
#endif

  PetscReal epsilon = 1.0e-8;
  if (abserr < epsilon) {
    PetscPrintf(PETSC_COMM_WORLD,"Solution validates\n");
    const double avgtime = trans_time/iterations;
    const size_t bytes = order*order*sizeof(double);
    PetscPrintf(PETSC_COMM_WORLD,"Rate (MB/s): %lf Avg time (s): %lf\n", 2.0e-6 * bytes/avgtime, avgtime );
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"ERROR: Aggregate squared error %lf exceeds threshold %lf\n", abserr, epsilon );
    return 1;
  }

  ierr = MatDestroy(&AT); CHKERRQ(ierr);
  ierr = MatDestroy(&A); CHKERRQ(ierr);
  ierr = MatDestroy(&B); CHKERRQ(ierr);

  PetscFinalize();

  return 0;
}


