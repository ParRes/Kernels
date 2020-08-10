#ifndef PRK_PETSC_H_
#define PRK_PETSC_H_

#define PRK_PETSC_USE_MPI 1

#ifdef PRK_PETSC_USE_MPI
#include <mpi.h>
#endif

#include <petsc.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscmat.h>

#endif // PRK_PETSC_H_
