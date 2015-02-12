#include <par-res-kern_general.h>
#include <par-res-kern_mpi.h>

int main(int argc, char ** argv)
{
  int    my_ID;         /* Process ID (i.e. MPI rank)                            */
  int    Num_procs;     /* Number of processors                                  */
  int    group_size;
  MPI_Comm shm_comm_prep;/* Shared Memory prep Communicator                      */
  MPI_Comm shm_comm;    /* Shared Memory Communicator                            */
  int shm_procs;        /* # of processes in shared domain                       */
  int shm_ID;           /* MPI rank within coherence domain                      */
  int shm_prep_procs;   /* # of processes in shared domain                       */
  int shm_prep_ID;      /* MPI rank within coherence domain                      */

/*********************************************************************************
** Initialize the MPI environment
**********************************************************************************/
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_ID);
  MPI_Comm_size(MPI_COMM_WORLD, &Num_procs);

/*********************************************************************
** process, test and broadcast input parameter
*********************************************************************/

  group_size = atoi(*++argv);

  /* Setup for Shared memory regions */

  /* first divide WORLD into groups of size group_size */
  MPI_Comm_split(MPI_COMM_WORLD, my_ID/group_size, my_ID%group_size, &shm_comm_prep);
  MPI_Comm_rank(shm_comm_prep, &shm_prep_ID);
  MPI_Comm_size(shm_comm_prep, &shm_prep_procs);

  /* derive from that a SHM communicator */
  MPI_Comm_split_type(shm_comm_prep, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shm_comm);
  MPI_Comm_rank(shm_comm, &shm_ID);
  MPI_Comm_size(shm_comm, &shm_procs);
  printf("My ID=%d, prep_ID=%d, shm_ID=%d\n", my_ID, shm_prep_ID, shm_ID);
  MPI_Barrier(MPI_COMM_WORLD);
  /* do sanity check, making sure groups did not shrink in second comm split */
  if (shm_procs != group_size) MPI_Abort(MPI_COMM_WORLD, 666);

  MPI_Finalize();                                                                                                                                                                                                                
  exit(EXIT_SUCCESS);
}
