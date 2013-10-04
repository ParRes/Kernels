/*
Copyright (c) 2013, Intel Corporation

Redistribution and use in source and binary forms, with or without 
modification, are permitted provided that the following conditions 
are met:

* Redistributions of source code must retain the above copyright 
      notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above 
      copyright notice, this list of conditions and the following 
      disclaimer in the documentation and/or other materials provided 
      with the distribution.
* Neither the name of Intel Corporation nor the names of its 
      contributors may be used to endorse or promote products 
      derived from this software without specific prior written 
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS 
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT 
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS 
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE 
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN 
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
POSSIBILITY OF SUCH DAMAGE.
*/

/*******************************************************************

NAME:    RefCount

PURPOSE: This program tests the efficiency of exclusive access to a
         pair of non-adjacent shared reference counters
  
USAGE:   The program takes as input the total number of times the reference 
         counters are updated, and the number of threads 
         involved.

               <progname>  <# threads><# iterations>
  
         The output consists of diagnostics to make sure the 
         algorithm worked, and of timing statistics.

FUNCTIONS CALLED:

         Other than OpenMP or standard C functions, the following 
         functions are used in this program:

         wtime()
         bail_out()
         getpagesize()

HISTORY: Written by Rob Van der Wijngaart, January 2006.
  
*******************************************************************/

#include <par-res-kern_general.h>
#include <par-res-kern_omp.h>

/* shouldn't need the prototype below, since it is defined in <unistd.h>. But it
   depends on the existence of symbols __USE_BSD or _USE_XOPEN_EXTENDED, neither
   of which may be present. To avoid warnings, we define the prototype here      */
#if !defined(__USE_BSD) && !defined(__USE_XOPEN_EXTENDED)
extern int getpagesize(void);
#endif

int main(int argc, char ** argv)
{
  int        iterations;      /* total number of reference pair counter updates */
  int        page_fit;        /* indicates that counters fit on different pages */
  size_t     store_size;      /* amount of space reserved for counters          */
  s64Int     *pcounter1,     
             *pcounter2;      /* pointers to counters                           */
  s64Int     *counter_space;  /* pointer to space reserved for counters         */
  omp_lock_t counter_lock;    /* lock that guards access to counters            */
  double     refcount_time;   /* timing parameter                               */
  int        nthread_input;   /* thread parameters                              */
  int        nthread; 
  s64Int     sum_distance=0, 
             sum_distance2=0; /* distance and distance squared between 
                                 reference counter updates by the same thread,
                                 summed over all threads                        */
  double     avg_distance,
             avg_distance2;   /* distances averaged over all threads            */
  int        num_error=0;     /* flag that signals that requested and obtained
                                 numbers of threads are the same                */

/*********************************************************************
** process and test input parameters    
*********************************************************************/

  if (argc != 3){
    printf("Usage: %s <# threads> <# counter pair updates>\n", *argv);
    return(1);
  }

  nthread_input = atoi(*++argv);
  if ((nthread_input < 1) || (nthread_input > MAX_THREADS)) {
    printf("ERROR: Invalid number of threads: %d\n", nthread_input);
    exit(EXIT_FAILURE);
  }

  iterations  = atoi(*++argv);
  if (iterations < 1){
    printf("ERROR: iterations must be >= 1 : %d \n",iterations);
    exit(EXIT_FAILURE);
  }

  omp_set_num_threads(nthread_input);
   
  /* initialize shared counters; we put them on different pages, if possible.
     If the page size equals the whole memory, this will fail, and we reduce
     the space required */
  page_fit = 1;
  store_size = (size_t) getpagesize();
#ifdef VERBOSE
  printf("Page size = %d\n", getpagesize());
#endif
  counter_space = (s64Int *) malloc(store_size+sizeof(s64Int));
  while (!counter_space && store_size>2*sizeof(s64Int)) {
    page_fit=0;
    store_size/=2;
    counter_space = (s64Int *) malloc(store_size+sizeof(s64Int));
  }
  if (!counter_space) {
    printf("ERROR: could not allocate space for counters\n");
    exit(EXIT_FAILURE);
  }

#ifdef VERBOSE
  if (!page_fit) printf("Counters do not fit on different pages\n");      
  else           printf("Counters fit on different pages\n");      
#endif
   
  pcounter1 = counter_space;
  pcounter2 = counter_space + store_size/sizeof(s64Int);
  (*pcounter1) = 0;
  (*pcounter2) = 0;

  /* initialize the lock on which we will be pounding */
  omp_init_lock(&counter_lock);

  #pragma omp parallel reduction(+:sum_distance,sum_distance2)
  {
  int iter;         /* dummy                                          */
  /* we declare everything the same type/length to avoid consversions */
  s64Int oldcounter;/* previous thread value of reference counter     */
  s64Int newcounter;/* current thread value of reference counter      */
  s64Int distance;  /* distance between successive counter updates by
                       same thread                                    */

  #pragma omp master
  {
  nthread = omp_get_num_threads();
  printf("OpenMP exclusive access test RefCount, shared counters\n");
  if (nthread != nthread_input) {
    num_error = 1;
    printf("ERROR: number of requested threads %d does not equal ",
           nthread_input);
    printf("number of spawned threads %d\n", nthread);
  } 
  else {
    printf("Number of threads              = %i\n",nthread_input);
    printf("Number of counter pair updates = %i\n", iterations);
  }
  }
  bail_out(num_error);

  #pragma omp master
  {
  refcount_time = wtime();
  }

  /* the first iteration that any thread does initializes oldcounter. 
     We could treat this situation with a test in the main loop, but 
     that adds overhead to each iteration, so we keep it separate     */
  omp_set_lock(&counter_lock);
  (*pcounter1)++;
#ifdef VERBOSE
  oldcounter=*pcounter1;
#endif
  (*pcounter2)++;
  omp_unset_lock(&counter_lock);

  #pragma omp for
  /* start with iteration nthread to take into account pre-loop iter  */
  for (iter=nthread; iter<iterations; iter++) { 
    omp_set_lock(&counter_lock);
    /* keep stuf within lock region as brief as possible              */
#ifdef VERBOSE   
    distance = ((*pcounter1)++)-oldcounter;
    oldcounter = (*pcounter1);
#else
    (*pcounter1)++;
#endif
    (*pcounter2)++;
    omp_unset_lock(&counter_lock);
#ifdef VERBOSE
    sum_distance  += distance;
    sum_distance2 += distance*distance;
#endif
  }

  #pragma omp master 
  { 
  refcount_time = wtime() - refcount_time;
  }
  } /* end of OpenMP parallel region */

#ifdef VERBOSE
  if (iterations > 1) {
    avg_distance  = (double) sum_distance/(iterations-1);
    avg_distance2 = (double) sum_distance2/(iterations-1);
  }
#endif
  if ((*pcounter1) != iterations || (*pcounter1) != (*pcounter2)) {
     printf("ERROR: Incorrect or inconsistent counter values "FSTR64U,
            (*pcounter1));
     printf(FSTR64U"; should be %d\n", (*pcounter2), iterations);
  }
  else {
#ifdef VERBOSE
    printf("Solution validates; Correct counter value of "FSTR64"\n", (*pcounter1));
#else
    printf("Solution validates\n");
#endif
    printf("Rate (CPUPs/s): %d, time (s): %lf\n", 
           (int)(iterations/refcount_time), refcount_time);
#ifdef VERBOSE
    if (iterations > 1) {
      printf("Average update distance: %lf\n", avg_distance);
      printf("Standard deviation of update distance: %lf\n", 
              sqrt(avg_distance2-avg_distance*avg_distance));
      printf("Mean and standard deviation of update distance for random locks: ");
      printf("%lf, %lf\n", (double)(nthread-1), sqrt((double)(nthread*(nthread-1))));
      printf("                                                   fair locks:   ");
      printf("%lf, %lf\n", (double)(nthread-1), 0.0);
    }
#endif
  }

  exit(EXIT_SUCCESS);
}
