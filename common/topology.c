/*
Copyright (c) 2015, Intel Corporation

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

/****************************************************************

Name:      print_topology

Purpose:   Write topology-related information to a file.

Arguments: File to which the information is to be written
           Label on output, typically the rank/pe/thread.

Returns:   None.

Notes:     Currently, physics topology information is only available
           for Cray XC systems.  It's not easy to get this info
           on InfiniBand clusters (requires admin rights) and we
           have no use for Blue Gene support (which I have elsewhere).

           The implementation is runtime-agnostic on Cray XC and
           should work with MPI, UPC, SHMEM, etc.
           Otherwise, MPI is required.

History:   Written by Jeff Hammond, August 2015.

****************************************************************/

#include <stdio.h>
#include <stdlib.h>
#if defined(__CRAYXC)
#elif defined(MPI)
  #include "mpi.h"
#else
  #include <unistd.h>
#endif


void print_topology(FILE * output, int label)
{
#if defined(__CRAYXC)
    {
        /* see https://github.com/jeffhammond/HPCInfo/wiki/Cray */

        FILE * procfile = fopen("/proc/cray_xt/cname","r");
        if (procfile!=NULL)
        {
            /* format example: c1-0c1s2n1 c3-0c2s15n3 */
            char a, b, c, d;
            int i, j, k, l, m;
            fscanf(procfile, "%c%d-%d%c%d%c%d%c%d", &a, &i, &j, &b, &k, &c, &l, &d, &m);
            fprintf(output,"%d: Cray XC coords = (%d,%d,%d,%d,%d) \n", label, i, j, k, l, m);

            fclose(procfile);
        } else {
            fprintf(output, "%d: could not open /proc/cray_xt/cname\n", label);
        }
    }
#elif defined(MPI_VERSION)
    {
        /* see http://www.mpich.org/static/docs/v3.1/www3/MPI_Get_processor_name.html */
        int len;
        char procname[MPI_MAX_PROCESSOR_NAME];
        MPI_Get_processor_name(&procname,&len);
        fprintf(output,"%d: MPI proc name =  %s\n", label, procname);
    }
#else
    {
        /* see http://linux.die.net/man/2/gethostname */
        char procname[HOST_NAME_MAX];
        gethostname(&procname,HOST_NAME_MAX);
        fprintf(output,"%d: POSIX host name =  %s\n", label, procname);
    }
#endif
    return;
}
