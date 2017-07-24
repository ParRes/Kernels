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

#ifndef PRK_MPI_H
#define PRK_MPI_H

#include <mpi.h>

#if defined(ADAPTIVE_MPI)
# ifndef MPI_INT64_T
#  define MPI_INT64_T MPI_LONG_LONG
# endif /* MPI_INT64_T */
# ifndef MPI_UINT64_T
#  define MPI_UINT64_T MPI_UNSIGNED_LONG_LONG
# endif /* MPI_UINT64_T */
#endif /* AMPI */

/* This code appears in MADNESS, which is GPL, but it was
 * written by Jeff Hammond and contributed to multiple projects
 * using an implicit public domain license. */
#define PRK_MPI_THREAD_STRING(level)  \
        ( level==MPI_THREAD_SERIALIZED ? "THREAD_SERIALIZED" : \
            ( level==MPI_THREAD_MULTIPLE ? "THREAD_MULTIPLE" : \
                ( level==MPI_THREAD_FUNNELED ? "THREAD_FUNNELED" : \
                    ( level==MPI_THREAD_SINGLE ? "THREAD_SINGLE" : "THREAD_UNKNOWN" ) ) ) )

/* We should set an attribute that indicates we need to free memory
 * when using this so that the MPI_Win_free does not create a
 * double-free situation when paired with a real MPI_Win_create call. */
int PRK_Win_allocate(MPI_Aint size, int disp_unit, MPI_Info info,
                     MPI_Comm comm, void * baseptr, MPI_Win * win)
{
#if MPI_VERSION < 3
    int rc = MPI_SUCCESS;
    /* Strip info keys because MPI-2 will not understand the ones
     * that were added in MPI-3, which are the only useful ones. */
    MPI_Info alloc_info = MPI_INFO_NULL;
    MPI_Info win_info = MPI_INFO_NULL;
    rc = MPI_Alloc_mem(size, alloc_info, &baseptr);
    if (rc!=MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD,rc);
    rc = MPI_Win_create(baseptr, size, disp_unit, win_info, comm, win);
    if (rc!=MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD,rc);
    return MPI_SUCCESS;
#else
    return MPI_Win_allocate(size, disp_unit, info, comm, baseptr, win);
#endif
}

int PRK_Win_free(MPI_Win * win)
{
    int rc = MPI_SUCCESS;
#if MPI_VERSION < 3
    int flag = 0;
    void * attr_ptr;
#ifndef ADAPTIVE_MPI
    rc = MPI_Win_get_attr(*win, MPI_WIN_BASE, (void*)&attr_ptr, &flag);
    if (rc!=MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD,rc);
#endif
    /* We do not check for the case of size=0 here,
     * but it may be worth adding in the future. */
    if (flag) {
        void * baseptr = (void*)attr_ptr;
        rc = MPI_Free_mem(baseptr);
        if (rc!=MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD,rc);
    } else {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD,&rank);
        printf("%d: could not capture baseptr from win attribute: memory leak.\n",rank);
    }
    rc = MPI_Win_free(win);
    if (rc!=MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD,rc);
    return MPI_SUCCESS;
#else
    int free_mem = 0;
    void * baseptr = NULL;
    /* See if window was created with MPI_Win_create... */
    int flag = 0;
    void * attr_ptr;
#ifndef ADAPTIVE_MPI
    rc = MPI_Win_get_attr(*win, MPI_WIN_CREATE_FLAVOR, (void*)&attr_ptr, &flag);
    if (rc!=MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD,rc);
#endif
    if (flag) {
        int * flavor = (int*)attr_ptr;
        /* ...if it was, determine the base address of the user buffer... */
        if (*flavor==MPI_WIN_FLAVOR_CREATE) {
            flag = 0;
#ifndef ADAPTIVE_MPI
            rc = MPI_Win_get_attr(*win, MPI_WIN_BASE, (void*)&attr_ptr, &flag);
            if (rc!=MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD,rc);
#endif
            /* ...and free the base address. */
            if (flag) {
                baseptr = attr_ptr;
                free_mem = 1;
                /* We cannot free memory until after the window has been freed:
                 * "The memory associated with windows created by a call to
                 *  MPI_WIN_CREATE may be freed after the call returns."
                 *          - MPI 3.1 11.2.5                                    */
            }
        }
    }
    rc = MPI_Win_free(win);
    if (rc!=MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD,rc);
    if (free_mem) {
        rc = MPI_Free_mem(baseptr);
        if (rc!=MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD,rc);
    }
    return MPI_SUCCESS;
#endif
}

extern void bail_out(int);

#endif /* PRK_MPI_H */
