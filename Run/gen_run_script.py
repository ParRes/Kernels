#!/usr/bin/env python

import os, sys, argparse, time, datetime, math
from time import strftime

#repeats = 5
#queue_name="regular" # debug, premium

PRK_dir = os.path.realpath("esg-prk-devel")
results_dir = os.path.realpath("results_IPDPS")
PRK_dir = PRK_dir  + '/'
results_dir = results_dir  + '/'

nspn = 2 # number of sockets per node
nfg_list = [1, 4, 16]

#cores = [24, 48, 96, 192, 384, 768]

#PGAS15 tests
#dict_kernels = {'Transpose' : ['MPI', 'BUPC', 'CRAYUPC'],
                 #'Stencil'   : ['MPI', 'BUPC', 'CRAYUPC', 'MPIRMA', 'GRAPPA'],
                 #'p2p'       : ['MPI', 'MPIRMA', 'GRAPPA', 'SHMEM', 'SHMEM_HS']
                #}
#IPDPS16 tests
dict_kernels = {'Transpose' : ['MPI1','MPIRMA','MPIOPENMP','MPISHM',
                                'BUPC','CRAYUPC','SHMEM','CHARM++'],
                 'Stencil'   : ['MPI1','MPIRMA','MPIOPENMP','MPISHM',
                                'BUPC','CRAYUPC','SHMEM','GRAPPA','CHARM++'],
                 'p2p'       : ['MPI1','MPIRMA','MPIOPENMP','MPISHM',
                                'BUPC','CRAYUPC','SHMEM','GRAPPA','CHARM++'],
                }
#dict_kernels = {'Transpose' : ['MPI1','MPIRMA','MPIOPENMP','MPISHM',
                                #'GRAPPA','BUPC','CRAYUPC','SHMEM','FG_MPI','CHARM++'],
                 #'Stencil'   : ['MPI1','MPIRMA','MPIOPENMP','MPISHM',
                                #'GRAPPA','BUPC','CRAYUPC','SHMEM','FG_MPI','CHARM++'],
                 #'p2p'       : ['MPI1','MPIRMA','MPIOPENMP','MPISHM',
                                #'GRAPPA','BUPC','CRAYUPC','SHMEM','FG_MPI','CHARM++'],
                #}

dict_runtimes = {'MPI1'  : {'run':'aprun', 'env_vars':'', 'extra_flags':'', 'cpulist':''},
                'MPIRMA': {'run':'aprun', 'env_vars':'', 'extra_flags':'', 'cpulist':''},
                'MPIOPENMP': {'run':'aprun', 'env_vars':'', 'extra_flags':'', 'cpulist':''},
                'MPISHM': {'run':'aprun', 'env_vars':'', 'extra_flags':'', 'cpulist':''},
                'GRAPPA': {'run':'aprun', 'env_vars':'', 'extra_flags':'', 'cpulist':''},
                'BUPC': {'run':'upcrun', 'env_vars':'', 'extra_flags':'', 'cpulist':''},
                'CRAYUPC': {'run':'aprun', 'env_vars':'', 'extra_flags':'', 'cpulist':''},
                'SHMEM': {'run':'aprun', 'env_vars':'', 'extra_flags':'', 'cpulist':''},
                'FG_MPI': {'run':'ccmrun /global/homes/a/apokayi/opt/fgmpi/bin/mpirun', 'env_vars':'', 'extra_flags':'', 'cpulist':''},
                'CHARM++': {'run':'aprun', 'env_vars':'', 'extra_flags':'', 'cpulist':''},
                'LEGION': {'run':'aprun', 'env_vars':'', 'extra_flags':'', 'cpulist':''},
                }

def get_dir_list( dir ):
    if not os.path.exists(dir):
        return []
    lst = os.listdir( dir )
    lst = [elem for elem in lst if elem != ".svn" and elem != "c_version"]
    return lst

def create_unique_folder(basename):
    count = 1
    while True:
        name = get_unique_dir_name( basename )
        count = count + 1
        if count > 25:
            raise NameError('Unable to create a directory ' + name)
        no_error = True
        try:
            os.makedirs( name )
        except os.error as e:
            no_error = False
            if e.errno != errno.EEXIST:
                raise
        if no_error:
            break
    return name

def file_append(filename, data):
    with open(filename, 'a') as f:
        f.write(data)

def get_unique_dir_name(basename):
    i = 1
    if not os.path.isdir( basename ):
        return basename

    while  os.path.isdir( basename + "_" + str(i) ):
        i = i + 1

    return basename + "_" + str(i)

def get_unique_file_name(basename):
    i = 1
    if not os.path.isfile( basename ):
        return basename

    while  os.path.isfile( basename + "_" + str(i) ):
        i = i + 1

    return basename + "_" + str(i)

def run_cmd(cmd_line, no_run = False, print_cmd = True):
    fullcommand = "bash -c \"" +  cmd_line + "\""
    if print_cmd:
        print(fullcommand)
    if not no_run:
        return os.system( fullcommand )
    return None

def gen_base_script(dirName, fileName, r, k, threads, time_limit, alloc, max_threads,omp_affinity):
    fn = dirName + "/" + fileName
    with open(fn, 'w') as fscript:
      fscript.write("""#!/bin/bash

#PBS -j eo
#PBS -V
""")

      fscript.write("#PBS -l walltime=00:" + str(time_limit) + ":00\n")
      if alloc == 'normal':
          fscript.write("#PBS -l mppwidth=" + str(max(24,threads)) + "\n")
      elif alloc == 'MAX':
          fscript.write("#PBS -l mppwidth=" + str(max_threads) + "\n")
      else:
          print "Something is wrong here, check your alloc choice!"
      fscript.write("#PBS -N " + r + "_" + k + "_" + str(threads) + "\n")
      fscript.write("\nexport XT_SYMMETRIC_HEAP_SIZE=1G\n")
      fscript.write("\nexport UPC_SHARED_HEAP_SIZE=1G\n")
      #fscript.write("\nexport CRAY_PGAS_GROW_REGISTRATIONS=\"y\"\n")
      fscript.write("\nexport SHMEM_ROUTING_MODE=2\n")
      fscript.write("\nexport OMP_NUM_THREADS=12\n")
      fscript.write("\nexport KMP_AFFINITY=" + omp_affinity + "\n")
      fscript.write("\ncd $PBS_O_WORKDIR\n")
      fscript.close()

def gen_aprun_script(dirName, fileName, r, k, threads, repeats, psize, iterations, tile_size, cpn):
    fn = dirName + "/" + fileName
    with open(fn, 'a') as f:
      f.write("\n")

      if k == "Stencil":
        exe = PRK_dir + r + "/" + k + "/stencil"
        if(threads>96):
          xx = threads / 96
          iterations=iterations*xx
        print threads, "=>", iterations
      elif k == "Transpose":
        exe = PRK_dir + r + "/" + k + "/transpose"
        # run for more iterations to avoid noise at large core counts
        # assumes total time is a few seconds at 384
        # so iafter 384, keep doubling iterations as we double the thread count accordingly
        if(threads>384):
          xx = threads / 384
          iterations=iterations*xx
        print threads, "=>", iterations
      elif k == "p2p":
        exe = PRK_dir + r + "/Synch_p2p/p2p"

      if r == "BUPC":
        if k == "Stencil":
          args = " "  + str(iterations) + " " + str(psize)
        elif k == "Transpose":
          args = " " + str(iterations) + " " + str(psize) + " " + str(tile_size)
        elif k == "p2p":
          args = " " + str(iterations) + " " + str(psize) + " " + str(psize)
        run_args = " -n " + str(threads) + " -q "
        linestring = dict_runtimes[r]['run'] + run_args + exe + args + "\n"
        for x in range(repeats):
          f.write(linestring)
      elif r == "MPIOPENMP":
        ppn = 2 # process per node
        pps = 1 # process per socket
        nodes = threads/cpn
        nn = ppn * nodes
        dd = cpn / ppn / pps
        if k == "Stencil":
          args = " " + str(cpn/ppn) + " " + str(iterations) + " " + str(psize)
        elif k == "Transpose":
          args = " " + str(cpn/ppn) + " " + str(iterations) + " " + str(psize) + " " + str(tile_size)
        elif k == "p2p":
          f.write("export MPICH_MAX_THREAD_SAFETY=multiple\n")
          args = " " + str(cpn/ppn) + " " + str(iterations) + " " + str(psize) + " " + str(psize)

        run_args = " -n " + str(nn) + " -N " + str(ppn) + " -S " + str(pps) + " -d " + str(dd) + " -cc numa_node "
        linestring = dict_runtimes[r]['run'] + run_args + exe + args + "\n"
        for x in range(repeats):
          f.write(linestring)
      elif r == "MPISHM":
        ppn = cpn
        pps = cpn/nspn
        for th in 12,24:
          if k == "Stencil":
            args = " " + str(th) + " " + str(iterations) + " " + str(psize)
          elif k == "Transpose":
            args = " " + str(th) + " " + str(iterations) + " " + str(psize) + " " + str(tile_size)
          elif k == "p2p":
            if th == 12:
              continue
            args = " " + str(iterations) + " " + str(psize) + " " + str(psize)

          run_args = " -n " + str(threads) + " -N " + str(ppn) + " -S " + str(pps) + " -cc cpu "
          linestring = dict_runtimes[r]['run'] + run_args + exe + args + "\n"
          for x in range(repeats):
            f.write(linestring)
      elif r == "CHARM++":
        f.write("export HUGETLB_MORECORE=no\n")
        ppn = 1
        nodes = threads/cpn
        nn = ppn * nodes
        charm_psize = psize / 24 * 23
        extra_args = " +ppn 23 +pemap 1-23 +commap 0"
        run_args = " -n " + str(nn) + " -N " + str(ppn) + " "
        for ovd in nfg_list:
          for x in range(repeats):
            if k == "Stencil":
              args = " " + str(iterations) + " " + str(charm_psize) + " " + str(ovd)
            elif k == "Transpose":
              args = " " + str(iterations) + " " + str(charm_psize) + " " + str(tile_size) + " " + str(ovd)
            elif k == "p2p":
              args = " " + str(iterations) + " " + str(charm_psize) + " " + str(charm_psize) + " " + str(ovd)

            linestring = dict_runtimes[r]['run'] + run_args + exe + args + extra_args + "\n"
            f.write(linestring)
        f.write("export HUGETLB_MORECORE=yes\n")
      elif r == "FG_MPI":
        f.write("\nmodule load ccm\n")
        ppn = cpn
        if k == "Stencil":
          args = " " + str(iterations) + " " + str(psize)
        elif k == "Transpose":
          args = " " + str(iterations) + " " + str(psize) + " " + str(tile_size)
        elif k == "p2p":
          args = " " + str(iterations) + " " + str(psize) + " " + str(psize)

        for nfg in nfg_list:
          for x in range(repeats):
            run_args = " -np " + str(threads) + " -ppn " + str(ppn) + " -hostfile $PBS_NODEFILE " + " -nfg " + str(nfg) + " "
            linestring = dict_runtimes[r]['run'] + run_args + exe + args + "\n"
            f.write(linestring)
      elif r == "GRAPPA":
        ppn = cpn
        pps = cpn/nspn
        run_args = " -n " + str(threads) + " -N " + str(ppn) + " -S " + str(pps) + " -cc cpu "
        if k == "Stencil":
          args = " " + str(iterations) + " " + str(psize) + " --locale_shared_fraction=0.9  "
        elif k == "Transpose":
          args = " " + str(iterations) + " " + str(psize) + " " + str(tile_size) + " --locale_shared_fraction=0.9  "
        elif k == "p2p":
          args = " " + str(iterations) + " " + str(psize) + " " + str(psize)

        linestring = dict_runtimes[r]['run'] + run_args + exe + args + "\n"
        for x in range(repeats):
          f.write(linestring)
      else:
        ppn = cpn
        pps = cpn/nspn
        run_args = " -n " + str(threads) + " -N " + str(ppn) + " -S " + str(pps) + " -cc cpu "
        if k == "Stencil":
          args = " " + str(iterations) + " " + str(psize)
        elif k == "Transpose":
          args = " " + str(iterations) + " " + str(psize) + " " + str(tile_size)
        elif k == "p2p":
          args = " " + str(iterations) + " " + str(psize) + " " + str(psize)

        linestring = dict_runtimes[r]['run'] + run_args + exe + args + "\n"
        for x in range(repeats):
          f.write(linestring)

def gen_shell_script(dirName, fileName):
    script = open(dirName + "/" + fileName, "w")
    script.write("#!/bin/bash\n\n")

def clean(dir):
    run_cmd("rm -rf " + results_dir + dir)

def cleanall():
    for dir in runtimes:
        run_cmd("rm -rf " + results_dir + dir)

def main():
    parser = argparse.ArgumentParser(description='Generate BATCH scripts for PRK runs...')
    parser.add_argument("batch_mode", choices=['IND', 'ALL'], help="batch for IND runtimes or ALL runtimes for kernel/thread")
    parser.add_argument("problem_size", type=int, help="problem size to bo solved")
    parser.add_argument('-c', action='append', dest='cores', type=int,
                    #default=[24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288, 24576, 49152, 98304],
                    #default=[24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288, 24576],
                    default=[24, 48, 96, 192, 384, 768, 1536, 3072, 6144, 12288],
                    #default=[24, 48, 96, 192, 384, 768, 1536, 3072],
                    help='Add repeated values to cores list',
                    )
    parser.add_argument('-r', action='append', dest='runtimes',
                    default=[], choices=['MPI1', 'MPIRMA', 'MPISHM', 'MPIOPENMP',
                                         'UPC', 'CRAYUPC', 'BUPC', 'SHMEM', 'GRAPPA','CHARM++',
                                         'FG_MPI', 'LEGION'],
                    help='Add repeated values to runtimes list in IND mode useless in ALL mode',
                    )
    parser.add_argument('-k', action='append', dest='kernels',
                    default=[], choices=['Stencil', 'p2p', 'Transpose'],
                    help='Add repeated values to kernels list',
                    )
    parser.add_argument('--alloc', action='store', dest='alloc',
                    default='normal', choices=['normal', 'MAX'],
                    help='Specify allocation scheme MAX[allocate for max(THREADS)], default=normal[allocate cores for a given THREADS]',
                    )
    parser.add_argument('-i', dest='repeats', nargs='?',
                    default=5, type=int,
                    help='# of times to run each point -- default=5',
                    )
    parser.add_argument('--tile', dest='tile_size', nargs='?',
                    default=32, type=int,
                    help='tile size for TRANSPOSE -- default=32',
                    )
    parser.add_argument('--cpn', dest='cpn', nargs='?',
                    default=24, type=int,
                    help='number of cores per node -- default=24',
                    )
    parser.add_argument('-q', dest='queue_name', nargs='?',
                    default='regular', choices=['debug', 'regular', 'premium'],
                    help='batch queue name to run codes -- default=regular',
                    )
    parser.add_argument('-a', dest='omp_affinity', nargs='?',
                    default='compact', choices=['compact', 'scatter'],
                    help='KMP_AFFINITY for INTEL OPENMP compiler -- default=compact',
                    )
    parser.add_argument('-t', dest='time_limit', nargs='?',
                    default=10,
                    help='Time limit for batch job -- default=10',
                    )

    args = parser.parse_args()
    batch_mode = args.batch_mode
    print batch_mode
    cores = args.cores
    print cores
    max_threads = max(cores)
    runtimes = args.runtimes
    print 'MY_RUNTIMES :', runtimes
    kernels = args.kernels
    print 'MY_KERNELS :', kernels
    repeats = args.repeats
    print 'repeats :', repeats
    queue_name = args.queue_name
    print 'queue_name :', queue_name
    time_limit = args.time_limit
    print 'time : ', time_limit
    tile_size = args.tile_size
    print 'tile_size : ', tile_size
    alloc = args.alloc
    print 'alloc : ', alloc
    psize = args.problem_size
    print 'problem size : ', psize
    cpn = args.cpn
    print 'Cores per node : ', cpn
    omp_affinity = args.omp_affinity
    print 'omp_affinity: ', omp_affinity

    iterations = 10;

    if batch_mode == 'IND':
      if len(runtimes) == 0:
        print "available runtimes = ['MPI', 'MPIRMA', 'UPC', 'SHMEM', 'GRAPPA']"
        print "you need to specify some runtimes with -r to work on, EXITING!"
        sys.exit(1)
      elif len(kernels) == 0:
        print "available kernels = ['Stencil', 'Transpose', 'p2p']"
        print "you need to specify some kernels with -k to work on, EXITING!"
        sys.exit(1)

      for r in runtimes:
          for k in kernels:
            dirname = results_dir + r + "/" + k
            print dirname
            new_dir = create_unique_folder(dirname)
            shellFileName = "submit_" + r + "_" + k + ".sh"
            gen_shell_script(new_dir, shellFileName)

            for threads in cores:
              fname = "run_" + r + "_" + k + "_" + str(threads) + ".pbs"
              gen_base_script(new_dir, fname, r , k, threads, time_limit, alloc, max_threads, omp_affinity)
              #gen_script(new_dir, fname, r, k, threads, repeats, time_limit, alloc, max_threads, iterations, psize, tile_size, cpn)
              gen_aprun_script(new_dir, fname, r, k, threads, repeats, psize, iterations, tile_size, cpn)

              if r == "FG_MPI":
                appendLine = "qsub -q ccm_queue " + fname + "\n"
              else:
                appendLine = "qsub -q " + queue_name + " " + fname + "\n"
              shellFullName = new_dir + "/" + shellFileName
              file_append(shellFullName, appendLine)

            run_cmd("chmod +x " + shellFullName)
    elif batch_mode == 'ALL':
      print "Generating batch script for all runtimes"
      if len(kernels) == 0:
        print "available kernels = ['Stencil', 'Transpose', 'p2p']"
        print "you need to specify some kernels with -k to work on, EXITING!"
        sys.exit(1)

      for k in kernels:
        dirname = results_dir + "ALL/" + k
        print dirname

        new_dir = create_unique_folder(dirname)
        shellFileName = "submit_" + k + ".sh"
        gen_shell_script(new_dir, shellFileName)

        runtime_list = dict_kernels[k]
        print runtime_list

        for threads in cores:
          fname = "run_" + k + "_" + str(threads) + ".pbs"
          gen_base_script(new_dir, fname, k, threads, time_limit, alloc, max_threads, omp_affinity)

          for r in runtime_list:
            gen_aprun_script(new_dir, fname, r, k, threads, repeats, psize, iterations, tile_size, cpn)

          appendLine = "qsub -q " + queue_name + " " + fname + "\n"
          shellFullName = new_dir + "/" + shellFileName
          file_append(shellFullName, appendLine)

        run_cmd("chmod +x " + shellFullName)

    #cleanall()

if __name__ == "__main__":
    main()
