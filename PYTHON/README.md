# How to run

## mpi4py

```
 mpiexec -n 4 python3 -m mpi4py nstream-numpy-mpi.py 10 10000000
 mpiexec -n 4 python3 -m mpi4py transpose-numpy-mpi.py 10 1000
```

On Mac with Homebrew, this might work better:

```
 mpiexec -n 4 ./nstream-numpy-mpi.py 10 10000000
 mpiexec -n 4 ./transpose-numpy-mpi.py 10 1000
```

## shmem4py

Checkout shmem4py and build against e.g. SOS like this:
```
$ export OSHCC=oshcc
$ python3 -m pip install .
```

Run like this:
```
$ oshrun -n 4 python3 nstream-numpy-shmem.py 10 10000000
Parallel Research Kernels version
Python SHMEM/NumPy STREAM triad: A = B + scalar * C
Number of ranks      =  4
Number of iterations =  10
Vector length        =  10000000
Solution validates
Rate (MB/s):  22345.12038433607  Avg time (s):  0.0143208
```
