# How to run

```
 mpiexec -n 4 python3 -m mpi4py nstream-numpy-mpi.py 10 10000000
 mpiexec -n 4 python3 -m mpi4py transpose-numpy-mpi.py 10 1000
```

On Mac with Homebrew, this might work better:

```
 mpiexec -n 4 ./nstream-numpy-mpi.py 10 10000000
 mpiexec -n 4 ./transpose-numpy-mpi.py 10 1000
```
