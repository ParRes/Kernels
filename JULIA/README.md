(Note: Requires Julia >= 1.11)

# Instantiate Julia environment

```
julia --project -e 'using Pkg; Pkg.instantiate()'
```

To get the `mpiexecjl` driver run:
```
julia --project -e 'using MPI; MPI.install_mpiexecjl()'
```

Afterwards, put '$HOME/.julia/bin' on `PATH`, e.g.
```sh
export PATH=$HOME/.julia/bin:$PATH
```

(If you don't want to modify `PATH` use `$HOME/.julia/bin/mpiexecjl` directly to run the MPI code.)

## Optional: System MPI

If you want to use a system MPI run the following:
```
julia --project -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
```

# Run stuff

No MPI:
```sh
julia nstream.jl 10 1000000
```

With MPI:
```sh
mpiexecjl -n 4 julia --project nstream-mpi.jl 10 1000000
```
