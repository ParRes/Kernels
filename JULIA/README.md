


# Install MPI.jl

```sh
git clone https://github.com/JuliaParallel/MPI.jl
```

Read https://juliaparallel.github.io/MPI.jl/stable/configuration/
or run the following and hope for the best:

```sh
julia --project -e 'ENV["JULIA_MPI_BINARY"]="system"; using Pkg; Pkg.build("MPI"; verbose=true)'
```

# Run stuff

This is trivial, because this program doesn't use MPI.jl:
```sh
MPI.jl/bin/mpiexecjl --project=MPI.jl -n 1 julia nstream.jl 10 1000000
```
