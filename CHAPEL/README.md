## Disclaimer

This directory’s contents were created as a copy of 
https://github.com/chapel-lang/chapel/tree/08f3e9c8414d0aeedaa9ad8667edc556c0cb86e8/test/studies/prk 
and the ongoing maintained versions of them as Chapel evolves can be found in 
https://github.com/chapel-lang/chapel/tree/main/test/studies/prk.
Note that these versions have not necessarily been studied in great depth
w.r.t. performance nor considered in light of Chapel features that have been
added since they were originally written circa 2016
(e.g., foreach loops supporting vectorization; GPU support).
As a result, please take any performance obtained from them
with a grain of salt and consider contacting the Chapel team
if you have any questions, find any surprises, or are planning
to publish conclusions based on their results (https://chapel-lang.org/community/).

## Implementations:

Each kernel is implemented in a variety of parallel flavors, such as "serial",
"shared", and "dist"-ributed. There is also two versions for each flavor:
a "fast" version optimized for performance and a
"pretty" version optimized for elegance. Ideally, the performance delta
between these versions will approach zero over time.

The test inputs were taken from the inputs used in the PRK test scripts.

## Options

==> DGEMM/dgemm.execopts <==
--correctness --iterations=2 --order=64 --blockSize=8
--correctness --iterations=2 --order=64

==> Nstream/nstream.execopts <==
--correctness --iterations=2 --length=100

==> PIC/pic.execopts <==
--correctness --L=100 --n=1000 --particleMode=SINUSOIDAL
--correctness --L=100 --n=1000 --particleMode=GEOMETRIC
--correctness --L=100 --n=1000 --particleMode=LINEAR
--correctness --L=100 --n=1000 --particleMode=PATCH

==> Sparse/sparse.execopts <==
--correctness --iterations=2 --lsize=3

==> Stencil/stencil.compopts <==
--set correctness --set order=100 --set iterations=3
--set correctness --set order=100 --set iterations=3 --set tileSize=20
--set correctness --set order=100 --set iterations=3                   --set compact=true
--set correctness --set order=100 --set iterations=3 --set tileSize=20 --set compact=true
--set correctness --set order=100 --set iterations=3 --set useBlockDist=true
--set correctness --set order=100 --set iterations=3 --set useStencilDist=true

==> Stencil/stencil.ml-compopts <==
--set useBlockDist=true   --set order=128000 # stencil-blockdist
--set useStencilDist=true --set order=128000 # stencil-stencildist

==> Stencil/stencil.perfcompopts <==
                          --set order=32000 # stencil-defaultdist
--set useBlockDist=true   --set order=32000 # stencil-blockdist
--set useStencilDist=true --set order=32000 # stencil-stencildist

==> Stencil/stencil-serial.compopts <==
--set correctness --set order=100 --set iterations=3
--set correctness --set order=100 --set iterations=3 --set tileSize=20
--set correctness --set order=100 --set iterations=3                   --set compact=true
--set correctness --set order=100 --set iterations=3 --set tileSize=20 --set compact=true

==> Stencil/stencil-serial.perfcompopts <==
--set order=32000

==> Transpose/transpose.compopts <==
--set iterations=10 --set order=200 --set tileSize=64 --set correctness
--set iterations=10 --set order=200 --set correctness

==> Transpose/transpose.perfcompopts <==
--set iterations=10 --set order=2000 --set tileSize=64  # transpose-defaultdist

==> Transpose/transpose-serial.compopts <==
--set iterations=10 --set order=200 --set tileSize=64 --set correctness
--set iterations=10 --set order=200 --set correctness

==> Transpose/transpose-serial.perfcompopts <==
--set iterations=10 --set order=2000 --set tileSize=64

==> Synch_p2p/EXECOPTS <==
--iterations=10 --m=100 --n=10 --correctness

==> Synch_p2p/PERFEXECOPTS <==
--iterations=10 --m=1000 --n=100

==> dgemm-summa-details/dgemm.compopts <==
-s useBlockDist=true -s blasImpl=off -s lapackImpl=off
-s useBlockDist=false -s blasImpl=off -s lapackImpl=off

==> dgemm-summa-details/dgemm.execopts <==
--correctness --iterations=2 --order=64 --windowSize=8
--correctness --iterations=2 --order=64

==> dgemm-summa-details/dgemm.ml-compopts <==
-s useBlockDist=true -s blasImpl=off -s lapackImpl=off   # dgemmNative.ml-keys
-s useBlockDist=true -s blasImpl=blas                    # dgemmBLAS.ml-keys

==> dgemm-summa-details/dgemm.ml-execenv <==
CRAYBLAS_FORCE_HEAP_ALLOC=1

==> dgemm-summa-details/dgemm.ml-execopts <==
--order 4096 --iterations 10 --windowSize 4096

==> stencil-opt-details/stencil-opt.execopts <==
--validate --order 100 --iterations 3

==> stencil-opt-details/stencil-opt.ml-compopts <==
--set order=128000
