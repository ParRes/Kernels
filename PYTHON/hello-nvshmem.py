import numpy
from mpi4py import MPI
from cuda.core import Device
from cuda.core import system
import nvshmem.core as nvshmem

# Initialize MPI
comm = MPI.COMM_WORLD
me = comm.Get_rank()
np = comm.Get_size()

# Initialize NVSHMEM with MPI
dev = Device(me % system.get_num_devices())
dev.set_current()
nvshmem.init(device=dev, mpi_comm=comm, initializer_method="mpi")

#uid = nvshmem.get_unique_id(empty=(me != 0))
#comm.Bcast(uid._data.view(numpy.int8), root=0)
#dev = Device()
#dev.set_current()
#nvshmem.init(device=dev, uid=uid, rank=me, nranks=np, initializer_method="uid")

#dev = Device(me % system.get_num_devices())
#dev.set_current()
#nvshmem.init(device=dev, mpi_comm=comm, initializer_method="emulated_mpi")

stream = dev.create_stream()

# Get information about the current PE
my_pe = nvshmem.my_pe()
n_pes = nvshmem.n_pes()

# Allocate symmetric memory
# array() returns a CuPy NDArray object
x = nvshmem.array((1024,), dtype="float32")
y = nvshmem.array((1024,), dtype="float32")

#if my_pe == 0:
#    y[:] = 1.0

# Perform communication operations
# Put y from PE 0 into x on PE 1
if my_pe == 0:
    nvshmem.put(x, y, 1, stream=stream)

# Synchronize PEs
nvshmem.barrier(nvshmem.Teams.TEAM_WORLD,stream=stream)
stream.sync()

# Clean up
nvshmem.free_array(x)
nvshmem.free_array(y)
nvshmem.finalize()
print('OK')

