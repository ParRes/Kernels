# Ray Setup and Usage Guide for PRK

This guide explains how to install, set up, and run the Ray-based parallel implementation of the Parallel Research Kernels (PRK) nstream benchmark.

## What is Ray?

Ray is a distributed computing framework that makes it easy to scale Python applications. It provides simple APIs for building distributed applications and can run on a single machine or across multiple nodes in a cluster.

## Prerequisites

- Python 3.7 or later
- macOS, Linux, or Windows

## Installation

### Option 1: Using Virtual Environment (Recommended)

This approach isolates Ray and its dependencies from your system Python installation.

```bash
# Navigate to the PYTHON directory
cd /path/to/PRK/PYTHON

# Create a virtual environment
python3 -m venv ray_env

# Activate the virtual environment
source ray_env/bin/activate  # On macOS/Linux
# OR
ray_env\Scripts\activate     # On Windows

# Install required packages
pip install ray numpy

# Verify installation
python3 -c "import ray; print('Ray version:', ray.__version__)"
python3 -c "import numpy; print('Numpy version:', numpy.__version__)"
```

### Option 2: Using pip with --user flag

If you prefer not to use a virtual environment:

```bash
pip3 install --user ray numpy
```

### Option 3: Using conda

If you use conda for package management:

```bash
conda install -c conda-forge ray numpy
```

## Running the Ray Implementation

### Basic Usage

```bash
# If using virtual environment, activate it first
source ray_env/bin/activate

# Run with basic parameters
python3 nstream-ray.py <iterations> <vector_length>

# Example: 10 iterations with vector length of 1,000,000
python3 nstream-ray.py 10 1000000
```

### Example Output

```
Parallel Research Kernels version 
Python Ray/Numpy STREAM triad: A = B + scalar * C
Ray version =  2.48.0
Numpy version =  2.3.2
Number of workers    =  8
Number of iterations =  10
Vector length        =  1000000
Chunk sizes          =  [125000, 125000, 125000, 125000, 125000, 125000, 125000, 125000]
Solution validates
Rate (MB/s):  1180.9981900205635  Avg time (s):  0.027095723152160644
```

## Performance Comparison

Compare the Ray implementation with other versions:

```bash
# Activate virtual environment if using one
source ray_env/bin/activate

# Test Ray version
echo "=== Ray Implementation ==="
python3 nstream-ray.py 10 1000000

# Test original numpy version
echo "=== Original Numpy Implementation ==="
python3 nstream-numpy.py 10 1000000

# Test MPI version (if mpi4py is available)
echo "=== MPI Implementation ==="
mpirun -np 4 python3 nstream-numpy-mpi.py 10 1000000
```

## Understanding the Output

- **Number of workers**: Number of CPU cores Ray detected and will use for parallel execution
- **Chunk sizes**: How the vector is divided among workers
- **Rate (MB/s)**: Memory bandwidth measurement
- **Avg time (s)**: Average time per iteration (excludes validation overhead)

## Tuning Performance

### Vector Size Considerations

- **Small vectors** (< 100K elements): Ray overhead may exceed parallelism benefits
- **Large vectors** (> 1M elements): Ray parallelism shows better performance gains
- **Very large vectors** (> 10M elements): Best candidates for Ray acceleration

### Worker Configuration

By default, Ray uses all available CPU cores. You can limit this:

```python
# In your script, modify the ray.init() call:
ray.init(num_cpus=4)  # Use only 4 cores
```

## Troubleshooting

### Common Issues

1. **"ModuleNotFoundError: No module named 'ray'"**
   ```bash
   # Make sure Ray is installed
   pip install ray
   # Or activate your virtual environment
   source ray_env/bin/activate
   ```

2. **"ConnectionError: Could not find any running Ray instance"**
   - This usually means the script is trying to connect to an existing cluster
   - The current implementation uses `ray.init()` to start a local cluster automatically

3. **Poor performance on small vectors**
   - This is expected due to Ray's distribution overhead
   - Use larger vector sizes to see parallelism benefits
   - For small problems, the original numpy version will be faster

4. **Ray not shutting down cleanly**
   ```bash
   # Force cleanup if needed
   ray stop
   ```

### Performance Tips

1. **Use appropriate vector sizes**: Ray performs best with larger datasets
2. **Monitor memory usage**: Each worker needs to hold its chunk in memory
3. **Consider cluster deployment**: For very large problems, deploy Ray on multiple machines

## Ray Cluster Setup (Advanced)

For distributed execution across multiple machines:

```bash
# On head node
ray start --head --port=6379

# On worker nodes  
ray start --address='<head-node-ip>:6379'

# In your script, connect to cluster
ray.init(address='ray://<head-node-ip>:10001')
```

## Cleanup

### Virtual Environment

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment (if desired)
rm -rf ray_env
```

### Ray processes

```bash
# Stop all Ray processes
ray stop
```

## Implementation Details

The Ray implementation (`nstream-ray.py`) is based on the MPI parallelization scheme (`nstream-numpy-mpi.py`) but uses Ray's distributed computing framework instead of MPI. Key features:

- **Domain decomposition**: Vector is split into chunks across workers
- **Pre-initialized data**: Arrays are allocated and initialized before timing begins
- **Separated timing**: Validation is computed separately from timed operations
- **Proper timing isolation**: Only the core computation loop is timed, excluding initialization and validation
- **Fault tolerance**: Ray provides automatic error handling and task retry
- **Scalability**: Can run on single machine or distributed cluster

## Files

- `nstream-ray.py`: Ray-based parallel implementation
- `nstream-numpy.py`: Original single-threaded numpy implementation  
- `nstream-numpy-mpi.py`: MPI-based parallel implementation
- `README-Ray.md`: This setup guide

## Further Reading

- [Ray Documentation](https://docs.ray.io/)
- [Ray Core Walkthrough](https://docs.ray.io/en/latest/ray-core/walkthrough.html)
- [Parallel Research Kernels](https://github.com/ParRes/Kernels)
