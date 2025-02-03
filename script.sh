#!/bin/bash
#SBATCH --account=GOV113123
#SBATCH --partition=QUEUE
#SBATCH -J QAOA_test
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gpus=16
#SBATCH --gpus-per-task=1
#SBATCH --gpus-per-node=8
#SBATCH -o out_multi_node.log
#SBATCH -e err_multi_node.log

module purge
module load cuda/12.2 
module load nvhpc/24.7
# module load singularity

export UCX_NET_DEVICES=mlx5_0:1
export UCX_IB_GPU_DIRECT_RDMA=1
export MPI_HOME=/work/HPC_software/LMOD/nvidia/packages/hpc_sdk-24.7/Linux_x86_64/24.7/comm_libs/mpi/
export NCCL_ROOT=/work/HPC_software/LMOD/nvidia/packages/hpc_sdk-24.7/Linux_x86_64/24.7/comm_libs/nccl/

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_EXT=1
export NCCL_SOCKET_IFNAME=ib0

mpirun -np 16 -bind-to none --map-by ppr:8:node ./test_nccl

