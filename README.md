# MPI + NCCL Peer-to-Peer Communication Example (high-performance computing (HPC))
## Introduction
This program demonstrates peer-to-peer communication between GPUs across multiple nodes using MPI (Message Passing Interface) and NCCL (NVIDIA Collective Communications Library). It integrates CUDA, MPI, and NCCL to facilitate efficient inter-GPU communication in a multi-node distributed system.

## The program follows these key steps:

1. MPI Initialization:\
  - Initializes the MPI environment, and retrieves the world rank and size.
    
2. NCCL Unique ID Handling:\
  - Rank 0 generates an NCCL unique ID, which is broadcasted to all other ranks.

3. GPU Selection & Memory Allocation:\
  - Each MPI rank selects a GPU based on its rank ID (modulo the number of GPUs per node).
  - Memory buffers are allocated on the selected GPUs for communication.

4. NCCL Communicator Initialization:\
  - Each rank initializes an NCCL communicator with the world size and unique ID.

5. Peer-to-Peer Communication:\
  - The program demonstrates direct data exchange between two designated ranks (PEER0 and PEER1) using ncclSend and ncclRecv.
  - Rank PEER0 sends an array of floating-point numbers to Rank PEER1.
  - Rank PEER1 receives the data and verifies its correctness.

6. Synchronization & Cleanup:\
  - Ensures data transfers are completed using CUDA stream synchronization.
  - Cleans up allocated resources, destroys NCCL communicators, and finalizes MPI.

## Key Features
  - Uses MPI for process communication and rank management.
  - Uses NCCL for optimized GPU-to-GPU communication.
  - Implements CUDA memory management and kernel execution with CUDA streams.
  - Supports multi-node environments with up to 8 GPUs per node (configurable via RANK_PER_NODE).
  - Demonstrates direct peer-to-peer GPU communication using ncclSend and ncclRecv.
