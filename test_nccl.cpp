#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <nccl.h>
#include <cuda_runtime.h>

#define RANK_PER_NODE 8
#define PEER0 0
#define PEER1 14
// Helper macro to check CUDA errors
#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(err));                                      \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

// Helper macro to check NCCL errors
#define CHECK_NCCL(call)                                      \
  do {                                                        \
    ncclResult_t res = call;                                  \
    if (res != ncclSuccess) {                                 \
      fprintf(stderr, "NCCL error at %s:%d: %s\n", __FILE__,  \
              __LINE__, ncclGetErrorString(res));             \
      exit(EXIT_FAILURE);                                     \
    }                                                         \
  } while (0)

int main(int argc, char* argv[]) {
    // 1. Initialize MPI
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if(world_rank == 0)
        printf("Total rank: %d \n", world_size);

    // 2. Get NCCL unique ID (only rank 0 creates, then bcast to others)
    ncclUniqueId id;
    if (world_rank == 0) {
        CHECK_NCCL(ncclGetUniqueId(&id));
    }
    MPI_Bcast((void*)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // 3. Each rank picks a GPU (for instance, GPU = world_rank % 8)
    int local_gpu = world_rank % RANK_PER_NODE;
    CHECK_CUDA(cudaSetDevice(local_gpu));

    // 4. Create NCCL communicator
    ncclComm_t comm;
    CHECK_NCCL(ncclCommInitRank(&comm, world_size, id, world_rank));

    // 5. Allocate GPU buffers
    size_t N = 1024; // number of floats
    float* d_sendbuff;
    float* d_recvbuff;
    CHECK_CUDA(cudaMalloc(&d_sendbuff, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_recvbuff, N * sizeof(float)));

    // Initialize send buffer (only on rank 0 for demonstration)
    if (world_rank == 0) {
        float* h_tmp = (float*)malloc(N * sizeof(float));
        for (size_t i = 0; i < N; i++) {
            h_tmp[i] = (float)(i);
        }
        CHECK_CUDA(cudaMemcpy(d_sendbuff, h_tmp, N*sizeof(float),
                              cudaMemcpyHostToDevice));
        free(h_tmp);
    }

    // 6. Create a CUDA stream
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));


    if (world_rank == PEER0) {
        int peer = PEER1;
        CHECK_NCCL(ncclSend((const void*)d_sendbuff,
                            N, /*count*/
                            ncclFloat,
                            peer, /* dest rank */
                            comm,
                            stream));
    } else if (world_rank == PEER1) {
        int peer = PEER0;
        CHECK_NCCL(ncclRecv((void*)d_recvbuff,
                            N,
                            ncclFloat,
                            peer, /* source rank */
                            comm,
                            stream));
    }

    CHECK_CUDA(cudaStreamSynchronize(stream));

    if (world_rank == PEER1) {
        float* h_result = (float*)malloc(N * sizeof(float));
        CHECK_CUDA(cudaMemcpy(h_result, d_recvbuff, N*sizeof(float),
                            	cudaMemcpyDeviceToHost));

        printf("Node %d _ Rank %d received data: %f, %f, %f ...\n",
            	PEER1 / RANK_PER_NODE, PEER1 % RANK_PER_NODE, h_result[0], h_result[1], h_result[2]);
        free(h_result);
    }

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_sendbuff));
    CHECK_CUDA(cudaFree(d_recvbuff));
    CHECK_NCCL(ncclCommDestroy(comm));

    MPI_Finalize();
    return 0;
}
