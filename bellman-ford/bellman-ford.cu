#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 512
#define CHUNK_SIZE 8 // can be also thought as nodes per chunk
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (THREADS_PER_BLOCK / WARP_SIZE)
#define NODES_PER_BLOCK (THREADS_PER_BLOCK / WARP_SIZE * CHUNK_SIZE)

extern float toBW(int bytes, float sec);
extern uint N, M;
extern uint *nodes, *edges, *weights, *dists;

#define DEBUG
#ifdef DEBUG
#define cudaCheckError(ans) cudaAssert((ans), __FILE__, __LINE__);
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n",
        cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#else
#define cudaCheckError(ans) ans
#endif

// BASELINE VERSION ******************************** 

__global__ 
void baseline_BF_kernel(uint *nodes, uint *edges, uint *weights, uint *dists, uint num_nodes) {
    uint v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    for (uint i = nodes[v]; i < nodes[v+1]; i++) {
        uint u = edges[i];
        // updating an edge from v to u
        uint new_dist = dists[v] + weights[i];
        if (new_dist < dists[u]) {
            dists[u] = new_dist;
        }
    }
}

// END BASELINE VERSION ******************************** 


// WARP-BASED VERSION ******************************** 

__inline__ __device__ 
void warp_memcpy(uint start, uint offset, uint end, uint *warp_array, uint *array) {
    for (uint i = start+offset; i < end; i += WARP_SIZE) {
        warp_array[i-start] = array[i];
    }
}

__inline__ __device__
void warp_update_neighbors(uint start, uint end, uint *edges, uint *dists, uint *warp_dists, uint *weights, uint v) {
    for (uint i = start; i < end; i += WARP_SIZE) {
        uint u = edges[i];
        // updating an edge from v to u
        uint new_dist = warp_dists[v] + weights[i];
        if (new_dist < dists[u]) {
            dists[u] = new_dist;
        }
        // atomicMin(&(dists[u]), new_dist);
    }
}

__global__ 
void warp_BF_kernel(uint *nodes, uint *edges, uint *weights, uint *dists, uint num_nodes) {

    // uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint warp_offset = threadIdx.x % WARP_SIZE;
    uint warp_id = threadIdx.x / WARP_SIZE;

    // this is the range of indexes of nodes for which this warp is responsible
    uint chunkStart = blockIdx.x * NODES_PER_BLOCK + warp_id * CHUNK_SIZE;
    if (chunkStart >= num_nodes) return;
    uint chunkEnd = chunkStart + CHUNK_SIZE;
    if (chunkEnd > num_nodes) chunkEnd = num_nodes;
    
    // shared memory across threads in a block
    __shared__ uint block_nodes[NODES_PER_BLOCK + WARPS_PER_BLOCK];
    __shared__ uint block_dists[NODES_PER_BLOCK];

    // pointers to the start of the region corresponding to this warp
    uint *warp_nodes = block_nodes + warp_id * (CHUNK_SIZE+1); 
    uint *warp_dists = block_dists + warp_id * CHUNK_SIZE;
    
    warp_memcpy(chunkStart, warp_offset, chunkEnd+1, warp_nodes, nodes);
    warp_memcpy(chunkStart, warp_offset, chunkEnd, warp_dists, dists);

    // iterate over my work
    for (uint v = 0; v < chunkEnd - chunkStart; v++) {
        uint nbr_start = warp_nodes[v];
        uint nbr_end = warp_nodes[v+1];
        warp_update_neighbors(nbr_start + warp_offset, nbr_end, edges, dists, warp_dists, weights, v);
    }

}

// END WARP-BASED VERSION ******************************** 

// main function
void bellman_ford(bool use_warp) {
    uint *device_nodes, *device_edges, *device_weights, *device_dists;
  
    // TODO: how do we compute number of blocks and threads per block
    int blocks;
    if (!use_warp) blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    else blocks = (N + NODES_PER_BLOCK - 1) / NODES_PER_BLOCK;

    cudaCheckError(cudaMalloc(&device_nodes, (N+1) * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_edges, M * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_weights, M * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_dists, N * sizeof(uint)));

    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    cudaCheckError(cudaMemcpy(device_nodes, nodes, (N+1) * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_edges, edges, M * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_weights, weights, M * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_dists, dists, N * sizeof(uint), cudaMemcpyHostToDevice));

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured before launching: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    // run kernel
    double kernelStartTime = CycleTimer::currentSeconds();

    for (uint i = 0; i < N-1; i++) {
        if (!use_warp)
            baseline_BF_kernel<<<blocks, THREADS_PER_BLOCK>>>(device_nodes, device_edges, device_weights, device_dists, N);
        else
            warp_BF_kernel<<<blocks, THREADS_PER_BLOCK>>>(device_nodes, device_edges, device_weights, device_dists, N);
        
        cudaCheckError ( cudaDeviceSynchronize() );
    }

    double kernelEndTime = CycleTimer::currentSeconds();

    cudaMemcpy(dists, device_dists, N * sizeof(uint), cudaMemcpyDeviceToHost);
    
    // printf("dists:\n");
    // for (int i = 0; i < N; i++) {
    //     printf("%d: %d\n", i, dists[i]);
    // }

    // end timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();

    errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured after launching: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    double kernelDuration = kernelEndTime - kernelStartTime;
    int totalBytes = sizeof(uint) * (N + M) * 2; // TODO: UPDATE LATER
    printf("CUDA Baseline - Overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));
    printf("CUDA Baseline - Kernel: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * kernelDuration, toBW(totalBytes, kernelDuration));

    cudaFree(device_nodes);
    cudaFree(device_edges);
    cudaFree(device_weights);
    cudaFree(device_dists);
}