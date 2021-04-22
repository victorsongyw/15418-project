#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

#define DIST_OFFSET 32
// #define DIST_MASK 0xFFFFFFFF00000000
#define NODE_MASK 0xFFFFFFFF

#define threadsPerBlock 512

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
void baseline_Dijkstra_find_next_node(uint *nodes, uint *edges, uint *weights, uint *dists,
                              bool *finalized, unsigned long long int *min_dist_and_node, int num_nodes) {
    uint v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    if (finalized[v]) return;
    unsigned long long int dist_and_node = ((unsigned long long int)dists[v] << DIST_OFFSET) | (unsigned long long int)v;
    atomicMin(min_dist_and_node, dist_and_node); // dist is the upper bits, so we overwrite only if we have a smaller dist
}

__global__ 
void baseline_Dijkstra_update_dists(uint *nodes, uint *edges, uint *weights, uint *dists,
                              bool *finalized, unsigned long long int *min_dist_and_node, int num_nodes) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint min_node = *min_dist_and_node & NODE_MASK;
    finalized[min_node] = true;

    // idx is the edge index for min_node's neighboring edges
    if (idx >= nodes[min_node+1] - nodes[min_node]) return;
    idx += nodes[min_node];

    uint v = edges[idx];
    if (!finalized[v] && dists[min_node] + weights[idx] < dists[v]) {
        dists[v] = dists[min_node] + weights[idx];
    }
}

// END BASELINE VERSION ******************************** 


// WARP-BASED VERSION ******************************** 

__global__ 
void warp_Dijkstra_find_next_node(uint *nodes, uint *edges, uint *weights, uint *dists,
                              bool *finalized, unsigned long long int *min_dist_and_node, int num_nodes) {
    uint v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;

    // copy my work to shared memory
    __shared__ uint warp_dist[threadsPerBlock];
    warp_dist[threadIdx.x] = dists[v];
    
    if (finalized[v]) return;
    
    unsigned long long int dist_and_node = ((unsigned long long int)warp_dist[threadIdx.x] << DIST_OFFSET) | (unsigned long long int)v;
    atomicMin(min_dist_and_node, dist_and_node); // dist is the upper bits, so we overwrite only if we have a smaller dist
}

__global__ 
void warp_Dijkstra_update_dists(uint *nodes, uint *edges, uint *weights, uint *dists,
                              bool *finalized, unsigned long long int *min_dist_and_node, int num_nodes) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint min_node = *min_dist_and_node & NODE_MASK;
    finalized[min_node] = true;

    // idx is the edge index for min_node's neighboring edges
    if (idx >= nodes[min_node+1] - nodes[min_node]) return;
    idx += nodes[min_node];

    uint v = edges[idx];
    if (!finalized[v] && dists[min_node] + weights[idx] < dists[v]) {
        dists[v] = dists[min_node] + weights[idx];
    }
}

// END WARP-BASED VERSION ******************************** 


void baseline_Dijkstra() {
    uint *device_nodes, *device_edges, *device_weights, *device_dists;
    bool *finalized;
    bool *device_finalized;
    unsigned long long int min_dist_and_node, *device_min_dist_and_node; // upper 32 bytes represent dist, lower 32 bytes represent node

    // TODO: how do we compute number of blocks and threads per block
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaCheckError(cudaMalloc(&device_nodes, (N+1) * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_edges, M * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_weights, M * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_dists, N * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_finalized, N * sizeof(bool)));
    cudaCheckError(cudaMalloc(&device_min_dist_and_node, sizeof(unsigned long long int)));

    finalized = new bool[N];
    for (uint i = 0; i < N; i++) {
        finalized[i] = false;
    }

    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    cudaCheckError(cudaMemcpy(device_nodes, nodes, (N+1) * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_edges, edges, M * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_weights, weights, M * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_dists, dists, N * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_finalized, finalized, N * sizeof(bool), cudaMemcpyHostToDevice));

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured before launching: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    // run kernel
    double kernelStartTime = CycleTimer::currentSeconds();

    for (uint i = 0; i < N-1; i++) {
        min_dist_and_node = ULLONG_MAX;
        cudaCheckError(cudaMemcpy(device_min_dist_and_node, &min_dist_and_node, sizeof(unsigned long long int), cudaMemcpyHostToDevice));
        baseline_Dijkstra_find_next_node<<<blocks, threadsPerBlock>>>(device_nodes, device_edges, device_weights, device_dists, device_finalized, device_min_dist_and_node, N);
        cudaCheckError ( cudaDeviceSynchronize() );
        baseline_Dijkstra_update_dists<<<blocks, threadsPerBlock>>>(device_nodes, device_edges, device_weights, device_dists, device_finalized, device_min_dist_and_node, N);
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
    cudaFree(device_finalized);
    delete[] finalized;
}
