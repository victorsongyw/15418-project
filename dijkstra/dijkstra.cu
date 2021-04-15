#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

extern float toBW(int bytes, float sec);
extern int N, M;
extern int *nodes, edges, weights, dists;

int* device_nodes, device_edges, device_weights, device_dists;

__global__ 
void baseline_Dijkstra_kernel() {
    
}

void baseline_Dijkstra() {

    int totalBytes = sizeof(int) * (N + M) * 2;

    // TODO: compute number of blocks and threads per block
    // const int threadsPerBlock = 512;
    // const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc(&device_nodes, (N+1) * sizeof(int));
    cudaMalloc(&device_edges, M * sizeof(int));
    cudaMalloc(&device_weights, M * sizeof(int));
    cudaMalloc(&device_dists, N * sizeof(int));

    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    cudaMemcpy(device_nodes, nodes, (N+1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_edges, edges, M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_weights, weights, M * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_dists, dists, N * sizeof(int), cudaMemcpyHostToDevice);

    // run kernel
    double kernelStartTime = CycleTimer::currentSeconds();

    for (round = 0; round < N; round++) {
        baseline_Dijkstra_kernel<<<blocks, threadsPerBlock>>>(round);
        cudaDeviceSynchronize();
    }

    double kernelEndTime = CycleTimer::currentSeconds();

    cudaMemcpy(dists, device_dists, N * sizeof(int), cudaMemcpyDeviceToHost);

    // end timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    double kernelDuration = kernelEndTime - kernelStartTime;
    printf("Overall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));
    printf("Kernel: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * kernelDuration, toBW(totalBytes, kernelDuration));

    cudaFree(device_nodes);
    cudaFree(device_edges);
    cudaFree(device_weights);
    cudaFree(device_dists);
}
