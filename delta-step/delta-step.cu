#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

// parameter to tune
#define DELTA 10

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
__device__ __inline__
uint relax(uint v, uint new_dist, uint *dists, uint *bucket_num)
{
    atomicMin(&(dists[v]), new_dist);
    uint new_bucket = dists[v] / DELTA;
    atomicMin(&(bucket_num[v]), new_bucket);
    return bucket_num[v];
}

__global__ 
void baseline_Delta_initialize(uint *nodes, uint *edges, uint *weights, uint *dists, 
    uint *bucket_num)
{
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint cur_node = 0;

    // idx is the edge index for min_node's neighboring edges
    if (idx >= nodes[cur_node+1] - nodes[cur_node]) return;
    idx += nodes[cur_node];

    uint v = edges[idx];
    // if (dists[cur_node] + weights[idx] < dists[v]) {
    //     dists[v] = dists[cur_node] + weights[idx];
    //     bucket_num[v] = dists[v] / DELTA;
    // }
    uint new_dist = dists[cur_node] + weights[idx];
    relax(v, new_dist, dists, bucket_num);
}

// one inner loop iteration for processing light edges
__global__ 
void baseline_Delta_process_light(uint *nodes, uint *edges, uint *weights, 
    uint *dists, uint *bucket_num, uint *bucket_num_next, bool *bucket_deleted, bool *current_bucket_nonempty, bool *next_bucket, uint current_bucket, uint num_nodes) 
{
    uint v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    if (bucket_num[v] != current_bucket) 
    {
        // preserve other buckets
        bucket_num_next[v] = bucket_num[v];
        return;
    }

    // delete from bucket
    bucket_deleted[v] = true;
    bucket_num[v] = INT_MAX / DELTA;    // so that next iteration always gets an empty bucket after swapping bucket_num with .._next

    for (uint i = nodes[v]; i < nodes[v+1]; i++) {
        uint u = edges[i];
        uint u_weight = weights[i];
        if (u_weight > DELTA) continue; // skip a heavy edge
        // updating an edge from v to u
        uint new_dist = dists[v] + u_weight;
        uint new_bucket = relax(u, new_dist, dists, bucket_num_next);
        if (new_bucket == current_bucket) *current_bucket_nonempty = true;
        if (new_bucket > current_bucket) *next_bucket = true;
    }
}

// update all heavy edges, once for each bucket
__global__ 
void baseline_Delta_process_heavy(uint *nodes, uint *edges, uint *weights, 
    uint *dists, uint *bucket_num, bool *bucket_deleted, bool *next_bucket, uint current_bucket, uint num_nodes) 
{
    uint v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    if (!bucket_deleted[v]) return;

    for (uint i = nodes[v]; i < nodes[v+1]; i++) {
        uint u = edges[i];
        uint u_weight = weights[i];
        if (u_weight > DELTA)
        {
            // updating an edge from v to u
            uint new_dist = dists[v] + u_weight;
            uint new_bucket = relax(u, new_dist, dists, bucket_num);
            if (new_bucket > current_bucket) *next_bucket = true;
        }
    }
    bucket_deleted[v] = false; // restore for next use
}

// END BASELINE VERSION ******************************** 


// WARP-BASED VERSION ******************************** 

// END WARP-BASED VERSION ******************************** 


void baseline_Delta() {
    uint *device_nodes, *device_edges, *device_weights, *device_dists;
    uint *bucket_num, *bucket_num_next;    // which bucket the node belongs to
    bool *bucket_deleted;   // whether deleted before processing heavy
    uint *device_bucket_num, *device_bucket_num_next;
    bool *device_bucket_deleted;
    bool next_bucket, current_bucket_nonempty, *device_next_bucket, *device_current_bucket_nonempty;

    // TODO: how do we compute number of blocks and threads per block
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaCheckError(cudaMalloc(&device_nodes, (N+1) * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_edges, M * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_weights, M * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_dists, N * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_bucket_num, N * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_bucket_num_next, N * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_bucket_deleted, N * sizeof(bool)));
    cudaCheckError(cudaMalloc(&device_next_bucket, sizeof(bool)));
    cudaCheckError(cudaMalloc(&device_current_bucket_nonempty, sizeof(bool)));

    bucket_num = new uint[N];
    bucket_num_next = new uint[N];
    bucket_deleted = new bool[N];
    for (uint i = 0; i < N; i++) {
        bucket_num[i] = dists[i] / DELTA;
        bucket_num_next[i] = dists[i] / DELTA;
        bucket_deleted[i] = false;
    }

    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    cudaCheckError(cudaMemcpy(device_nodes, nodes, (N+1) * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_edges, edges, M * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_weights, weights, M * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_dists, dists, N * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_bucket_num, bucket_num, N * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_bucket_num_next, bucket_num_next, N * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_bucket_deleted, bucket_deleted, N * sizeof(bool), cudaMemcpyHostToDevice));

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured before launching: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    // run kernel
    double kernelStartTime = CycleTimer::currentSeconds();

    // should not affect correctness, just an optimization for first iteration
    baseline_Delta_initialize<<<blocks, threadsPerBlock>>>(device_nodes, device_edges, device_weights, device_dists, device_bucket_num);
    cudaCheckError ( cudaDeviceSynchronize() );

    next_bucket = true;
    uint i = 0;
    while (next_bucket) {
        next_bucket = false;
        cudaCheckError(cudaMemcpy(device_next_bucket, &next_bucket, sizeof(bool), cudaMemcpyHostToDevice));
        current_bucket_nonempty = true;
        while (current_bucket_nonempty)
        {
            current_bucket_nonempty = false;
            cudaCheckError(cudaMemcpy(device_current_bucket_nonempty, &current_bucket_nonempty, sizeof(bool), cudaMemcpyHostToDevice));
            baseline_Delta_process_light<<<blocks, threadsPerBlock>>>(device_nodes, device_edges, device_weights, device_dists, device_bucket_num, device_bucket_num_next, device_bucket_deleted, device_current_bucket_nonempty, device_next_bucket, i, N);
            cudaCheckError ( cudaDeviceSynchronize() );
            cudaCheckError(cudaMemcpy(&current_bucket_nonempty, device_current_bucket_nonempty, sizeof(bool), cudaMemcpyDeviceToHost));

            // swap
            uint *tmp = bucket_num;
            bucket_num = bucket_num_next;
            bucket_num_next = tmp;
        }
        
        baseline_Delta_process_heavy<<<blocks, threadsPerBlock>>>(device_nodes, device_edges, device_weights, device_dists, device_bucket_num, device_bucket_deleted, device_next_bucket, i, N);
        cudaCheckError ( cudaDeviceSynchronize() );
        cudaCheckError(cudaMemcpy(&next_bucket, device_next_bucket, sizeof(bool), cudaMemcpyDeviceToHost));
        i++;
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

    cudaCheckError( cudaFree(device_nodes) );
    cudaCheckError( cudaFree(device_edges) );
    cudaCheckError( cudaFree(device_weights) );
    cudaCheckError( cudaFree(device_dists) );
    cudaCheckError( cudaFree(device_bucket_num) );
    cudaCheckError( cudaFree(device_bucket_num_next) );
    cudaCheckError( cudaFree(device_bucket_deleted) );
    cudaCheckError( cudaFree(device_next_bucket) );
    cudaCheckError( cudaFree(device_current_bucket_nonempty) );
    delete[] bucket_num;
    delete[] bucket_num_next;
    delete[] bucket_deleted;
}