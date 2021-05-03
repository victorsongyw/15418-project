#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

// parameter to tune
#define DELTA 5
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

// Relax node v with distance new_dist and update its bucket if needed.
// flag is set to true iff the distance actually gets updated.
__device__ __inline__
uint relax(uint v, uint new_dist, uint *dists, uint *bucket_num, bool *flag)
{
    uint old_dist = atomicMin(&(dists[v]), new_dist);
    uint new_bucket = dists[v] / DELTA;
    atomicMin(&(bucket_num[v]), new_bucket);
    *flag = (new_dist < old_dist);
    return bucket_num[v];
}

__global__
void delta_find_next_bucket(uint *bucket_num, uint *next_bucket, uint curr_bucket, uint num_nodes) {
    uint v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    if (bucket_num[v] > curr_bucket) {
        atomicMin(next_bucket, bucket_num[v]);
    }
}

// BASELINE VERSION ******************************** 

__global__ 
void baseline_delta_initialize(uint *nodes, uint *edges, uint *weights, uint *dists, 
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
    bool updated;
    relax(v, new_dist, dists, bucket_num, &updated);
}

__global__ 
void baseline_delta_process(bool process_light, uint *nodes, uint *edges, uint *weights, uint *dists, uint *bucket_num, 
    uint curr_bucket, uint num_nodes, uint *bucket_num_next = NULL, bool *curr_bucket_nonempty = NULL) 
{
    uint v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    if (bucket_num[v] != curr_bucket) return;

    for (uint i = nodes[v]; i < nodes[v+1]; i++) {
        uint u = edges[i];
        uint u_weight = weights[i];
        if (u_weight > DELTA && !process_light) {
            // updating a heavy edge from v to u
            uint new_dist = dists[v] + u_weight;
            bool updated = false;
            relax(u, new_dist, dists, bucket_num, &updated);
        }
        else if (u_weight <= DELTA && process_light) {
            // updating a light edge from v to u
            uint new_dist = dists[v] + u_weight;
            bool updated = false;
            uint new_bucket = relax(u, new_dist, dists, bucket_num_next, &updated);
            if (updated && new_bucket == curr_bucket) *curr_bucket_nonempty = true;
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

__global__
void warp_delta_process(bool process_light, uint *nodes, uint *edges, uint *weights, uint *dists, uint *bucket_num, 
    uint curr_bucket, uint num_nodes, uint *bucket_num_next = NULL, bool *curr_bucket_nonempty = NULL)
{
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
    __shared__ uint block_bucket_num[NODES_PER_BLOCK];

    // pointers to the start of the region corresponding to this warp
    uint *warp_nodes = block_nodes + warp_id * (CHUNK_SIZE+1); 
    uint *warp_dists = block_dists + warp_id * CHUNK_SIZE;
    uint *warp_bucket_num = block_bucket_num + warp_id * CHUNK_SIZE;

    warp_memcpy(chunkStart, warp_offset, chunkEnd+1, warp_nodes, nodes);
    warp_memcpy(chunkStart, warp_offset, chunkEnd, warp_dists, dists);
    warp_memcpy(chunkStart, warp_offset, chunkEnd, warp_bucket_num, bucket_num);

    // iterate over my work
    for (uint v = 0; v < chunkEnd - chunkStart; v++) {
        if (warp_bucket_num[v] == curr_bucket) {
            uint nbr_start = warp_nodes[v]; 
            uint nbr_end = warp_nodes[v+1];
            for (uint i = nbr_start + warp_offset; i < nbr_end; i += WARP_SIZE) {
                uint u = edges[i];
                uint u_weight = weights[i];

                if (u_weight > DELTA && !process_light) {
                    // updating a heavy edge from v to u
                    uint new_dist = warp_dists[v] + u_weight;
                    bool updated = false;
                    relax(u, new_dist, dists, bucket_num, &updated);
                }
                else if (u_weight <= DELTA && process_light) {
                    // updating a light edge from v to u
                    uint new_dist = warp_dists[v] + u_weight;
                    bool updated = false;
                    uint new_bucket = relax(u, new_dist, dists, bucket_num_next, &updated);
                    if (updated && new_bucket == curr_bucket) *curr_bucket_nonempty = true;
                }
            }
        }
    }
}

// END WARP-BASED VERSION ******************************** 


// main function
void delta_stepping(bool use_warp) {
    uint *device_nodes, *device_edges, *device_weights, *device_dists;
    uint *bucket_num, *bucket_num_next;    // which bucket the node belongs to
    uint *device_bucket_num, *device_bucket_num_next;
    bool curr_bucket_nonempty, *device_curr_bucket_nonempty;
    uint next_bucket, *device_next_bucket;

    // TODO: how do we compute number of blocks and threads per block
    int blocks;
    if (!use_warp) blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    else blocks = (N + NODES_PER_BLOCK - 1) / NODES_PER_BLOCK;

    cudaCheckError(cudaMalloc(&device_nodes, (N+1) * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_edges, M * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_weights, M * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_dists, N * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_bucket_num, N * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_bucket_num_next, N * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_next_bucket, sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_curr_bucket_nonempty, sizeof(bool)));

    bucket_num = new uint[N];
    bucket_num_next = new uint[N];
    for (uint i = 0; i < N; i++) {
        bucket_num[i] = dists[i] / DELTA;
        bucket_num_next[i] = dists[i] / DELTA;
    }

    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    cudaCheckError(cudaMemcpy(device_nodes, nodes, (N+1) * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_edges, edges, M * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_weights, weights, M * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_dists, dists, N * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_bucket_num, bucket_num, N * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_bucket_num_next, bucket_num_next, N * sizeof(uint), cudaMemcpyHostToDevice));
    
    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured before launching: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    // run kernel
    double kernelStartTime = CycleTimer::currentSeconds();

    // should not affect correctness, just an optimization for first iteration
    // baseline_delta_initialize<<<blocks, THREADS_PER_BLOCK>>>(device_nodes, device_edges, device_weights, device_dists, device_bucket_num);
    // cudaCheckError(cudaDeviceSynchronize());

    uint curr_bucket = 0;
    while (true) {
        curr_bucket_nonempty = true;
        while (curr_bucket_nonempty) // loop until the current bucket is empty
        {
            curr_bucket_nonempty = false;
            cudaCheckError(cudaMemcpy(device_curr_bucket_nonempty, &curr_bucket_nonempty, sizeof(bool), cudaMemcpyHostToDevice));
            
            cudaCheckError(cudaMemcpy(device_bucket_num_next, device_bucket_num, N * sizeof(uint), cudaMemcpyDeviceToDevice));
    
            if (!use_warp) {
                // run baseline kernels
                baseline_delta_process<<<blocks, THREADS_PER_BLOCK>>>(
                    true, device_nodes, device_edges, device_weights, device_dists, device_bucket_num,
                    curr_bucket, N, device_bucket_num_next, device_curr_bucket_nonempty);
            }
            else {
                // run warp-centric kernels
                warp_delta_process<<<blocks, THREADS_PER_BLOCK>>>(
                    true, device_nodes, device_edges, device_weights, device_dists, device_bucket_num, 
                    curr_bucket, N, device_bucket_num_next, device_curr_bucket_nonempty);
            }
            cudaCheckError(cudaDeviceSynchronize());
            cudaCheckError(cudaMemcpy(&curr_bucket_nonempty, device_curr_bucket_nonempty, sizeof(bool), cudaMemcpyDeviceToHost));
            
            // swap for next phase
            std::swap(device_bucket_num, device_bucket_num_next);
        }
         
        if (!use_warp) {
            // run baseline kernels
            baseline_delta_process<<<blocks, THREADS_PER_BLOCK>>>(
                false, device_nodes, device_edges, device_weights, device_dists, device_bucket_num, curr_bucket, N);
        }
        else {
            warp_delta_process<<<blocks, THREADS_PER_BLOCK>>>(
                false, device_nodes, device_edges, device_weights, device_dists, device_bucket_num, curr_bucket, N);
        }

        cudaCheckError(cudaDeviceSynchronize());

        next_bucket = INT_MAX;
        cudaCheckError(cudaMemcpy(device_next_bucket, &next_bucket, sizeof(uint), cudaMemcpyHostToDevice));
        delta_find_next_bucket<<<blocks, THREADS_PER_BLOCK>>>(
                    device_bucket_num, device_next_bucket, curr_bucket, N);
        cudaCheckError(cudaDeviceSynchronize());
        cudaCheckError(cudaMemcpy(&next_bucket, device_next_bucket, sizeof(uint), cudaMemcpyDeviceToHost));
        if (next_bucket == INT_MAX) break; // WE ARE DONE
        curr_bucket = next_bucket;
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
    if (!use_warp) {
        printf("CUDA Baseline\n");
    } else {
        printf("CUDA Warp\n");
    }
    printf("\tOverall: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));
    printf("\tKernel: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * kernelDuration, toBW(totalBytes, kernelDuration));

    cudaCheckError(cudaFree(device_nodes));
    cudaCheckError(cudaFree(device_edges));
    cudaCheckError(cudaFree(device_weights));
    cudaCheckError(cudaFree(device_dists));
    cudaCheckError(cudaFree(device_bucket_num));
    cudaCheckError(cudaFree(device_bucket_num_next));
    cudaCheckError(cudaFree(device_next_bucket));
    cudaCheckError(cudaFree(device_curr_bucket_nonempty));
    delete[] bucket_num;
    delete[] bucket_num_next;
}