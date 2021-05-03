#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"

// parameter to tune
#define DELTA 3
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
void delta_delete_bucket(uint *nodes, uint *edges, uint *weights, 
    uint *dists, uint *bucket_num, uint *bucket_num_next, uint num_nodes) 
{
    uint v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    // if (bucket_num[v] != curr_bucket) 
    // {
        // preserve other buckets
        bucket_num_next[v] = bucket_num[v];
    // }
    // else
    // {
    //     // save for later heavy edge processing
    //     // bucket_num_next[v] = INT_MAX / DELTA;
    //     bucket_num_next[v] = bucket_num[v];
    //     // bucket_deleted[v] = true;
    // }
}

__global__
void delta_find_next_bucket(uint *bucket_num, uint *next_bucket, uint curr_bucket, uint num_nodes) {
    uint v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    printf("node %d in bucket %d\n", v, bucket_num[v]);
    if (bucket_num[v] > curr_bucket) {
        atomicMin(next_bucket, bucket_num[v]);
    }
}

// One inner loop iteration for processing light edges.
__global__ 
void baseline_delta_process_light(uint *nodes, uint *edges, uint *weights, uint *dists, uint *bucket_num, uint *bucket_num_next, 
    bool *current_bucket_nonempty, uint current_bucket, uint num_nodes) 
{
    uint v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    if (bucket_num[v] != current_bucket) return;

    for (uint i = nodes[v]; i < nodes[v+1]; i++) {
        uint u = edges[i];
        uint u_weight = weights[i];
        if (u_weight > DELTA) continue; // skip a heavy edge
        // updating an edge from v to u
        uint new_dist = dists[v] + u_weight;
        bool updated = false;
        uint new_bucket = relax(u, new_dist, dists, bucket_num_next, &updated);
        if (updated && new_bucket == current_bucket) *current_bucket_nonempty = true;
        // if (updated && new_bucket > current_bucket) *next_bucket = true;
    }
}

// Update all heavy edges (once for each bucket).
__global__ 
void baseline_delta_process_heavy(uint *nodes, uint *edges, uint *weights, uint *dists, 
    uint *bucket_num, uint current_bucket, uint num_nodes) 
{
    uint v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_nodes) return;
    if (bucket_num[v] != current_bucket) return;

    for (uint i = nodes[v]; i < nodes[v+1]; i++) {
        uint u = edges[i];
        uint u_weight = weights[i];
        if (u_weight > DELTA)
        {
            // updating an edge from v to u
            uint new_dist = dists[v] + u_weight;
            bool updated = false;
            uint new_bucket = relax(u, new_dist, dists, bucket_num, &updated);
            // if (updated && new_bucket > current_bucket) *next_bucket = true;
        }
    }
}

// END BASELINE VERSION ******************************** 


// WARP-BASED VERSION ******************************** 
// __inline__ __device__ 
// void warp_memcpy(uint start, uint offset, uint end, uint *warp_array, uint *array) {
//     for (uint i = start+offset; i < end; i += WARP_SIZE) {
//         warp_array[i-start] = array[i];
//     }
// }

// __inline__ __device__ 
// void warp_memcpy(uint start, uint offset, uint end, bool *warp_array, bool *array) {
//     for (uint i = start+offset; i < end; i += WARP_SIZE) {
//         warp_array[i-start] = array[i];
//     }
// }

// One inner loop iteration for processing light edges.
// __global__ 
// void warp_delta_process_light(uint *nodes, uint *edges, uint *weights, uint *dists, uint *bucket_num, uint *bucket_num_next, 
//     bool *bucket_deleted, bool *current_bucket_nonempty, bool *next_bucket, uint current_bucket, uint num_nodes, uint *dists_copy) 
// {
//     uint warp_offset = threadIdx.x % WARP_SIZE;
//     uint warp_id = threadIdx.x / WARP_SIZE;

//     // this is the range of indexes of nodes for which this warp is responsible
//     uint chunkStart = blockIdx.x * NODES_PER_BLOCK + warp_id * CHUNK_SIZE;
//     if (chunkStart >= num_nodes) return;
//     uint chunkEnd = chunkStart + CHUNK_SIZE;
//     if (chunkEnd > num_nodes) chunkEnd = num_nodes;

//     // shared memory across threads in a block
//     __shared__ uint block_nodes[NODES_PER_BLOCK + WARPS_PER_BLOCK];
//     __shared__ uint block_dists[NODES_PER_BLOCK];
//     __shared__ uint block_bucket_num[NODES_PER_BLOCK];

//     // pointers to the start of the region corresponding to this warp
//     uint *warp_nodes = block_nodes + warp_id * (CHUNK_SIZE+1); 
//     uint *warp_dists = block_dists + warp_id * CHUNK_SIZE;
//     uint *warp_bucket_num = block_bucket_num + warp_id * CHUNK_SIZE;

//     warp_memcpy(chunkStart, warp_offset, chunkEnd+1, warp_nodes, nodes);
//     warp_memcpy(chunkStart, warp_offset, chunkEnd, warp_dists, dists);
//     warp_memcpy(chunkStart, warp_offset, chunkEnd, warp_bucket_num, bucket_num);

//     // iterate over my work
//     for (uint v = 0; v < chunkEnd - chunkStart; v++) {
//         if (warp_bucket_num[v] == current_bucket) {
//             uint nbr_start = warp_nodes[v]; 
//             uint nbr_end = warp_nodes[v+1];
//             for (uint i = nbr_start; i < nbr_end; i += WARP_SIZE) {
//                 uint u = edges[i];
//                 uint u_weight = weights[i];
//                 if (u_weight > DELTA) continue; // skip a heavy edge
//                 // updating an edge from v to u
//                 uint new_dist = warp_dists[v] + u_weight;
//                 bool updated = false;
//                 uint new_bucket = relax(u, new_dist, dists, bucket_num_next, &updated);
//                 if (updated && new_bucket == current_bucket) *current_bucket_nonempty = true;
//                 if (updated && new_bucket > current_bucket) *next_bucket = true;
//             }
//         }
//     }
// }

// // Update all heavy edges (once for each bucket).
// __global__ 
// void warp_delta_process_heavy(uint *nodes, uint *edges, uint *weights, uint *dists, 
//     uint *bucket_num, bool *bucket_deleted, bool *next_bucket, uint current_bucket, uint num_nodes) 
// {
//     uint warp_offset = threadIdx.x % WARP_SIZE;
//     uint warp_id = threadIdx.x / WARP_SIZE;

//     // this is the range of indexes of nodes for which this warp is responsible
//     uint chunkStart = blockIdx.x * NODES_PER_BLOCK + warp_id * CHUNK_SIZE;
//     if (chunkStart >= num_nodes) return;
//     uint chunkEnd = chunkStart + CHUNK_SIZE;
//     if (chunkEnd > num_nodes) chunkEnd = num_nodes;

//     // shared memory across threads in a block
//     __shared__ uint block_nodes[NODES_PER_BLOCK + WARPS_PER_BLOCK];
//     __shared__ uint block_dists[NODES_PER_BLOCK];
//     __shared__ bool block_bucket_deleted[NODES_PER_BLOCK];

//     // pointers to the start of the region corresponding to this warp
//     uint *warp_nodes = block_nodes + warp_id * (CHUNK_SIZE+1); 
//     uint *warp_dists = block_dists + warp_id * CHUNK_SIZE;
//     bool *warp_bucket_deleted = block_bucket_deleted + warp_id * CHUNK_SIZE;

//     warp_memcpy(chunkStart, warp_offset, chunkEnd+1, warp_nodes, nodes);
//     warp_memcpy(chunkStart, warp_offset, chunkEnd, warp_dists, dists);
//     warp_memcpy(chunkStart, warp_offset, chunkEnd, warp_bucket_deleted, bucket_deleted);

//     // iterate over my work
//     for (uint v = 0; v < chunkEnd - chunkStart; v++) {
//         if (warp_bucket_deleted[v]) {
//             uint nbr_start = warp_nodes[v]; 
//             uint nbr_end = warp_nodes[v+1];
//             for (uint i = nbr_start; i < nbr_end; i += WARP_SIZE) {
//                 uint u = edges[i];
//                 uint u_weight = weights[i];
//                 if (u_weight > DELTA)
//                 {
//                     // updating an edge from v to u
//                     uint new_dist = warp_dists[v] + u_weight;
//                     bool updated = false;
//                     uint new_bucket = relax(u, new_dist, dists, bucket_num, &updated);
//                     if (updated && new_bucket > current_bucket) *next_bucket = true;
//                 }
//             }
//         }
//     }
// }



// END WARP-BASED VERSION ******************************** 


void delta_stepping(bool use_warp) {
    uint *device_nodes, *device_edges, *device_weights, *device_dists;
    uint *bucket_num, *bucket_num_next;    // which bucket the node belongs to
    // bool *bucket_deleted;   // whether deleted before processing heavy
    uint *device_bucket_num, *device_bucket_num_next;
    // bool *device_bucket_deleted;
    bool current_bucket_nonempty, *device_current_bucket_nonempty;
    uint next_bucket, *device_next_bucket;

    // TODO: how do we compute number of blocks and threads per block
    const int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaCheckError(cudaMalloc(&device_nodes, (N+1) * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_edges, M * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_weights, M * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_dists, N * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_bucket_num, N * sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_bucket_num_next, N * sizeof(uint)));
    // cudaCheckError(cudaMalloc(&device_bucket_deleted, N * sizeof(bool)));
    cudaCheckError(cudaMalloc(&device_next_bucket, sizeof(uint)));
    cudaCheckError(cudaMalloc(&device_current_bucket_nonempty, sizeof(bool)));

    bucket_num = new uint[N];
    bucket_num_next = new uint[N];
    // bucket_deleted = new bool[N];
    for (uint i = 0; i < N; i++) {
        bucket_num[i] = dists[i] / DELTA;
        bucket_num_next[i] = dists[i] / DELTA;
        // bucket_deleted[i] = false;
    }

    // start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    cudaCheckError(cudaMemcpy(device_nodes, nodes, (N+1) * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_edges, edges, M * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_weights, weights, M * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_dists, dists, N * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_bucket_num, bucket_num, N * sizeof(uint), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(device_bucket_num_next, bucket_num_next, N * sizeof(uint), cudaMemcpyHostToDevice));
    // cudaCheckError(cudaMemcpy(device_bucket_deleted, bucket_deleted, N * sizeof(bool), cudaMemcpyHostToDevice));

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured before launching: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    // run kernel
    double kernelStartTime = CycleTimer::currentSeconds();

    // should not affect correctness, just an optimization for first iteration
    // baseline_delta_initialize<<<blocks, THREADS_PER_BLOCK>>>(device_nodes, device_edges, device_weights, device_dists, device_bucket_num);
    // cudaCheckError(cudaDeviceSynchronize());

    uint *dists_copy;
    uint *dists_baseline;
    uint *dists_warp;
    dists_copy = new uint[N];
    dists_baseline = new uint[N];
    dists_warp = new uint[N];

    // next_bucket = true;
    uint curr_bucket = 0;
    while (true) {
        printf("curr bucket: %d\n", curr_bucket);
        // next_bucket = false;
        // cudaCheckError(cudaMemcpy(device_next_bucket, &next_bucket, sizeof(bool), cudaMemcpyHostToDevice));
        current_bucket_nonempty = true;
        while (current_bucket_nonempty) // loop until the current bucket is empty
        {
            current_bucket_nonempty = false;
            cudaCheckError(cudaMemcpy(device_current_bucket_nonempty, &current_bucket_nonempty, sizeof(bool), cudaMemcpyHostToDevice));
            delta_delete_bucket<<<blocks, THREADS_PER_BLOCK>>>(
                device_nodes, device_edges, device_weights, device_dists, device_bucket_num, device_bucket_num_next, N);
            cudaCheckError(cudaDeviceSynchronize());

            // cudaCheckError(cudaMemcpy(dists_copy, device_dists, N * sizeof(uint), cudaMemcpyDeviceToHost));
            // cudaCheckError(cudaMemcpy(device_dists_copy, dists_copy, N * sizeof(uint), cudaMemcpyHostToDevice));

            // if (!use_warp) {
                // run baseline kernels
                baseline_delta_process_light<<<blocks, THREADS_PER_BLOCK>>>(
                    device_nodes, device_edges, device_weights, device_dists, device_bucket_num, device_bucket_num_next, 
                    device_current_bucket_nonempty, curr_bucket, N);
            // }
            // else {
            //     // run warp-centric kernels
            //     warp_delta_process_light<<<blocks, THREADS_PER_BLOCK>>>(
            //         device_nodes, device_edges, device_weights, device_dists, device_bucket_num, device_bucket_num_next, device_bucket_deleted, 
            //         device_current_bucket_nonempty, device_next_bucket, i, N, device_dists_copy);
            // }
            cudaCheckError(cudaDeviceSynchronize());
            cudaCheckError(cudaMemcpy(&current_bucket_nonempty, device_current_bucket_nonempty, sizeof(bool), cudaMemcpyDeviceToHost));

            // swap for next phase
            std::swap(device_bucket_num, device_bucket_num_next);
        }

        // cudaCheckError(cudaMemcpy(dists_copy, device_dists, N * sizeof(uint), cudaMemcpyDeviceToHost));
         
        // if (!use_warp) {
            // run baseline kernels
            baseline_delta_process_heavy<<<blocks, THREADS_PER_BLOCK>>>(
                device_nodes, device_edges, device_weights, device_dists, device_bucket_num, curr_bucket, N);
        // }

        // cudaCheckError(cudaMemcpy(dists_baseline, device_dists, N * sizeof(uint), cudaMemcpyDeviceToHost));
        // cudaCheckError(cudaMemcpy(device_dists, dists_copy, N * sizeof(uint), cudaMemcpyHostToDevice));
        

        // else {
            // baseline_delta_process_heavy<<<blocks, THREADS_PER_BLOCK>>>(
            //     device_nodes, device_edges, device_weights, device_dists, device_bucket_num, device_bucket_deleted, device_next_bucket, i, N);
        // }

        // cudaCheckError(cudaMemcpy(dists_warp, device_dists, N * sizeof(uint), cudaMemcpyDeviceToHost));
            
        // bool err = false;
        // for (uint j = 0; j < N; j++) {
        //     if (dists_warp[j] != dists_baseline[j]) {
        //         printf("At bucket %d, for dists index %d, warp had %d but baseline had %d\n", i, j, dists_warp[j], dists_baseline[j]);
        //         err = true;
        //     }
        // }
        // if (err) exit(1);
        

        cudaCheckError(cudaDeviceSynchronize());

        next_bucket = INT_MAX;
        cudaCheckError(cudaMemcpy(device_next_bucket, &next_bucket, sizeof(uint), cudaMemcpyHostToDevice));
        delta_find_next_bucket<<<blocks, THREADS_PER_BLOCK>>>(
                    device_bucket_num, device_next_bucket, curr_bucket, N);
        cudaCheckError(cudaDeviceSynchronize());
        cudaCheckError(cudaMemcpy(&next_bucket, device_next_bucket, sizeof(uint), cudaMemcpyDeviceToHost));
        if (next_bucket == INT_MAX) break; // WE ARE DONE
        // cudaCheckError(cudaMemset(device_bucket_deleted, false, N * sizeof(bool)));
        // cudaCheckError(cudaMemcpy(&next_bucket, device_next_bucket, sizeof(bool), cudaMemcpyDeviceToHost));
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
    // cudaCheckError(cudaFree(device_bucket_deleted));
    cudaCheckError(cudaFree(device_next_bucket));
    cudaCheckError(cudaFree(device_current_bucket_nonempty));
    delete[] bucket_num;
    delete[] bucket_num_next;
    // delete[] bucket_deleted;
}