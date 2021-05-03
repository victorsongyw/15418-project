#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <getopt.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

#include "CycleTimer.h"
#include "input_graph.h"

uint N = NUM_NODES; // number of nodes in the graph
uint M = NUM_EDGES; // number of directed edges in the graph
uint *nodes; // start index of edges from nth node
uint *edges; // destination node of edges
uint *weights; // weight of the edge in the corresponding index of edges
uint *dists; // distances from the source node

void delta_stepping(bool use_warp);

// return GB/s
float toBW(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}

// For fun, just print out some stats on the machine
// void printCudaInfo() {
//     int deviceCount = 0;
//     cudaError_t err = cudaGetDeviceCount(&deviceCount);

//     printf("---------------------------------------------------------\n");
//     printf("Found %d CUDA devices\n", deviceCount);

//     for (int i=0; i<deviceCount; i++) {
//         cudaDeviceProp deviceProps;
//         cudaGetDeviceProperties(&deviceProps, i);
//         printf("Device %d: %s\n", i, deviceProps.name);
//         printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
//         printf("   Global mem: %.0f MB\n",
//                static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
//         printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
//     }
//     printf("---------------------------------------------------------\n");
// }

// Adopted from https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/
void dijkstra_ref(uint *ref_dists) {
    bool finalized[N]; // finalized[i] will be true if node i is finalized
    for (uint i = 0; i < N; i++) {
        ref_dists[i] = INT_MAX;
        finalized[i] = false;
    }
    ref_dists[0] = 0;

    double startTime = CycleTimer::currentSeconds();

    // Find shortest path for all vertices
    for (uint count = 0; count < N-1; count++) {
        // Find the minimum distance node from the set of unfinalized nodes.
        // u is 0 in the first iteration.
        uint min_dist = INT_MAX;
        uint u = 0;
        for (uint v = 0; v < N; v++) {
            if (!finalized[v] && ref_dists[v] <= min_dist) {
                min_dist = ref_dists[v];
                u = v;
            }
        }

        finalized[u] = true;

        // Update dist value of neighbors of the picked node
        for (uint i = nodes[u]; i < nodes[u+1]; i++) {
            uint v = edges[i];
            if (!finalized[v] && ref_dists[u] + weights[i] < ref_dists[v]) {
                ref_dists[v] = ref_dists[u] + weights[i];
            }
        }
    }
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    int totalBytes = sizeof(uint) * (N + M) * 2; // TODO: UPDATE LATER
    printf("Sequential Ref: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));
}


void verifyCorrectness() {
    uint *ref_dists = new uint[N];
    dijkstra_ref(ref_dists);
    for (uint i = 0; i < N; i++) {
        if (dists[i] != ref_dists[i]) {
            printf("Solution incorrect!\n");
            printf("ref_dists:\n");
            for (uint j = 0; j < N; j++)
            {
                if (ref_dists[j] != dists[j]) printf("--> ");
                printf("ref %d: %d || ", j, ref_dists[j]);
                printf("cuda %d: %d\n", j, dists[j]);
            }
            delete[] ref_dists;
            return;
        }
    }
    printf("Solution correct!\n");
    delete[] ref_dists;
}


void usage(const char* progname) {
    printf("Usage: %s [options]\n", progname);
    printf("Nodes should range from 0 to N-1 and the source node is node 0\n");
    printf("Edge weights should be non-negative and should not overflow\n");
    printf("Program Options:\n");
    printf("  -c  --check                  Check correctness of output\n");
    printf("  -a  --algorithm <base/warp>  Select renderer: ref or cuda\n");
    printf("  -?  --help                   This message\n");
}

int main(int argc, char** argv)
{
    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"check",     0, 0, 'c'},
        {"help",      0, 0, '?'},
        {"algorithm", 1, 0, 'a'},
        {0, 0, 0, 0}
    };

    while ((opt = getopt_long(argc, argv, "?ca:", long_options, NULL)) != EOF) {
        switch (opt) {
        // case 'n':
        //     N = atoi(optarg);
        //     break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////

    nodes = NODES;
    edges = EDGES;
    weights = WEIGHTS;

    // printf("N=%d, M=%d\n", N, M);
    // printf("nodes:   ");
    // for (int i = 0; i < N + 1; i++) {
    //     printf("%d ", nodes[i]);
    // }
    // printf("\nedges:   ");
    // for (int i = 0; i < M; i++) {
    //     printf("%d ", edges[i]);
    // }
    // printf("\nweights: ");
    // for (int i = 0; i < M; i++) {
    //     printf("%d ", weights[i]);
    // }
    // printf("\n");
    
    dists = new uint[N]; // will contain distances from the start node
    for (uint i = 0; i < N; i++) {
        dists[i] = INT_MAX;
    }
    dists[0] = 0;

    printf("init done\n");

    // printCudaInfo();
    delta_stepping(true);
    // delta_stepping(false);

    if (true)
        verifyCorrectness();

    delete[] dists;

    return 0;
}