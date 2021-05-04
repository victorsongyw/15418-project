#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <getopt.h>
#include <string>

#include "CycleTimer.h"
#include "input_graph.h"

uint N = NUM_NODES; // number of nodes in the graph
uint M = NUM_EDGES; // number of directed edges in the graph
uint *nodes; // start index of edges from nth node
uint *edges; // destination node of edges
uint *weights; // weight of the edge in the corresponding index of edges
uint *dists; // distances from the source node

extern void dijkstra_seq(uint *dists);
extern void bellman_ford_seq(uint *dists);
extern void delta_stepping_seq(uint *dists);

void dijkstra_cuda(bool use_warp);
void bellman_ford_cuda(bool use_warp);
void delta_stepping_cuda(bool use_warp);

enum Algorithm { D, BF, DS, ALG_NIL };
enum Version { SEQ, BASE, WARP, VER_NIL };

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


// Verify correctness of output against sequential Dijkstra's
void verifyCorrectness() {
    uint *ref_dists = new uint[N];
    dijkstra_seq(ref_dists);
    for (uint i = 0; i < N; i++) {
        if (dists[i] != ref_dists[i]) {
            printf("Solution incorrect!\n");
            printf("ref_dists:\n");
            for (uint j = 0; j < N; j++)
            {
                if (ref_dists[j] != dists[j]) {
                    printf("%d: %d (ref) || %d\n", j, ref_dists[j], dists[j]);
                }
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
    printf("Inputs should be defined in src/input_graph.h\n");
    printf("Nodes should range from 0 to N-1 and the source node is node 0\n");
    printf("Edge weights should be non-negative and should not overflow\n");
    printf("Program Options:\n");
    printf("  -a  --algorithm <dijkstra/bellman-ford/delta-stepping>  Select SSSP algorithm\n");
    printf("  -v  --version <seq/base/warp>                           Select version to run: sequential or baseline CUDA or warp-centric CUDA\n");
    printf("  -c  --check                                             Set this to verify correctness of output against sequential Dijkstra's\n");
    printf("  -?  --help                                              This message\n");
}

int main(int argc, char** argv)
{
    // parse commandline options ////////////////////////////////////////////
    int opt;
    static struct option long_options[] = {
        {"algorithm", 1, 0, 'a'},
        {"version",   1, 0, 'v'},
        {"check",     0, 0, 'c'},
        {"help",      0, 0, '?'},
        {0, 0, 0, 0}
    };

    Algorithm alg = ALG_NIL; 
    Version ver = VER_NIL;
    bool check_correctness = false;

    while ((opt = getopt_long(argc, argv, "a:v:c?", long_options, NULL)) != EOF) {
        switch (opt) {
        case 'a':
            if (std::string(optarg).compare("dijkstra") == 0) {
                alg = D;
            } else if (std::string(optarg).compare("bellman-ford") == 0) {
                alg = BF;
            } else if (std::string(optarg).compare("delta-stepping") == 0) {
                alg = DS;
            } else {
                fprintf(stderr, "Invalid argument to -a option\n");
                usage(argv[0]);
                return 1;
            }
            break;
        case 'v':
            if (std::string(optarg).compare("seq") == 0) {
                ver = SEQ;
            } else if (std::string(optarg).compare("base") == 0) {
                ver = BASE;
            } else if (std::string(optarg).compare("warp") == 0) {
                ver = WARP;
            } else {
                fprintf(stderr, "Invalid argument to -a option\n");
                usage(argv[0]);
                return 1;
            }
            break;
        case 'c':
            check_correctness = true;
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    if (alg == ALG_NIL || ver == VER_NIL) {
        fprintf(stderr, "Missing arguments! -a and -v are required.\n");
        usage(argv[0]);
        return 1;
    }
    // end parsing of commandline options //////////////////////////////////////

    nodes = NODES;
    edges = EDGES;
    weights = WEIGHTS;
    
    dists = new uint[N]; // will contain distances from the start node
    for (uint i = 0; i < N; i++) {
        dists[i] = INT_MAX;
    }
    dists[0] = 0;

    printf("Init done\n");

    switch (alg) {
    case D:
        if (ver == SEQ) {
            dijkstra_seq(dists);
        } else {
            dijkstra_cuda(ver == WARP);
        }
        break;
    case BF:
        if (ver == SEQ) {
            bellman_ford_seq(dists);
        } else {
            bellman_ford_cuda(ver == WARP);
        }
        break;
    case DS:
        if (ver == SEQ) {
            delta_stepping_seq(dists);
        } else {
            delta_stepping_cuda(ver == WARP);
        }
        break;
    default:
        return 1;
    }

    if (check_correctness)
        verifyCorrectness();

    delete[] dists;
    return 0;
}