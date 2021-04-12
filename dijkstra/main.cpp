#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
using namespace std;

int N = -1; // number of nodes in the graph
int M = -1; // number of directed edges in the graph
int *nodes; // start index of edges from nth node
int *edges; // destination node of edges

void baseline_Dijkstra(int* nodes, int *edges, int *dists);
void warp_Dijkstra(int* nodes, int *edges, int *dists);

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


// void verifyCorrectness(int* nodes, int *edges, int *dists) {
//     int *correct_dists = new int[N];
//     dijkstra_ref(nodes, edges, correct_dists);
//     for 
// }


void usage(const char* progname) {
    printf("Usage: %s [options] filename\n", progname);
    printf("Valid files contain:\n");
    printf("first line: N, M\n");
    printf("second line: start index of edges from nth node (N+1 numbers)\n");
    printf("third line: destination node of edges (M lines)\n");
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
        case 'n':
            N = atoi(optarg);
            break;
        case '?':
        default:
            usage(argv[0]);
            return 1;
        }
    }
    // end parsing of commandline options //////////////////////////////////////

    // read and parse input file ////////////////////////////////////////////
    printf("start\n");
    if (optind + 1 > argc) {
        fprintf(stderr, "Error: missing input file name\n");
        usage(argv[0]);
        return 1;
    }
    ifstream inputFile(argv[optind]);
    if (!inputFile) {
        fprintf(stderr, "Error: failed to open input file %s\n", argv[optind]);
        usage(argv[0]);
        return 1;
    }
    printf("opened file\n");

    int lineNum = 0;
    bool success = true;
    string line;
    while (getline(inputFile, line)) {
        printf("reading line %d\n", lineNum);
        lineNum++;
        istringstream iss(line);
        if (lineNum == 1) {
            iss >> N >> M;
            if (N <= 0 || M <= 0) {
                success = false;
                goto readDone;
            }
            nodes = new int[N+1]; 
            edges = new int[M]; 
            for (int i = 0; i < N+1; i++) {
                nodes[i] = -1;
            }
            for (int i = 0; i < M; i++) {
                edges[i] = -1;
            }
        }
        else if (lineNum == 2) {
            for (int i = 0; i < N+1; i++) {
                iss >> nodes[i];
                if ((i == 0 && nodes[0] != 0) ||
                    (i == N && nodes[N] != M) ||
                    (i > 0 && nodes[i] <= nodes[i-1]))
                {
                    success = false;
                    goto readDone;
                }
            }

        }
        else if (lineNum == 3) {
            for (int i = 0; i < M; i++) {
                iss >> edges[i];
                if (edges[i] < 0 || edges[i] >= N) {
                    success = false;
                    goto readDone;
                }
            }
        }
    }
    readDone:
    inputFile.close();
    if (!success || lineNum != 3) {
        delete [] nodes;
        delete [] edges;
        fprintf(stderr, "Error: illegal input file content\n");
        return 1;
    }
    // end reading and parsing input file ////////////////////////////////////////////

    // printf("%d, %d\n", N, M);
    // for (int i = 0; i < N + 1; i++) {
    //     printf("%d ", nodes[i]);
    // }
    // printf("\n");
    // for (int i = 0; i < M; i++) {
    //     printf("%d ", edges[i]);
    // }
    // printf("\n");
    
    int *dists = new int[N]; // will contain distances from the start node
    for (int i = 0; i < N; i++) {
        dists[i] = -1;
    }

    printf("init done\n");

    // printCudaInfo();
    // baseline_Dijkstra(nodes, edges, dists);

    // if (true)
    //     verifyCorrectness(nodes, edges, dists);

    delete [] nodes;
    delete [] edges;
    delete [] dists;

    return 0;
}



