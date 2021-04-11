#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <string>

int N = -1;
int M = -1;

void baseline_Dijkstra(int* nodes, int *edges, int *dists);
void warp_Dijkstra(int* nodes, int *edges, int *dists);

// return GB/s
float toBW(int bytes, float sec) {
  return static_cast<float>(bytes) / (1024. * 1024. * 1024.) / sec;
}


void usage(const char* progname) {
    printf("Usage: %s [options] filename\n", progname);
    printf("Valid files contain:\n");
    printf("first line: N, M\n");
    printf("second line: start index of edges from nth node (N+1 numbers)\n");
    printf("third line: destination node of edges (M lines)\n");
    printf("Program Options:\n");
    printf("  -b  --bench <START:END>    Benchmark mode, do not create display. Time frames [START,END)\n");
    printf("  -c  --check                Check correctness of output\n");
    printf("  -a  --algorithm <baseline/warp>  Select renderer: ref or cuda\n");
    printf("  -?  --help                 This message\n");
    // printf("Usage: %s [options]\n", progname);
    // printf("Program Options:\n");
    // printf("  -n  --arraysize <INT>  Number of elements in arrays\n");
    // printf("  -?  --help             This message\n");
}

// For fun, just print out some stats on the machine
void printCudaInfo() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}


void verifyCorrectness(int* nodes, int *edges, int *dists) {
    int *correct_dists = new int[N];
    dijkstra_ref(nodes, edges, correct_dists);
    for 
}




int main(int argc, char** argv)
{

    // int N = 20 * 1000 * 1000;

    // parse commandline options ////////////////////////////////////////////
    // int opt;
    // static struct option long_options[] = {
    //     {"arraysize",  1, 0, 'n'},
    //     {"help",       0, 0, '?'},
    //     {0 ,0, 0, 0}
    // };

    // while ((opt = getopt_long(argc, argv, "?n:", long_options, NULL)) != EOF) {

    //     switch (opt) {
    //     case 'n':
    //         N = atoi(optarg);
    //         break;
    //     case '?':
    //     default:
    //         usage(argv[0]);
    //         return 1;
    //     }
    // }
    // end parsing of commandline options //////////////////////////////////////

    N = 1000;
    int *nodes = new int[N+1]; // start index of edges from nth node
    int *edges = new int[M]; // destination node of edges
    int *dists = new int[N]; // will contain distances from the start node
    // TODO: read from input file

    for (int i=0; i<N; i++) {
        dists[i] = -1;
    }

    printCudaInfo();
    baseline_Dijkstra(nodes, edges, dists);

    if (true)
        verifyCorrectness(nodes, edges, dists);

    delete [] nodes;
    delete [] edges;
    delete [] dists;



    return 0;
}



