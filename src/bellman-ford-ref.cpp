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

extern float toBW(int bytes, float sec);
extern uint N, M;
extern uint *nodes, *edges, *weights, *dists;

void BF_ref(uint *ref_dists)
{
    for (uint i = 0; i < N; i++)
    {
        ref_dists[i] = INT_MAX;
    }
    ref_dists[0] = 0;

    double startTime = CycleTimer::currentSeconds();

    // Relax all edges N-1 times
    for (uint count = 0; count < N - 1; count++)
    {

        for (uint v = 0; v < N; v++)
        {
            for (uint i = nodes[v]; i < nodes[v + 1]; i++)
            {
                uint u = edges[i];
                // updating an edge from v to u
                uint new_dist = ref_dists[v] + weights[i];
                if (new_dist < ref_dists[u])
                {
                    ref_dists[u] = new_dist;
                }
            }
        }
    }
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    int totalBytes = sizeof(uint) * (N + M) * 2; // TODO: UPDATE LATER
    printf("Sequential Ref - BF: %.3f ms\t\t[%.3f GB/s]\n", 1000.f * overallDuration, toBW(totalBytes, overallDuration));
}