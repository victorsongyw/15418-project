#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include "CycleTimer.h"

extern float toBW(int bytes, float sec);
extern uint N, M;
extern uint *nodes, *edges, *weights, *dists;

void dijkstra_seq(uint *dists) {
    //TODO: implement
    bool finalized[N]; // finalized[i] will be true if node i is finalized
    for (uint i = 0; i < N; i++) 
    {
        dists[i] = INT_MAX;
        finalized[i] = false;
    }
    dists[0] = 0;

    double startTime = CycleTimer::currentSeconds();

    // Find shortest path for all vertices
    for (uint count = 0; count < N-1; count++) 
    {
        // Find the minimum distance node from the set of unfinalized nodes.
        // u is 0 in the first iteration.
        uint min_dist = INT_MAX;
        uint u = 0;
        for (uint v = 0; v < N; v++) 
        {
            if (!finalized[v] && dists[v] <= min_dist) 
            {
                min_dist = dists[v];
                u = v;
            }
        }

        finalized[u] = true;

        // Update dist value of neighbors of the picked node
        for (uint i = nodes[u]; i < nodes[u+1]; i++) 
        {
            uint v = edges[i];
            if (!finalized[v] && dists[u] + weights[i] < dists[v])
                dists[v] = dists[u] + weights[i];
        }
    }
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    printf("Sequential Dijkstra's: %.3f ms\n", 1000.f * overallDuration);
}