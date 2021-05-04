#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include "CycleTimer.h"

extern uint N, M;
extern uint *nodes, *edges, *weights;

void bellman_ford_seq(uint *dists)
{
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
                uint new_dist = dists[v] + weights[i];
                if (new_dist < dists[u])
                    dists[u] = new_dist;
            }
        }
    }
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    printf("Sequential Bellman-Ford: %.3f ms\n", 1000.f * overallDuration);
}