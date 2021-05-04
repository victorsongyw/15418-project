#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <queue>
using namespace std;

#include "CycleTimer.h"

extern float toBW(int bytes, float sec);
extern uint N, M;
extern uint *nodes, *edges, *weights;

// stores (dist, node) that are yet to be visited
typedef pair<uint, uint> uint_pair;

// Sequential Dijkstra's Algorithm using priority_queue of STL
// Adopted from https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-using-priority_queue-stl/
void dijkstra_seq(uint *dists) {
    double startTime = CycleTimer::currentSeconds();

    bool finalized[N]; // finalized[i] will be true if node i is finalized
    for (uint i = 0; i < N; i++) 
        finalized[i] = false;
    
    priority_queue< uint_pair, vector<uint_pair>, greater<uint_pair> > pq;
    pq.push(make_pair(0, 0));

    while (!pq.empty())
    {
        // Find the minimum distance node from the set of unfinalized nodes.
        // u is 0 in the first iteration.
        uint u = pq.top().second;
        pq.pop();

        if (finalized[u]) continue;

        finalized[u] = true;
        // Update dist value of neighbors of the picked node
        for (uint i = nodes[u]; i < nodes[u + 1]; i++) 
        {
            uint v = edges[i];
            if (!finalized[v] && dists[u] + weights[i] < dists[v])
            {
                dists[v] = dists[u] + weights[i];
                pq.push(make_pair(dists[v], v));
            }
        }
    }
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    printf("Sequential Dijkstra's: %.3f ms\n", 1000.f * overallDuration);
}