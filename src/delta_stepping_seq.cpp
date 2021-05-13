#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <map>
#include <set>
using namespace std;

#include "CycleTimer.h"

extern uint DELTA;

extern uint N, M;
extern uint *nodes, *edges, *weights;

void update_buckets(uint u, map< uint, set<uint> > &buckets, uint old_bucket, uint new_bucket) 
{
    buckets[old_bucket].erase(u);
    if (buckets.find(new_bucket) != buckets.end())
        buckets[new_bucket].insert(u);
    else
    {
        set<uint> bucket;
        bucket.insert(u);
        buckets[new_bucket] = bucket;
    }
}

// Adopted from the delta-stepping paper by U. Meyers, P. Sanders
// Available at https://www.cs.utexas.edu/~pingali/CS395T/2012sp/papers/delta-stepping.pdf
void delta_stepping_seq(uint *dists) 
{
    double startTime = CycleTimer::currentSeconds();

    map< uint, set<uint> > buckets; // map bucket to set of remaining nodes

    set<uint> zero_bucket;
    zero_bucket.insert(0);
    buckets[0] = zero_bucket;

    set<uint> max_bucket;
    for (uint i = 1; i < N; i++)
        max_bucket.insert(i);
    buckets[INT_MAX / DELTA] = max_bucket;

    uint curr_bucket_num;
    while (buckets.size() > 0)
    {
        // process next bucket
        if ((buckets.begin()->second).size() == 0)
        {   // empty bucket
            buckets.erase(buckets.begin()->first);
            continue;
        }
        curr_bucket_num = buckets.begin()->first;
        set<uint> curr_bucket = buckets[curr_bucket_num];
        map<uint, uint> heavy_edges; // map to source node v from a heavy edge's idx
        while (curr_bucket.size() > 0) 
        {
            uint v = *curr_bucket.begin();
            curr_bucket.erase(v);
            for (uint i = nodes[v]; i < nodes[v + 1]; i++)
            {   
                uint u = edges[i];
                uint u_weight = weights[i];
                if (u_weight > DELTA)
                    heavy_edges[i] = v; // save heavy edges for later
                else 
                {   // updating a light edge from v to u
                    uint new_dist = dists[v] + u_weight;
                    if (new_dist < dists[u])
                    {
                        uint old_bucket = dists[u] / DELTA;
                        uint new_bucket = new_dist / DELTA;
                        dists[u] = new_dist;
                        if (new_bucket == curr_bucket_num)
                        {
                            buckets[old_bucket].erase(u);
                            curr_bucket.insert(u);
                        }
                        else if (new_bucket < old_bucket)
                            update_buckets(u, buckets, old_bucket, new_bucket);
                    }
                }
            }
        }
        // relax previously deferred heavy edges
        for (auto it = heavy_edges.begin(); it != heavy_edges.end(); it++)
        {  
            uint v = it->second;
            uint idx = it->first;
            uint u = edges[idx];
            uint new_dist = dists[v] + weights[idx];
            if (new_dist < dists[u])
            {
                uint old_bucket = dists[u] / DELTA;
                uint new_bucket = new_dist / DELTA;
                dists[u] = new_dist;
                if (new_bucket < old_bucket)
                    update_buckets(u, buckets, old_bucket, new_bucket);
            }
        }

        // delete bucket
        buckets.erase(curr_bucket_num);
    }

    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    printf("Sequential delta-stepping: %.3f ms\n", 1000.f * overallDuration);
}