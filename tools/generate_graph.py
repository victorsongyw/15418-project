import argparse
import numpy as np
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
import copy


def plot_distribution(spW):
    row = spW.indptr
    prev_row = copy.deepcopy(row)[:-1]
    prev_row = np.insert(prev_row, 0, 0)
    row_length = row - prev_row
    
    '''
    plt.figure()
    plt.hist(row_length)
    plt.xlabel('degree')
    plt.ylabel('number of nodes')
    plt.ylim(0,100)
    plt.savefig("degree_distribution.png")
    '''

    plt.figure()
    node_id = np.arange(len(row_length))
    plt.plot(node_id, row_length, 'b.', alpha=0.5)
    #plt.ylim(0,500)
    plt.savefig("degree_vs_nodeid.png")


def main(args):
    n = args.N
    m = args.M
    max_weight = args.W

    G = nx.barabasi_albert_graph(n, m)  # m < n, m is num edges for each new node

    # create matrix
    spW = nx.attr_sparse_matrix(G)[0]
    spW = spW.tocsr()
    print(spW.todense())
    rng = np.random.default_rng()
    weights = rng.integers(1, max_weight, size=spW.nnz)

    plot_distribution(spW)

    fp = open(args.dir + '/input_graph.h', 'w')
    content = '#ifndef __input_graph_h_\n#define __input_graph_h_\n\n'
    content += '#include <stdint.h>\n\n'

    content += '#define NUM_NODES %d\n' % n
    content += '#define NUM_EDGES %d\n\n' % spW.nnz

    content += 'uint NODES[%d] = {\n' % (n + 1)
    for i in range(n + 1):
        content += str(spW.indptr[i])
        if (i != n):
            content += ', '
    content += '};\n\n'

    content += 'uint EDGES[%d] = {\n' % spW.nnz
    for i in range(spW.nnz):
        content += str(spW.indices[i])
        if (i != spW.nnz-1):
            content += ', '
    content += '};\n\n'

    # weights
    content += 'uint WEIGHTS[%d] = {\n' % spW.nnz
    for i in range(spW.nnz):
        content += str(weights[i])
        if (i != spW.nnz-1):
            content += ', '
    content += '};\n\n'

    content += '#endif\n'

    fp.write(content)
    fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", type=int, required=True, help='number of nodes ')
    parser.add_argument("-M", type=int, required=True, help='number of edges to attach from each new node ')
    parser.add_argument("-W", type=int, default=10, help='max weight ')
    parser.add_argument('-D', '--dir', type=str, default='dijkstra', #required=True,
        help='Directory to store file in ')
    
    args = parser.parse_args()
    main(args)