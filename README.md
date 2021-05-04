# Warp-Centric CUDA Programming for SSSP Algorithms

**15-418 S21 Final Project**  
**Authors: Yiwen (Victor) Song, Xingran Du**



## Running the project on a GHC machine

### Generating a random input graph 

Random graphs can be generated following the Barab ́asi–Albert (BA) model by running `tools/generate_graph.py`. However, to do this on a GHC machine, we need to use python virtual environments in order to install the necessary packages (there are permission restrictions). In the root project folder, do the following:
```
python3 -m venv env  
source env/bin/activate  
python3 -m pip install numpy scipy networkx matplotlib  
python3 tools/generate_graph.py -N 10000 -M 10
```
The last command generates a random graph with 10000 nodes and close to `10000 * 10 = 100000` edges (the exact number in this case is 99900); you can also specify the maximum weight of an edge via the `-W` option (defaults to 20). Running this overwrites `src/input_graph.h`.

### Running an algorithm

```
make clean; make  
./sssp -a <dijkstra/bellman-ford/delta-stepping> -v <seq/base/warp>
```
Add the `-c` flag if you want to verify correctness against sequential Dijkstra's.  
You should be able to see the runtime and bandwidth of the execution of the algorithm on the input graph in input_graph.h

