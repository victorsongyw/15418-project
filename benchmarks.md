### Graphs used in the paper: 
#### Generated graphs
Average degree = 12
- RMAT (irregular), n = 4e6

Us: barabasi-albert model with n nodes, m = 12

Reasonable n ~= 1e5

- Random (regular), n = 4e6
Random is a uniformly distributed graph instance created
by randomly connecting m pairs of nodes out of a total n nodes

Us: gnm_random_graph with n nodes, m = 12n edges

#### [Real-world graphs](http://snap.stanford.edu/data/index.html)

- LiveJournal

n = 4,308,451, m = 68,993,773

- Patents

n = 1,765,311, m = 10,564,104

For us, we could use 
- ego-Facebook: 4,039 & 88,234
- ego-Twitter: 81,306 & 1,768,149

same category as `LiveJournal` (Social Network)

- cit-HepPh: 34,546	& 421,578

same category as `Patents` (Citation Network)


### <span style="color:red"> _Note:_ </span>

`input_graph.h` represent each edge twice, so the NUM_EDGES in the file is twice the actual edge count.

