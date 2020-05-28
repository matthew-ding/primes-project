from networkx import *

byzantine_size = 4
honest_size = 6
p = 0.5
honest_subgraph = erdos_renyi_graph(honest_size, p)
byzantine_subgraph = erdos_renyi_graph(byzantine_size, p)

for i in range(byzantine_size):
    byzantine_subgraph.nodes
    {"byzantine": True}

for i in range(honest_size):
    honest_subgraph.nodes
    {"byzantine": False}

convert_node_labels_to_integers(byzantine_subgraph, honest_size+1)

print(byzantine_subgraph.nodes)
print(honest_subgraph.nodes)

print(byzantine_subgraph.nodes.data())
print(honest_subgraph.nodes.data())


