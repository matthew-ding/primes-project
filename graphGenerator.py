import random
from networkx import *
import matplotlib.pyplot as plt


def get_adj_list():
    byzantine_size = 15  # number of byzantine nodes
    honest_size = 15  # number of honest nodes
    corruptList = [1, 2]  # contains nodes that have majority byzantine neighbors

    # graph generation
    # honest graph is connected
    while True:
        honest_subgraph = erdos_renyi_graph(honest_size, 0.5)
        if is_connected(honest_subgraph):
            break
    # byzantine graph is disconnected
    while True:
        byzantine_subgraph = erdos_renyi_graph(byzantine_size, 0.15)
        if not is_connected(byzantine_subgraph):
            break

    colorList = []  # used for final graph coloring
    # labeling nodes
    for i in range(honest_size):
        honest_subgraph.nodes[i]['byzantine'] = False
        colorList.append('b')

    for i in range(byzantine_size):
        byzantine_subgraph.nodes[i]['byzantine'] = True
        colorList.append('r')

    byzantine_subgraph = convert_node_labels_to_integers(byzantine_subgraph, first_label=honest_size
                                                         , ordering='default', label_attribute=None)

    # printing graphs
    draw_networkx(honest_subgraph)
    plt.savefig("honest.png")
    plt.clf()
    draw_networkx(byzantine_subgraph, node_color="r")
    plt.savefig("byzantine.png")
    plt.clf()

    final_graph = compose(honest_subgraph, byzantine_subgraph)

    for i in range(honest_size):
        max_byzantine = len(list(honest_subgraph.neighbors(i))) // 2

        if i in corruptList:
            byzantine_num = random.randrange(max_byzantine + 1, int(1.5 * max_byzantine + 1) + 1)
        else:
            byzantine_num = random.randrange(int(0.75 * max_byzantine), max_byzantine + 1)

        entire_byzantine_set = list(byzantine_subgraph.nodes)
        byzantine_set = random.sample(entire_byzantine_set, byzantine_num)

        for j in byzantine_set:
            final_graph.add_edge(i, j)

    draw_networkx(final_graph, node_color=colorList)
    plt.savefig("final_graph.png")

    adjList = {}

    for i in range(byzantine_size+honest_size):
        adjList[i] = [n for n in final_graph.neighbors(i)]

    return adjList
