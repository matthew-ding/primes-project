import random
from networkx import *
import matplotlib.pyplot as plt


def get_adj_list():
    byzantine_size = 10  # number of byzantine nodes
    honest_size = 10  # number of honest nodes
    total_size = byzantine_size + honest_size
    corruptList = []  # contains nodes that have majority byzantine neighbors

    byzantine_output = []

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

    for i in range(byzantine_size + honest_size):
        adjList[i] = [n for n in final_graph.neighbors(i)]

        if final_graph.nodes[i]["byzantine"]:
            byzantine_output.append(i)

    return adjList, byzantine_output, total_size


def get_relay_graph(diameter):
    byzantine_size = diameter  # number of byzantine nodes
    honest_size = byzantine_size + 1  # number of honest nodes
    total_size = byzantine_size + honest_size

    byzantine_output = []
    for i in range(byzantine_size):
        byzantine_output.append(i)

    # graph generation
    G = Graph()

    # coloring graph
    colorList = []
    for i in range(byzantine_size):
        colorList.append('r')

    for i in range(honest_size):
        colorList.append('b')

    # adding nodes
    for i in range(byzantine_size):
        G.add_node(i, type='byzantine')

    for i in range(honest_size):
        G.add_node(byzantine_size+i, type='honest')

    # adding edges
    for i in range(byzantine_size):
        G.add_edge(i, byzantine_size)

    for i in range(byzantine_size):
        G.add_edge(byzantine_size+i, byzantine_size+i+1)

    # drawing graph
    draw_networkx(G, node_color=colorList)
    plt.savefig("relay_graph.png")
    plt.clf()

    # adjacency list generation
    adjList = {}

    for i in range(byzantine_size + honest_size):
        adjList[i] = [n for n in G.neighbors(i)]

    target = byzantine_size

    return adjList, byzantine_output, total_size, target
