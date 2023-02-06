import random
import sys
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def bellman_ford(graph, start, end):
    distance = [sys.maxsize] * len(graph)
    distance[int(start)] = 0

    for _ in range(len(graph) - 1):
        for i in range(len(graph)):
            for j, w in graph[i]:
                if distance[i] != sys.maxsize and distance[j] > distance[i] + w:
                    distance[j] = distance[i] + w
                    # print(f"Updating distance of node {j} to {distance[j]}")

    for i in range(len(graph)):
        for j, w in graph[i]:
            if distance[i] != sys.maxsize and distance[j] > distance[i] + w:
                print("Graph contains negative weight cycle")
                return None
    return distance

def eliminate_cycles(graph, start, end):
    distance = bellman_ford(graph, start, end)
    if distance is None:
        cycle = []
        for i in range(len(graph)):
            for j, w in graph[i]:
                if distance[j] > distance[i] + w:
                    cycle = [(i, j, w)]
                    u = i
                    while u != j:
                        for k, x in graph[u]:
                            if distance[k] == distance[u] + x:
                                cycle.append((u, k, x))
                                u = k
                                break
        cycle_edge = min(cycle, key=lambda x: x[2])
        graph[cycle_edge[0]].remove(cycle_edge[1:])
        print(f"Removed edge {cycle_edge[0]} -> {cycle_edge[1]} (weight {cycle_edge[2]}) from the graph")
        return eliminate_cycles(graph, start, end)
    else:
        return distance[end],graph

if __name__ == '__main__':
    n = 10
    m = 30
    graph = [[] for _ in range(n)]
    for _ in range(m):
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        w = random.randint(1, 10)
        graph[u].append((v, w))
    print("Input graph:")
    G = nx.Graph()
    #print(graph)
    edges = [(i,graph[i][j][0]) for i in range(len(graph)) for j in range(len(graph[i]))]
    weights = [(i,graph[i][j][0],graph[i][j][1]) for i in range(len(graph)) for j in range(len(graph[i]))]
    G.add_edges_from(edges)
    print(G.edges)
    for tuple in weights:
        G[tuple[0]][tuple[1]]["weight"]=tuple[2]
    start = 0
    end = n-1
    # print(end)
    #shortest_path,graph = eliminate_cycles(graph, start, end)
    #print(graph)
    shortest_path = nx.shortest_path(G,source = 0 , target = 4,weight=weights)
    print(shortest_path)
    nx.draw(G,with_labels=True)
    plt.show()
    
    # Y = nx.Graph()
    # edges = []
    # edges = [(i,graph[i][j][0]) for i in range(len(graph)) for j in range(len(graph[i]))]
    # Y.add_edges_from(edges)
    # nx.draw(Y)
    # plt.show()
    

    print(f"Shortest path from node {start} to node {end}: {shortest_path}")
