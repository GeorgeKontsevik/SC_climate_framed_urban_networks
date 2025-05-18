import networkx as nx
import numpy as np

def create_flow_graph(G):
    """Create a new graph for flows while preserving node attributes"""
    G_result = nx.DiGraph()
    G_result.add_nodes_from(G.nodes(data=True))
    G_result.graph['crs'] = G.graph.get('crs')
    return G_result

def add_flow_paths(G, assignments, demands):
    """Add flow paths and assignments to graph"""
    for i, row in enumerate(assignments):
        for j, flow in enumerate(row):
            if flow > 0:
                # Find shortest path between nodes
                path = nx.shortest_path(G, i, j, weight='weight')
                
                # Add assignment edge
                G.add_edge(i, j, 
                          assignment=demands[i] * flow,
                          path=path,
                          is_service_flow=True)
                
                # Add physical path edges
                for u, v in zip(path[:-1], path[1:]):
                    if not G.has_edge(u, v):
                        G.add_edge(u, v, **G[u][v])
    return G