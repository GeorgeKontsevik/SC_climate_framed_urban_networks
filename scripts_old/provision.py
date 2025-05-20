import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
import math

import scripts_old.model as model  # Assuming model is a module that contains the calculate_provision function

def calculate_base_demand(population, const_base_demand):
    """Calculate base demand from population with safety checks"""
    if population is None:
        return 0
    
    if isinstance(population, str):
        try:
            population = float(population)
        except (ValueError, TypeError):
            return 0
            
    try:
        return math.ceil((population / 1000) * const_base_demand)
    except (ValueError, TypeError):
        return 0

def create_adjacency_matrix(G):
    """
    Create full adjacency matrix from graph with shortest path distances between all nodes.
    
    Args:
        G: NetworkX graph
        
    Returns:
        pd.DataFrame: Full adjacency matrix with shortest path distances
    """
    # Get sorted list of nodes to ensure consistent ordering
    nodes = sorted(G.nodes())
    
    # Create empty DataFrame with inf values
    matrix = pd.DataFrame(
        float('inf'), 
        index=nodes, 
        columns=nodes
    )
    
    # Fill diagonal with zeros
    np.fill_diagonal(matrix.values, 0)
    
    # Calculate shortest paths between all pairs of nodes
    for source in nodes:
        try:
            # Get shortest path lengths from source to all other nodes
            path_lengths = nx.single_source_dijkstra_path_length(G, source, weight='weight')
            
            # Fill the matrix with these distances
            for target, distance in path_lengths.items():
                matrix.loc[source, target] = distance
                
        except nx.NetworkXNoPath:
            continue  # Skip if no path exists
            
    return matrix

def graph_to_city_model(G, service_radius, const_base_demand, service_name="hospital", adj_matrix=None):
    """
    Convert networkx graph or adjacency matrix to city_model format
    
    Args:
        G: NetworkX graph (for node attributes)
        service_radius: Maximum allowed service distance
        const_base_demand: Base demand per 1000 population
        service_name: Type of service to analyze
        adj_matrix: Optional pre-computed adjacency matrix
    """
    # Create blocks list with proper structure
    blocks = []
    
    for node, data in G.nodes(data=True):
        pop = data.get('population', 0)
        service_capacity = data.get(f'capacity_{service_name}', 0)
        
        block = {
            'id': data.get('id', node),
            'name': data.get('name', node),
            'geometry': data.get('geometry', None),
            'demand': calculate_base_demand(pop, const_base_demand),
            "population": pop,
            f'capacity_{service_name}': service_capacity,
            "demand_within": 0,
            "demand_without": 0,
            "capacity_left": 0,
        }
        blocks.append(block)

    # Use provided adjacency matrix or create from graph
    if adj_matrix is None:
        adj_matrix = create_adjacency_matrix(G)
    
    # Convert adjacency matrix to graph dict format
    graph_dict = {}
    for i in adj_matrix.index:
        graph_dict[i] = {}
        for j in adj_matrix.columns:
            if i != j and adj_matrix.loc[i, j] != float('inf'):
                graph_dict[i][j] = {'weight': adj_matrix.loc[i, j]}
    # print(graph_dict)
    city_model = {
        'epsg': G.graph.get('crs', None),
        'blocks': blocks,
        'graph': graph_dict,
        'service_types': {
            service_name: {
                'accessibility': service_radius,
                'demand': const_base_demand,
            },
        },
    }
    
    return city_model


def calculate_graph_provision(G, service_radius, const_base_demand, service_name="hospital", return_assignment=False):
    """
    Calculate provision metrics for graph and create assignment edges
    
    Args:
        G: NetworkX graph
        service_radius: Maximum allowed service distance
        const_base_demand: Base demand per 1000 population
        service_name: Type of service to analyze
        return_assignment: Whether to return assignment details
        
    Returns:
        tuple: (G_with_assignments, provision_result, assignment_matrix)
            - G_with_assignments: Graph with assignment edges added
            - provision_result: DataFrame with provision metrics
            - assignment_matrix: DataFrame showing demand distribution
    """
    # Convert graph to city_model format
    # Create adjacency matrix first
    adj_matrix = create_adjacency_matrix(G)
    
    # Convert to city_model using the adjacency matrix
    city_model = graph_to_city_model(
        G, 
        service_radius, 
        const_base_demand, 
        service_name,
        adj_matrix=adj_matrix
    )

    # Calculate provision using model function
    result, assignments = model.calculate_provision(
        city_model=city_model, 
        service_type=service_name, 
        method="lp",
    )

    # Create a fresh directed graph for service flows
    G_with_assignments = nx.DiGraph()
    
    # First add all nodes with their attributes
    for node, data in G.nodes(data=True):
        G_with_assignments.add_node(node, **data)
    
    # Add base network edges (undirected)
    for u, v, data in G.edges(data=True):
        G_with_assignments.add_edge(u, v, **data)
        G_with_assignments.add_edge(v, u, **data)
    
    # Create assignment matrix
    nodes = list(G_with_assignments.nodes())
    assignment_matrix = pd.DataFrame(
        assignments,
        index=nodes,
        columns=nodes
    )

    # print(assignment_matrix)
    
    # Add service flow assignments
    for i in assignment_matrix.index:
        for j in assignment_matrix.columns:
            flow = assignment_matrix.loc[i, j]
            if flow > 0:  # Only process non-zero flows
                # Find shortest path for this service flow
                try:
                    path = nx.shortest_path(G, i, j, weight='weight')
                    
                    # Add direct service flow edge
                    total_weight = sum(G[path[k]][path[k+1]]['weight'] for k in range(len(path)-1))
                    G_with_assignments.add_edge(i, j, 
                                             weight=total_weight,
                                             assignment=flow,
                                             path=path,
                                             is_service_flow=True)
                    
                    # Mark edges along physical path as service routes
                    for k in range(len(path)-1):
                        u, v = path[k], path[k+1]
                        G_with_assignments[u][v]['is_service_route'] = True
                        G_with_assignments[u][v]['service_flows'] = G_with_assignments[u][v].get('service_flows', 0) + flow
                        
                except nx.NetworkXNoPath:
                    # print(f"Warning: No path between service node {i} and demand node {j}")
                    continue
    # print(result)
    # Add provision values to graph nodes
    nx.set_node_attributes(G_with_assignments, {
        node: {
            'provision': result.loc[data['name'], 'provision'],
            'demand_within': result.loc[data['name'], 'demand_within'],
            'demand_without': result.loc[data['name'], 'demand_without'],
            'capacity_left': result.loc[data['name'], 'capacity_left']
        }
        for node, data in G_with_assignments.nodes(data=True)
    })
    
    if return_assignment:
        return G_with_assignments, result, assignment_matrix
    
    return G_with_assignments, result