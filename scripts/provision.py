import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
import math

import scripts.model as model  # Assuming model is a module that contains the calculate_provision function

def calculate_base_demand(population, const_base_demand=120):
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

def graph_to_city_model(G, service_radius=300, const_base_demand=120):
    """Convert networkx graph to city_model format"""
    # Create blocks list (not dict) with proper structure
    blocks = []
    
    for node, data in G.nodes(data=True):
        pop = data.get('population', 0)
        health_capacity = data.get('capacity', 0)
        
        block = {
            'id': data.get('id', node),
            'geometry': data.get('geometry', None),
            'demand': calculate_base_demand(pop, const_base_demand),
            "population": pop,
            'capacities': {
                'hospital': health_capacity  # Direct capacity value for hospital service
            }
        }
        blocks.append(block)
    
    # # Convert graph edges to weights
    graph_dict = {}
    for u, v, data in G.edges(data=True):
        if u not in graph_dict:
            graph_dict[u] = {}
        if v not in graph_dict:
            graph_dict[v] = {}
            
        weight = data.get('weight', float('inf'))
        graph_dict[u][v] = {'weight': weight}
        graph_dict[v][u] = {'weight': weight}

    
    # Create city_model dictionary
    city_model = {
        'epsg': 32639,
        'blocks': blocks,
        'graph': graph_dict,
        'service_types': {
            'hospital': {
                'name': 'hospital',
                'accessibility': service_radius,
                'demand': const_base_demand,
            },
        },
    }
    
    return city_model


def calculate_graph_provision(G, timestep=0):
    """Calculate provision metrics for graph at given timestep"""
    # try:
    # Convert graph to city_model format
    city_model = graph_to_city_model(G)
    
    # Calculate provision using model function
    result = model.calculate_provision(city_model, "hospital", method="lp")
    
    # Add provision values to graph nodes
    nx.set_node_attributes(G, {
        node: {'provision': result.loc[node, 'provision']}
        for node in G.nodes()
    })
    
    return result
    # except Exception as e:
    #     print(f"Error calculating provision: {e}")
    #     return None