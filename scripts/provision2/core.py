import numpy as np
import pandas as pd
import networkx as nx
from pysal.lib import weights
from access import Access

from .graph import create_flow_graph, add_flow_paths
from .utils import calculate_provision_metrics


def prepare_2sfca_data(G, service_type='hospital'):
    """Prepare data for 2SFCA calculation"""
    demands = np.array([data.get('population', 0)/1e3*120 for _, data in G.nodes(data=True)])
    capacities = np.array([data.get(f'capacity_{service_type}', 0) for _, data in G.nodes(data=True)])

    return demands, capacities

def calculate_graph_provision(G, service_radius,
                              service_type="hospital"):
    """Calculate service provision using Two-Stage FCA method"""
    
    demands, capacities = prepare_2sfca_data(G, service_type)
    nodes = [data.get('name') for _, data in G.nodes(data=True)]
    node_ids = [data.get('id') for _, data in G.nodes(data=True)]
    
    # Create cost DataFrame with proper format
    cost_data = []
    for i, source in zip(node_ids, nodes):
        for j, target in zip(node_ids, nodes):
            # if i != j:
            try:
                cost = nx.shortest_path_length(G, source, target, weight='weight')
                cost_data.append({
                    'origin': source,
                    'destination': target,
                    'cost': cost
                })
            except nx.NetworkXNoPath:
                continue

    cost_df = pd.DataFrame(cost_data)

    print(cost_df.loc[cost_df['origin'] == 'Бугрино'])

    # Create cost DataFrame with proper format
    cost_data = []
    for i, source in zip(node_ids, nodes):
        for j, target in zip(node_ids, nodes):
            # if i != j:
            try:
                cost = nx.shortest_path_length(G, source, target, weight='weight')
                cost = cost if cost < service_radius else np.inf
                cost_data.append({
                    'origin': i,
                    'destination': j,
                    'cost': cost,
                    'origin_name': source,
                    'destination_name': target,
                })
            except nx.NetworkXNoPath:
                continue

    cost_df = pd.DataFrame(cost_data)
    print(cost_df)

    # Create properly formatted demand and supply DataFrames
    demand_df = pd.DataFrame({
        'geoid': node_ids,
        'pop': demands  # Using 'pop' as demand value like in example
    })

    supply_df = pd.DataFrame({
        'geoid': node_ids,
        'supply': capacities
    })

    # Initialize Access object with proper format
    access_obj = Access(
        demand_df=demand_df,
        demand_index='geoid',
        demand_value='pop',
        supply_df=supply_df,
        supply_index='geoid',
        supply_value='supply',
        cost_df=cost_df,
        cost_origin='origin',
        cost_dest='destination',
        cost_name='cost',
        neighbor_cost_df=cost_df,  # Using same cost matrix for neighbors
        neighbor_cost_origin='origin',
        neighbor_cost_dest='destination',
        neighbor_cost_name='cost'
)


    # Get scores and assignment matrix from two_stage_fca
    result = access_obj.two_stage_fca(
        name='provision',
        cost='cost',
        max_cost=service_radius,
        normalize=False,
    )

    return result