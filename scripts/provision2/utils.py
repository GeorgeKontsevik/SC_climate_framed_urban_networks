import pandas as pd
import numpy as np

def calculate_provision_metrics(scores, demands, capacities):
    """Calculate provision metrics from 2SFCA results"""
    result = pd.DataFrame({
        'provision': scores,
        'demand_without': demands,
    })
    
    # Add capacity info if available
    if len(capacities) == len(scores):
        result['capacity_left'] = capacities
    else:
        result['capacity_left'] = 0
        
    return result