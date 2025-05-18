import math
from itertools import product
import random
import warnings


import numpy as np
from tqdm import tqdm, trange
# from lscp import LSCP
import pulp
import geopandas as gpd
import pandas as pd
import networkx as nx
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD


warnings.filterwarnings("ignore", category=FutureWarning)


def create_blocks(
    not_yet_blocks_gdf: gpd.GeoDataFrame, const_demand: int, epsg: int
) -> list[dict]:
    """
    Converts a GeoDataFrame of blocks into a list of block dictionaries.
    """
    blocks = []
    for _, row in not_yet_blocks_gdf.iterrows():
        pops = int(x) if (x := row.get("population")) else const_demand
        # demand = math.ceil(pops / 1000) if pops > const_demand else const_demand
        # capacity = row.get("capacity", 0)
        block = {
            "id": row["id"],
            "name": row["name"],
            "geometry": row["geometry"],
            "population": pops,
            # "demand": demand,
            # "capacities": capacity,
            "epsg": epsg,
        }
        blocks.append(block)
    
    # Create a GeoDataFrame for blocks
    blocks_gdf = gpd.GeoDataFrame(blocks).set_geometry("geometry").set_crs(epsg=epsg)

    return blocks_gdf


# def create_graph(matrix: pd.DataFrame) -> nx.DiGraph:
#     """
#     Creates a directed graph from a distance matrix and blocks.
#     """
#     #blocks: list[dict]

#     graph = nx.DiGraph()
#     # block_by_id = {block["id"]: block for block in blocks}

#     for i in matrix.index:
#         for j in matrix.columns:
#             # Используем id блоков как узлы графа
#             graph.add_edge(
#                 i,  # id первого блока
#                 j,  # id второго блока
#                 weight=matrix.loc[i, j],  # вес ребра
#             )

#     return graph


def update_blocks_with_services(
    blocks_gdf: gpd.GeoDataFrame,
    services_gdf: gpd.GeoDataFrame,
    service_type: str,
) -> gpd.GeoDataFrame:
    """
    Updates blocks GeoDataFrame with service capacities.
    """
    # Check CRS match
    assert blocks_gdf.crs.to_epsg() == services_gdf.crs.to_epsg(), "CRS mismatch"
    if 'capacity' in services_gdf.columns:
        services_gdf.rename(columns={"capacity": f"capacity_{service_type}"}, inplace=True)

    # First join to identify which services are in which blocks
    joined = gpd.sjoin(services_gdf, blocks_gdf, how="inner", predicate="intersects")
    
    
    # Then dissolve to aggregate capacities within each block
    aggregated = joined.dissolve(
        by="index_right",
        aggfunc={f"capacity_{service_type}": "sum"}
    )

    # Create capacity column name 
    capacity_col = f"capacity_{service_type}"
    
    # Update blocks with capacities
    blocks_gdf[capacity_col] = 0  # Initialize with zeros
    blocks_gdf.loc[aggregated.index, capacity_col] = aggregated[f"capacity_{service_type}"]

    return blocks_gdf

from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
from itertools import product
import pandas as pd
import geopandas as gpd

def lp_provision(
    city_model: dict, gdf: gpd.GeoDataFrame, service_type: str
) -> gpd.GeoDataFrame:
    """
    Linear programming assessment method for resource allocation using block names.
    """
    accessibility = city_model["service_types"][service_type]["accessibility"]
    gdf = gdf.copy()

    delta = gdf["demand"].sum() - gdf[f"capacity_{service_type}"].sum()
    fictive_block_id = "FICTIVE_BLOCK"

    if delta > 0:
        # Добавляем фиктивную строку с capacity
        gdf.loc[fictive_block_id, f"capacity_{service_type}"] = delta
        gdf.loc[fictive_block_id, "demand"] = 0
    elif delta < 0:
        # Добавляем фиктивную строку с demand
        gdf.loc[fictive_block_id, "demand"] = -delta
        gdf.loc[fictive_block_id, f"capacity_{service_type}"] = 0

    gdf["capacity_left"] = gdf[f"capacity_{service_type}"]

    def _get_weight(name1, name2):
        if name1 == name2:
            return 0
        if fictive_block_id in [name1, name2]:
            return 0
        try:
            return int(city_model['graph'][name1][name2]['weight'])
        except Exception as e:
            try:
                return int(city_model['graph'][name2][name1]['weight'])
            except Exception as e:
                # print(f"Error getting weight between {name1} and {name2}: {e}, {city_model['graph']}")
                return int(1e3)

    demand = gdf[gdf["demand"] > 0]
    capacity = gdf[gdf[f"capacity_{service_type}"] > 0]

    prob = LpProblem("Transportation", LpMinimize)
    x = LpVariable.dicts("Route", product(demand.index, capacity.index), 0, None)
    prob += lpSum(_get_weight(n, m) * x[n, m] for n, m in x)

    for n in demand.index:
        prob += lpSum(x[n, m] for m in capacity.index) == demand.loc[n, "demand"]
    for m in capacity.index:
        prob += lpSum(x[n, m] for n in demand.index) <= capacity.loc[m, f"capacity_{service_type}"]

    prob.solve(PULP_CBC_CMD(msg=False))

    # Before processing results, initialize new columns
    gdf["closest_service"] = None
    gdf["distance_to_service"] = float('inf')
    gdf["assigned_to"] = None

    # Process the results
    assignments = pd.DataFrame(0, index=gdf.index, columns=gdf.index)

    # ensure columns exist
    if "demand_within" not in gdf.columns:
        gdf["demand_within"] = 0
    if "demand_without" not in gdf.columns:
        gdf["demand_without"] = 0

    # Find closest service and process assignments
    for (a, b), var in x.items():
        value = var.varValue
        if value == 0 or (a == fictive_block_id or b == fictive_block_id):
            continue

        weight = _get_weight(a, b)
        assignments.loc[a, b] = value

        # Update closest service if this is closer
        if weight != 1e3:
            if  weight < gdf.loc[a, "distance_to_service"]:
                # print(f"Updating closest service for {a} to {b} with weight {weight}, value {value}, {gdf.loc[a, 'distance_to_service']}")
                gdf.loc[a, "closest_service"] = b
                gdf.loc[a, "distance_to_service"] = weight

            # Update assignment information
            if value > 0:
                if gdf.loc[a, "assigned_to"] is None:
                    gdf.loc[a, "assigned_to"] = b
                else:
                    gdf.loc[a, "assigned_to"] = f"{gdf.loc[a, 'assigned_to']},{b}"

            if weight <= accessibility:
                gdf.loc[a, "demand_within"] += value
            else:
                gdf.loc[a, "demand_without"] += value

            gdf.loc[b, "capacity_left"] -= value

    if fictive_block_id in gdf.index:
        gdf.drop(index=fictive_block_id, inplace=True)
        assignments.drop(index=fictive_block_id, errors="ignore", inplace=True)
        assignments.drop(columns=fictive_block_id, errors="ignore", inplace=True)

    return gdf, assignments

def calculate_provision(
    city_model: dict,
    service_type: str,
    # blocks_gdf: gpd.GeoDataFrame,
    update_df: pd.DataFrame = None,
    method: str = "lp",
) -> gpd.GeoDataFrame:
    """
    Provision assessment using a certain method for the given city and service type.
    Can be used with updated blocks GeoDataFrame.
    """

    # Prepare the GeoDataFrame for provision assessment
    def _get_blocks_gdf(city_model):
        return gpd.GeoDataFrame(city_model["blocks"]).set_index("name").set_crs(epsg=city_model["epsg"])

    gdf = _get_blocks_gdf(city_model)

    # Update the GeoDataFrame if update_df is provided
    if update_df is not None:
        gdf = gdf.join(update_df)
        gdf[update_df.columns] = gdf[update_df.columns].fillna(0)
        if "population" not in gdf:
            gdf["population"] = 0
        gdf["demand"] += gdf["population"].apply(
            lambda pop: math.ceil(pop / 1000 * service_type["demand"])
        )
        gdf["demand_left"] = gdf["demand"]
        if service_type["name"] not in gdf:
            gdf[service_type["name"]] = 0
        gdf["capacity"] += gdf[service_type["name"]]
        gdf["capacity_left"] += gdf[service_type["name"]]

    # Choose the method for provision assessment
    if method == "lp":
        gdf, assignments = lp_provision(city_model, gdf, service_type)
    elif method == "iterative":
        raise NotImplementedError("Iterative method is not implemented yet.")
    else:
        raise ValueError("Invalid method specified. Use 'lp' or 'iterative'.")

    # Calculate the provision ratio
    gdf["provision"] = gdf["demand_within"] / gdf["demand"]

    return gdf.fillna(0), assignments

# def make_adjacency_matrix(df_time):
#     # 1. Создание списка уникальных населенных пунктов
#     unique_settlements = sorted(set(df_time["edge1"]).union(set(df_time["edge2"])))

#     # 2. Создание словаря для отображения названий населенных пунктов в id
#     settlement_id_map = {name: idx for idx, name in enumerate(unique_settlements)}

#     # 3. Создание матрицы смежности с использованием id
#     adjacency_matrix = pd.DataFrame(
#         0, index=range(len(unique_settlements)), columns=range(len(unique_settlements))
#     )

#     for _, row in df_time.iterrows():
#         # for _, row in final_df[final_df['geometry'].notna()].iterrows():
#         edge1, edge2 = row["edge1"], row["edge2"]
#         id1, id2 = settlement_id_map[edge1], settlement_id_map[edge2]
#         adjacency_matrix.at[id1, id2] = row["weight"]
#         adjacency_matrix.at[id2, id1] = row["weight"]  # Граф ненаправленный

#     # 4
#     matrix = np.array(adjacency_matrix)

#     # Замена всех нулей на 9999
#     # (или любое другое значение, которое вы хотите использовать для обозначения отсутствия связи)
#     matrix[matrix == 0] = 1e6
#     np.fill_diagonal(matrix, 0)
#     df_matrix = pd.DataFrame(matrix)


#     return df_matrix

# ---------------------------------------------------
# the location allocation problem
# ---------------------------------------------------
# Совмещенная задача


# 1. Создаем переменные для объектов (Y_j) и их вместимости (C_k)
# def add_facility_variables(range_facility, var_name_y, var_name_c):
#     y_vars = [
#         pulp.LpVariable(
#             var_name_y.format(i=i), lowBound=0, upBound=1, cat=pulp.LpInteger
#         )
#         for i in range_facility
#     ]
#     c_vars = [
#         pulp.LpVariable(var_name_c.format(i=i), lowBound=0, cat=pulp.LpInteger)
#         for i in range_facility
#     ]
#     return y_vars, c_vars


# # 2. Создаем матрицу распределения спроса (Z_ij)
# def add_assignment_variables(range_client, range_facility, var_name):
#     return np.array(
#         [
#             [
#                 pulp.LpVariable(
#                     var_name.format(i=i, j=j),
#                     lowBound=0,
#                     upBound=1,
#                     cat=pulp.LpContinuous,
#                 )
#                 for j in range_facility
#             ]
#             for i in range_client
#         ]
#     )


# # 3. Ограничение на вместимость объектов
# def add_capacity_constraints(
#     problem, y_vars, c_vars, z_vars, demand, range_client, range_facility
# ):
#     for j in range_facility:
#         problem += (
#             pulp.lpSum([demand[i] * z_vars[i, j] for i in range_client]) <= c_vars[j],
#             f"capacity_constraint_{j}",
#         )
#         problem += c_vars[j] <= y_vars[j] * 4000, f"open_capacity_constraint_{j}"
#         problem += c_vars[j] >= y_vars[j] * 20, f"min_capacity_constraint_{j}"


# # 4. Ограничение на удовлетворение спроса
# def add_demand_constraints(
#     problem, z_vars, accessibility_matrix, range_client, range_facility
# ):
#     for i in range_client:
#         problem += (
#             pulp.lpSum(
#                 [accessibility_matrix[i, j] * z_vars[i, j] for j in range_facility]
#             )
#             == 1,
#             f"demand_constraint_{i}",
#         )


# # Основная функция для решения объединенной задачи
# def solve_combined_problem(
#     cost_matrix, service_radius, demand_quantity, name="combined_problem"
# ):
#     num_clients, num_facilities = cost_matrix.shape
#     range_clients = range(num_clients)
#     range_facilities = range(num_facilities)

#     # Инициализация задачи минимизации
#     problem = pulp.LpProblem(name, pulp.LpMinimize)

#     # Матрица доступности (a_ij)
#     accessibility_matrix = (cost_matrix <= service_radius).astype(int)

#     # Переменные
#     y_vars, c_vars = add_facility_variables(range_facilities, "y[{i}]", "c[{i}]")
#     z_vars = add_assignment_variables(range_clients, range_facilities, "z[{i}_{j}]")

#     # Целевая функция: минимизация количества объектов и общей вместимости
#     w1, w2 = 1000, 1
#     problem += (
#         pulp.lpSum([w1 * y_vars[j] + w2 * c_vars[j] for j in range_facilities]),
#         "objective_function",
#     )

#     # Ограничения
#     add_capacity_constraints(
#         problem,
#         y_vars,
#         c_vars,
#         z_vars,
#         demand_quantity,
#         range_clients,
#         range_facilities,
#     )
#     add_demand_constraints(
#         problem, z_vars, accessibility_matrix, range_clients, range_facilities
#     )

#     # Решение задачи
#     solver = pulp.PULP_CBC_CMD(msg=False)
#     problem.solve(solver)

#     if problem.status != 1:
#         raise RuntimeError(f"Problem not solved: {pulp.LpStatus[problem.status]}.")

#     fac2cli = []
#     for j in range(len(y_vars)):
#         if y_vars[j].value() > 0:
#             fac_clients = [
#                 i for i in range(num_clients) if accessibility_matrix[i, j] > 0
#             ]
#             fac2cli.append(fac_clients)
#         else:
#             fac2cli.append([])

#     # Формируем результаты
#     facilities_open = [j for j in range_facilities if y_vars[j].value() > 0.5]
#     assignment = np.array(
#         [[z_vars[i, j].value() for j in range_facilities] for i in range_clients]
#     )
#     capacities = [c_vars[j].value() for j in range_facilities]

#     return facilities_open, capacities, fac2cli, assignment

# # Generate an initial population with random numbers
# def generate_population(res, accessibility_matrix_demand, population_size):

#     matrix_range = (0.5, 0.9)
#     new_matrix = accessibility_matrix_demand.copy()
#     pops = []

#     if len(res) > 50:
#         res = random.sample(res, 50)

#     for _ in range(population_size):
#         for i in res:
#             new_matrix.loc[i[0], i[1]] = (
#                 random.uniform(matrix_range[0], matrix_range[1])
#                 * accessibility_matrix_demand.loc[i[0], i[1]]
#             )
#             new_matrix.loc[i[1], i[0]] = new_matrix.loc[i[0], i[1]]
#         pops.append(new_matrix)

#     return pops

# def choose_edges(sim_matrix, service_radius):

#     edges = []

#     for i in tqdm(sim_matrix.index):
#         for j in sim_matrix.columns:
#             if sim_matrix.loc[i, j] >= service_radius and i != j:
#                 # Reduce by 40% if the value is 15 or greater
#                 variant = sim_matrix.copy()
#                 if (
#                     variant.loc[i, j] > service_radius
#                     and variant.loc[i, j] * 0.6 <= service_radius
#                 ):
#                     if [j, i] not in edges:
#                         edges.append([i, j])
#     # print(len(options))
#     # print(options)

#     return edges

# def calculate_fitness(candidate_matrix, df, service_radius, my_version, demand_col="demand_without"):
#     try:
#         if my_version == False:
#             raise ValueError("my_version must be True")
#             # solver = pulp.PULP_CBC_CMD(msg=False, warmStart=True)
#             # clscpso_2 = LSCP.from_cost_matrix(
#             #     candidate_matrix,
#             #     service_radius,
#             #     demand_quantity_arr=df["demand_without"],
#             #     facility_capacity_arr=np.array([2590] * candidate_matrix.shape[0]),
#             #     name="CLSCP-SO",
#             # )
#             # clscpso_2 = clscpso_2.solve(solver)
#             # dict_fitness = dict(
#             #     [(k, l) for k, l in enumerate(clscpso_2.fac2cli) if len(l) > 0]
#             # )

#         else:
#             facilities, capacities, fac2cli, assignments = solve_combined_problem(
#                 np.array(candidate_matrix), service_radius, df[demand_col]
#             )
#             dict_fitness = dict([(k, l) for k, l in enumerate(fac2cli) if len(l) > 0])

#         fitness = len(dict_fitness)
#         for key, _ in dict_fitness.items():
#             if df.loc[key, "capacity"] > 0:
#                 fitness -= 1

#         return fitness
#     except RuntimeError:
#         return 0

# # Main genetic algorithm
# def genetic_algorithm(
#     matrix,
#     edges,
#     population_size,
#     num_generations,
#     df,
#     service_radius,
#     mutation_rate,
#     num_parents,
#     num_offspring,
#     my_version,
#     demand_col="demand_without",
# ):
#     population = generate_population(edges, matrix, population_size)
#     fitness_history = []  # История изменения фитнеса

#     for _ in trange(num_generations):
#         # Рассчитываем фитнес и отсортированную популяцию
#         population_with_fitness = [
#             (individual, calculate_fitness(individual, df, service_radius, my_version, demand_col=demand_col))
#             for individual in population
#         ]

#         # Сохраняем фитнес текущей популяции
#         fitness_history.append([fitness for _, fitness in population_with_fitness])

#         # Отбираем родителей
#         parents = [
#             individual
#             for individual, _ in sorted(population_with_fitness, key=lambda x: x[1])[
#                 :num_parents
#             ]
#         ]

#         # Генерируем потомков
#         offspring = crossover(parents, num_offspring, matrix)

#         # Применяем мутацию
#         offspring_mutationed = mutation(offspring, edges, mutation_rate)

#         # Обновляем популяцию
#         population = parents + offspring_mutationed

#     # Получаем лучшее решение
#     best_candidate, _ = min(population_with_fitness, key=lambda x: x[1])

#     return best_candidate, fitness_history


# # Perform selection based on fitness scores
# def selection(population, num_parents, df, service_radius, my_version):
#     sorted_population = sorted(
#         population,
#         key=lambda x: calculate_fitness(x, df, service_radius, my_version),
#         reverse=False,
#     )
#     parents = sorted_population[:num_parents]
#     return parents


# # Perform crossover to create offspring
# def crossover(parents, num_offspring, matrix):

#     matrix_size = len(matrix)

#     offspring = []
#     while len(offspring) < num_offspring:
#         parent1 = random.choice(parents)
#         parent2 = random.choice(parents)
#         crossover_point = random.randint(1, matrix_size - 1)
#         child1 = pd.DataFrame(
#             np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
#         )
#         child2 = pd.DataFrame(
#             np.vstack((parent2[:crossover_point], parent1[crossover_point:]))
#         )

#         # for i in range(matrix_size):
#         #     for j in range(i + 1, matrix_size):  # Проходим только верхнюю часть матрицы
#         #         child1[j][i] = child1[i][j]
#         #         child2[j][i] = child2[i][j]

#         offspring.append(child1)
#         offspring.append(child2)

#     return offspring

# # Perform mutation to introduce random changes
# def mutation(offspring, res, mutation_rate):

#     matrix_range = (0.5, 1.0)
#     offspring_mutationed = []

#     for child in offspring:
#         if random.random() < mutation_rate:
#             index = random.choice(res)
#             child.loc[index[0], index[1]] = (
#                 random.uniform(matrix_range[0], matrix_range[1])
#                 * child.loc[index[0], index[1]]
#             )
#             child.loc[index[1], index[0]] = child.loc[index[0], index[1]]

#     offspring_mutationed.append(child)

#     return offspring_mutationed




