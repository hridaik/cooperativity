import networkx as nx
import random
import math
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
import numpy as np

def metabolicAgeFactor(age):
    return 0

def find_index(data_list, value):
    for index, tuple_data in enumerate(data_list):
        if tuple_data[0] == value:
            return index
    return -1

def generate_network(n, k, alpha):
    g = nx.Graph()
    contributions = [0, 0.25, 0.5, 0.75]
    initial_mass = 5

    # Create clique of k+1 nodes with random coordinates
    for i in range(k+1):
        x = random.uniform(0.45, 0.55)
        y = random.uniform(0.45, 0.55)
        g.add_node(i, pos=(x, y), contribution=random.choice(contributions), mass=initial_mass, age=0)

    # Create remaining nodes
    for i in range(k+1, n):
        x = random.random()
        y = random.random()
        g.add_node(i, pos=(x, y), contribution=random.choice(contributions), mass=initial_mass, age=0)

        # Calculate mean degree
        mean_degree = k

        # Randomly set number_of_links for n[i] such that mean degree is k
        number_of_links = random.randint(0, mean_degree*2)

        # Create links
        for j in range(number_of_links):
            if random.random() < alpha:
                # Create link to existing node not neighbor chosen proportional to degree
                non_neighbors = [node for node in g.nodes() if node != i and node not in g.neighbors(i)]
                probs = [g.degree(node) for node in non_neighbors]
                chosen_node = random.choices(non_neighbors, weights=probs)[0]
                g.add_edge(i, chosen_node)
            else:
                # Create link to closest non-neighbor
                min_dist = float('inf')
                closest_node = None
                for node in g.nodes():
                    if node != i and node not in g.neighbors(i):
                        dist = math.sqrt((g.nodes[node]['pos'][0] - x)**2 + (g.nodes[node]['pos'][1] - y)**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_node = node
                if closest_node is not None:
                    g.add_edge(i, closest_node)

    return g


def public_goods_game(network, rewire_p, r):
    initial_mass_values = {}  # Dictionary to store initial mass values
    worst_links = {node_id: None for node_id in network.nodes()}

    nodes_to_cut = []

    fresh_network = network
    
    # Store initial mass values of each node
    for node in fresh_network.nodes(data=True):
        node_id = node[0]
        initial_mass_values[node_id] = node[1]['mass']
        lowest_contri = 1
        worst_link = None
        neighbors_now = list(fresh_network.neighbors(node_id))
        if len(neighbors_now) > 0:
            #for neighbor_id in neighbors_now:
                #neighbor = find_index(neighbors_now, neighbor_id)
            for neighbor_id in neighbors_now:
                neighbor = neighbors_now.index(neighbor_id)
                contri_n = list(fresh_network.nodes(data=True))[neighbor][1]['contribution']
                if contri_n < lowest_contri:
                    worst_link = neighbor
                    lowest_contri = contri_n
            worst_links[node_id] = worst_link

    
    # Iterate over each node in the fresh_network
    for node in fresh_network.nodes(data=True):
        node_id = node[0]
        node[1]['age'] += 1
        contribution = node[1]['contribution'] - metabolicAgeFactor(node[1]['age'])
        
        neighbors = list(fresh_network.neighbors(node_id))
        num_neighbors = len(neighbors)
        
        # Deduct contribution from mass
        node[1]['mass'] -= contribution

        # ALSO DEDUCT ROUND PENALTY
        play_penalty = 0.25
        node[1]['mass'] -= play_penalty

        # Distribute contribution equally among neighbors
        if num_neighbors > 0:
            contribution_per_neighbor = r * contribution / num_neighbors
            for neighbor_id in neighbors:
                # neighbor = find_index(neighbors, neighbor_id)
                neighbor = neighbors.index(neighbor_id)
                list(fresh_network.nodes(data=True))[neighbor][1]['mass'] += contribution_per_neighbor

        if node[1]['mass'] <= 0:
            nodes_to_cut.append((node_id, node[1]['age']))
            
        
        
        
    
    # Check if mass has increased or decreased for each node
    for node in fresh_network.nodes(data=True):
        node_id = node[0]
        current_mass = node[1]['mass']
        initial_mass = initial_mass_values[node_id]
        neighbors = list(fresh_network.neighbors(node_id))
        num_neighbors = len(neighbors)

        if current_mass >= initial_mass:
            # Do something
            if random.random() > 0.5:
                node[1]['contribution'] += 0.25
        else:
            if node[1]['contribution'] > 0:
                if random.random() < rewire_p:
                    if num_neighbors > 0:
                        fresh_network = rewire_links(fresh_network, node_id, worst_links[node_id])
                else:
                    node[1]['contribution'] -= 0.25
                    
            else:
                if num_neighbors > 0:
                    fresh_network = rewire_links(fresh_network, node_id, worst_links[node_id])


    # Remove nodes from the network and update worst_links
    print(f"Nodes {nodes_to_cut} are dying")
    for nid, age in nodes_to_cut:
        fresh_network.remove_node(nid)
        if nid in worst_links:
            del worst_links[nid]


    return fresh_network, nodes_to_cut

            


def rewire_links(network, node_id, worst_link):
    # Remove link to the worst node
    fresh_network = network
    if fresh_network.has_edge(node_id, worst_link):
        fresh_network.remove_edge(node_id, worst_link)
    
    # Choose a random node to join
    all_nodes = set(fresh_network.nodes())
    potential_nodes = all_nodes - set(fresh_network.neighbors(node_id)) - {node_id} - {worst_link}
    
    if potential_nodes:
        new_node = random.choice(list(potential_nodes))
        fresh_network.add_edge(node_id, new_node)
    # print(f'Rewired {node_id}-{worst_link} to {new_node}')
    return fresh_network
    
def remove_node(network, node_id):
    fresh_network = network.remove_node(node_id)
    return fresh_network

def range_from_list(lst):
    return list(range(lst[0], lst[1] + 1))




def add_nodes(graph, nodes_to_be_added, k, alpha, contributions=[0, 0.25, 0.5, 0.75], initial_mass=5):
    for i in range(len(graph.nodes), len(graph.nodes) + nodes_to_be_added):
        x = random.random()
        y = random.random()
        graph.add_node(i, pos=(x, y), contribution=random.choice(contributions), mass=initial_mass, age=0)

        mean_degree = k
        number_of_links = random.randint(0, mean_degree*2)

        for j in range(number_of_links):
            if random.random() < alpha:
                non_neighbors = [node for node in graph.nodes() if node != i and node not in graph.neighbors(i)]
                probs = [graph.degree(node) for node in non_neighbors]
                if not non_neighbors:
                    continue
                chosen_node = random.choices(non_neighbors, weights=probs)[0]
                graph.add_edge(i, chosen_node)
            else:
                min_dist = float('inf')
                closest_node = None
                for node in graph.nodes():
                    if node != i and node not in graph.neighbors(i):
                        dist = math.sqrt((graph.nodes[node]['pos'][0] - x)**2 + (graph.nodes[node]['pos'][1] - y)**2)
                        if dist < min_dist:
                            min_dist = dist
                            closest_node = node
                if closest_node is not None:
                    graph.add_edge(i, closest_node)

    return graph


# Game loop
n = 20 # number of players
enhancement_factor = 1.5 # enhancement factor
rewire_probability = 0.5

alpha = 0.3
avgDegree = 3

num_rounds = 20
game_network = generate_network(n, avgDegree, alpha)

initial_network = game_network.copy()

lifespan_by_contribution = {0: [], 0.25: [], 0.5: [], 0.75: [], 1.0: []}


fig, ax = plt.subplots()

output_dir = 'frames'
os.makedirs(output_dir, exist_ok=True)
nx.draw(game_network, with_labels=True, ax=ax)
ax.set_title("Initial Network")
plt.savefig(os.path.join(output_dir, 'frame_0.png'))


all_dead = False

print(f"Playing with rewiring probability {rewire_probability} and enhancement factor {enhancement_factor}")


for round in range(num_rounds):
    print(f'Playing round {round+1}')

    ax.clear()

    result, dead_nodes = public_goods_game(game_network, rewire_probability, enhancement_factor)

    for node_id, age in dead_nodes:
        if node_id < (n-1):
            initial_contribution = initial_network.nodes[node_id]['contribution']
        lifespan_by_contribution[initial_contribution].append(age)
    
    print(f'Number of players = {result.number_of_nodes()}')
    totalM = 0
    totalC = 0
    totalA = 0
    for node in result.nodes(data=True):
        node_id = node[0]
        contri = node[1]['contribution']
        mass = node[1]['mass']
        age = node[1]['age']
        
        totalM += mass
        totalC += contri
        totalA += age


    if result.number_of_nodes() <= 0:
        print("All nodes died")
        all_dead = True
        num_rounds_completed = round
        break
    avgC = totalC/(result.number_of_nodes())
    avgM = totalM/(result.number_of_nodes())
    avgA = totalA/(result.number_of_nodes())
    print(f'Avg Contribution: {avgC}, Avg Mass: {avgM}, Avg Age: {avgA}')

    nx.draw(result, with_labels=True, ax=ax)
    ax.set_title("Round {}".format(round+1))
    plt.savefig(os.path.join(output_dir, 'frame_{}.png'.format(round+1)))

    print('\n')

    if ((round+1)%3) == 0:
        game_network = add_nodes(result, 3, avgDegree, alpha)
    else:
        game_network = result


if all_dead:
    images = [os.path.join(output_dir, 'frame_{}.png'.format(i)) for i in range(num_rounds_completed)]
else:
    images = [os.path.join(output_dir, 'frame_{}.png'.format(i)) for i in range(num_rounds)]


with imageio.get_writer(os.path.join(output_dir, 'graph_animation.gif'), mode='I', duration=0.5) as writer:
    for image in images:
        writer.append_data(imageio.imread(image))


print('\n')

print("Final Results")

print(f"Rewiring probability {rewire_probability} and enhancement factor {enhancement_factor}")
print('\n')

for contribution, lifespans in lifespan_by_contribution.items():
    if lifespans:
        avg_lifespan = np.mean(lifespans)
        print(f"Average lifespan for initial contribution {contribution}: {avg_lifespan}")


