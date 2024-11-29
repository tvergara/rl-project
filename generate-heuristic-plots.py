import numpy as np
import matplotlib.pyplot as plt
from main_q_star import generate_heuristic
from basic_grid_env import BasicGridEnv

if __name__ == '__main__':
    seed = 42
    size = 25
    kappa = 5

    env = BasicGridEnv(seed=seed, size=size)
    heuristic = generate_heuristic(kappa)

    heatmap_values = np.zeros((size, size))

    for x in range(size):
        for y in range(size):
            env.agent_pos = (x, y)
            heatmap_values[x, y] = heuristic(env)

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_values, origin='lower', cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Valor de la heurística')
    plt.title('Heatmap de la heurística')
    plt.xlabel('')
    plt.ylabel('')
    plt.savefig('heatmap.png', dpi=300)
