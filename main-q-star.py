from grid_env import GridEnv
from q_learning import QLearning
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

def plot_steps(steps, filename="steps-plot-qstar.png"):
    window = 10
    averaged_steps = [sum(steps[i:i+window]) / len(steps[i:i+window]) for i in range(0, len(steps), window)]

    plt.figure(figsize=(10, 6))
    plt.plot(averaged_steps, marker='o', linestyle='-', label="Averaged Steps")
    plt.title("Steps Averaged Every 10 Values")
    plt.xlabel("Averaged Index (Every 10 Steps)")
    plt.ylabel("Averaged Steps")
    plt.grid(True)
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.close()


def heuristic(env):
    position_x, position_y, direction = env.get_state()

    distance_x = env.goal_x - position_x
    distance_y = env.goal_y - position_y

    return -math.sqrt(distance_x ** 2 + distance_y ** 2)


if __name__ == "__main__":
    budget = 5
    k = 5
    t = 20
    envs = [GridEnv(seed=42, size=10) for _ in range(budget)]
    qlearning_agent = QLearning(num_actions=4, epsilon=0.1, alpha=0.5, gamma=0.8, initial_value=0.0)

    num_episodes = 1000
    steps = []

    for episode in tqdm(range(num_episodes)):
        for env in envs:
            env.reset()

        solution_found = False
        ep_steps = 0

        while not solution_found:
            for env in envs:
                for step in range(t):
                    state = env.get_state()
                    action = qlearning_agent.sample_action(state)
                    next_state, reward, done, _, _ = env.step(action)
                    qlearning_agent.learn(state, action, reward, next_state, done)
                    if done:
                        if not solution_found:
                            solution_found = True
                            ep_steps += step
                        break

            if solution_found:
                break

            ep_steps += t
            envs = sorted(envs, key=heuristic, reverse=True)
            envs = envs[:budget // k]
            envs = [env.copy() for env in envs for _ in range(k)]

        steps.append(ep_steps)

    plot_steps(steps)
    print("Training complete.")
