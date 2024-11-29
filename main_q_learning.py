from basic_grid_env import BasicGridEnv
from q_learning import QLearning
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
import hashlib

def plot_steps(steps, filename="steps-plot-qlearning.png"):
    window = 10
    averaged_steps = [sum(steps[i:i+window]) / len(steps[i:i+window]) for i in range(0, len(steps), window)]

    plt.figure(figsize=(10, 6))
    plt.plot(averaged_steps, marker='o', linestyle='-', label="Averaged Steps")
    plt.title("Steps Averaged Every 10 Values")
    plt.xlabel("Averaged Index (Every 10 Steps)")
    plt.ylabel("Averaged Steps")
    plt.ylim(0,50)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.close()

def q_learning(
    epsilon=0.1,
    alpha=0.5,
    gamma=0.8,
    initial_value=0.0,
    episodes=1000,
    seed=42,
    size=10,
):
    hyperparams = {
        "epsilon": epsilon,
        "alpha": alpha,
        "gamma": gamma,
        "initial_value": initial_value,
        "episodes": episodes,
        "seed": seed,
        "size": size,
    }
    hyperparams_hash = hashlib.md5(json.dumps(hyperparams, sort_keys=True).encode()).hexdigest()
    cache_dir = "tmp"
    cache_file = os.path.join(cache_dir, f"{hyperparams_hash}.json")

    if os.path.exists(cache_file):
        print(f"Loading cached results from {cache_file}")
        with open(cache_file, "r") as file:
            return json.load(file)

    env = BasicGridEnv(seed=seed, size=size)
    qlearning_agent = QLearning(
        num_actions=4,
        epsilon=epsilon,
        alpha=alpha,
        gamma=gamma,
        initial_value=initial_value)

    num_episodes = episodes
    steps = []

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        ep_steps = 0

        while not done:
            action = qlearning_agent.sample_action(state)
            next_state, reward, done, _, _ = env.step(action)
            qlearning_agent.learn(state, action, reward, next_state, done)
            state = next_state
            ep_steps += 1
        steps.append(ep_steps)

    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, "w") as file:
        json.dump(steps, file)

    return steps


if __name__ == "__main__":
    steps = q_learning()
    plot_steps(steps)
    print("Training complete.")
