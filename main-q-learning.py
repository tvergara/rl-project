from grid_env import GridEnv
from q_learning import QLearning
from tqdm import tqdm
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    env = GridEnv(seed=42, size=10)
    qlearning_agent = QLearning(num_actions=4, epsilon=0.1, alpha=0.5, gamma=0.8, initial_value=0.0)

    num_episodes = 1000
    steps = []

    for episode in tqdm(range(num_episodes)):
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

    plot_steps(steps)
    print("Training complete.")
