from main_q_learning import q_learning
from main_q_star import q_star_learning
from utils import run_experiment
import matplotlib.pyplot as plt
import random


def plot_steps(q_learning, q_star):
    plt.figure(figsize=(10, 5))

    plt.plot(q_learning, label=f"Q-learning")
    plt.plot(q_star, label=f"Q*-Learning")

    plt.xlabel('Episodio')
    plt.ylabel('Pasos')
    plt.ylim(0, 600)
    plt.title('Q-learning vs Q*-Learning')
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig(f"k-exp.png")


def plot_in_range(steps, values, no_heuristic):
    plt.figure(figsize=(10, 5))

    for step, value in zip(steps, values):
        plt.plot(step, label=f"kappa: {value}")

    plt.plot(no_heuristic, label="No heuristic")

    plt.xlabel('Episodio')
    plt.ylabel('Pasos')
    plt.ylim(0, 500)
    plt.title('Q*-Learning bajo distintos niveles de ruido')
    plt.grid(True)
    plt.legend(loc='best')
    plt.savefig(f"h-exp.png")

if __name__ == '__main__':
    n_experiments = 10
    budget=5
    k=5
    t=20
    epsilon=0.1
    alpha=0.5
    gamma=0.8
    initial_value=0.0
    episodes=100
    seed=42
    size=100
    kappa=0.1
    random.seed(seed)

    # q_star_steps = run_experiment(
    #     lambda: q_star_learning(
    #         budget=budget,
    #         k=k,
    #         t=t,
    #         epsilon=epsilon,
    #         alpha=alpha,
    #         gamma=gamma,
    #         initial_value=initial_value,
    #         episodes=episodes,
    #         seed=random.randint(1, 2<<20),
    #         size=size,
    #         kappa=0,
    #     ),
    #     n=n_experiments,
    # )

    # # q_learning_steps = run_experiment(
    # #     lambda: q_learning(
    # #         epsilon=epsilon,
    # #         alpha=alpha,
    # #         gamma=gamma,
    # #         initial_value=initial_value,
    # #         episodes=episodes,
    # #         seed=random.randint(1, 2<<20),
    # #         size=size,
    # #     ),
    # #     n=n_experiments,
    # # )

    q_learning_steps = run_experiment(
        lambda: q_star_learning(
            budget=budget,
            k=k,
            t=t,
            epsilon=epsilon,
            alpha=alpha,
            gamma=gamma,
            initial_value=initial_value,
            episodes=episodes,
            seed=random.randint(1, 2<<20),
            size=size,
            kappa=kappa,
        ),
        n=n_experiments,
    )

    # plot_steps(q_learning_steps, q_star_steps)


#     steps = []
#     kappas = [0, 0.1, 0.3, 1, 1.5, 2.5, 5.0]
#     for selected_kappa in kappas:
#         steps.append(run_experiment(
#             lambda: q_star_learning(
#                 budget=budget,
#                 k=k,
#                 t=t,
#                 epsilon=epsilon,
#                 alpha=alpha,
#                 gamma=gamma,
#                 initial_value=initial_value,
#                 episodes=episodes,
#                 seed=random.randint(1, 2<<20),
#                 size=size,
#                 kappa=selected_kappa,
#             ),
#             n=n_experiments,
#         ))

#     plot_in_range(steps, kappas, q_learning_steps)



