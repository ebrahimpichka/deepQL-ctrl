
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    plt.style.use("ggplot")
    fig=plt.figure()
    ax=fig.add_subplot(121, label="1")
    ax2=fig.add_subplot(122, label="2", frame_on=True)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.set_ylabel('Score', color="C1")
    ax2.set_xlabel("Training Steps", color="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)
    plt.tight_layout(1)
    plt.savefig(filename)

