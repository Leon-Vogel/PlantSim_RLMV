import numpy as np
import matplotlib.pyplot as plt
import json
# plt.style.use('science')


def plotLearning(x, scores, epsilons, filename, lines=None):
    with plt.style.context(['science', 'ieee']):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax2 = fig.add_subplot(frame_on=False)

        ax.plot(x, epsilons, c='gray')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        #ax.tick_params(right=False)
        #ax.tick_params(top=False)
        ax.set_yticks(np.arange(0.2, 0.6, 0.1))
        # ax.tick_params(axis='y')
        ax.spines.top.set_visible(True)

        N = len(scores)
        running_avg = np.empty(N)
        for t in range(N):
            running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

        ax2.plot(x, running_avg)  #, 5, marker='x'
        ax2.axes.get_xaxis().set_visible(False)
        ax2.yaxis.tick_right()
        ax2.set_ylabel('Score')
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.set_ticks_position('right')
        ax2.xaxis.set_ticks_position('bottom')
        #ax2.tick_params(right=False)
        ax2.spines.top.set_visible(False)
        #ax2.set_yticks(np.arange(100, 300, 50))
        #ax2.tick_params(top=False)

        # ax2.tick_params(axis='y')
        #plt.tick_params(top=False)
        plt.savefig(filename+'.pdf')
        plt.savefig(filename+'.jpg', dpi=300)

        #  plt.show()


pfad = 'tmp\dddq_viele_states'
score = '\dddq_performance_train_18_scores.txt'
location = pfad + score
file = open(location, "r")
scores = json.loads(file.read())
file.close()

epsilon = '\dddq_performance_train_18_epsilon.txt'
location = pfad + epsilon
file = open(location, "r")
eps_history = json.loads(file.read())
file.close()

filename = pfad + '\\bilder\Dueling-DDQN-MV_18_viele_states'

x = [i + 1 for i in range(len(scores))]
plotLearning(x, scores, eps_history, filename)
