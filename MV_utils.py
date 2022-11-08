import numpy as np
import matplotlib.pyplot as plt
import json


# plt.style.use('science')


def plotLearning(x, scores, epsilons=None, name='test', lines=None):
    with plt.style.context(['science', 'ieee']):
        if epsilons is not None:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax2 = fig.add_subplot(frame_on=False)

            ax.plot(x, epsilons, c='gray')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Epsilon')
            ax.yaxis.set_ticks_position('left')
            ax.xaxis.set_ticks_position('bottom')
            # ax.tick_params(right=False)
            # ax.tick_params(top=False)
            # ax.set_yticks(np.arange(0.2, 0.6, 0.1))
            ax.tick_params(axis='y')
            ax.spines.top.set_visible(True)

            N = len(scores)
            running_avg = np.empty(N)
            for t in range(N):
                running_avg[t] = np.mean(scores[max(0, t - 30):(t + 1)])

            ax2.plot(x, running_avg)  # , 5, marker='x'
            ax2.axes.get_xaxis().set_visible(False)
            ax2.yaxis.tick_right()
            ax2.set_ylabel('Return')
            ax2.yaxis.set_label_position('right')
            ax2.yaxis.set_ticks_position('right')
            ax2.xaxis.set_ticks_position('bottom')
            # ax2.tick_params(right=False)
            ax2.spines.top.set_visible(False)
            # ax2.set_yticks(np.arange(100, 300, 50))
            # ax2.tick_params(top=False)

            # ax2.tick_params(axis='y')
            # plt.tick_params(top=False)

        else:
            fig = plt.figure()
            ax = fig.add_subplot()

            N = len(scores)
            running_avg = np.empty(N)
            for t in range(N):
                running_avg[t] = np.mean(scores[max(0, t - 30):(t + 1)])

            ax.plot(x, running_avg)  # , 5, marker='x'
            ax.set_xlabel('Episode')
            ax.set_ylabel('Return')
            ax.yaxis.set_ticks_position('both')
            ax.xaxis.set_ticks_position('bottom')
            ax.spines.top.set_visible(True)

        plt.savefig(name + '.pdf')
        plt.savefig(name + '.jpg', dpi=300)

        #  plt.show()


def get_data(pfad, file):
    location = pfad + file
    txt = open(location, "r")
    data = json.loads(txt.read())
    txt.close()
    return data


ergebnisse = 'Ergebnisse'

#  Simples Modell DDDQ
pfad_q = 'tmp\dddq'
# score_1 = '\dddq_performance_train_17_scores.txt'
score_2 = '\dddq_performance_train_17_scores_1.txt'
epsilon_2 = '\dddq_performance_train_17_epsilon_1.txt'
score_3 = '\dddq_performance_train_17_scores_2.txt'
epsilon_3 = '\dddq_performance_train_17_epsilon_2.txt'

scores = get_data(pfad_q, score_2)
eps_history = get_data(pfad_q, epsilon_2)
filename = ergebnisse + '\\Alles\Dueling-DDQN-MV_wenig_states1'
x = [i + 1 for i in range(len(scores))]
plotLearning(x=x, scores=scores, epsilons=eps_history, name=filename)

scores = get_data(pfad_q, score_3)
eps_history = get_data(pfad_q, epsilon_3)
filename = ergebnisse + '\\Alles\Dueling-DDQN-MV_wenig_states2'
x = [i + 1 for i in range(len(scores))]
plotLearning(x=x, scores=scores, epsilons=eps_history, name=filename)

#  Komplexes Modell DDDQ
pfad_q_v = 'tmp\dddq_viele_states'
score_1 = '\dddq_performance_train_17_scores.txt'
epsilon_1 = '\dddq_performance_train_17_epsilon.txt'
score_2 = '\dddq_performance_train_18_scores_1.txt'
epsilon_2 = '\dddq_performance_train_18_epsilon_1.txt'
score_3 = '\dddq_performance_train_18_scores_2.txt'
epsilon_3 = '\dddq_performance_train_18_epsilon_2.txt'
score_4 = '\dddq_performance_train_18_scores_3.txt'
epsilon_4 = '\dddq_performance_train_18_epsilon_3.txt'

scores = get_data(pfad_q_v, score_1)
eps_history = get_data(pfad_q_v, epsilon_1)
filename = ergebnisse + '\\Alles\Dueling-DDQN-MV_viele_states1'
x = [i + 1 for i in range(len(scores))]
plotLearning(x=x, scores=scores, epsilons=eps_history, name=filename)

scores = get_data(pfad_q_v, score_2)
eps_history = get_data(pfad_q_v, epsilon_2)
filename = ergebnisse + '\\Alles\Dueling-DDQN-MV_viele_states2'
x = [i + 1 for i in range(len(scores))]
plotLearning(x=x, scores=scores, epsilons=eps_history, name=filename)

scores = get_data(pfad_q_v, score_3)
eps_history = get_data(pfad_q_v, epsilon_3)
filename = ergebnisse + '\\Alles\Dueling-DDQN-MV_viele_states3'
x = [i + 1 for i in range(len(scores))]
plotLearning(x=x, scores=scores, epsilons=eps_history, name=filename)

scores = get_data(pfad_q_v, score_4)
eps_history = get_data(pfad_q_v, epsilon_4)
filename = ergebnisse + '\\Alles\Dueling-DDQN-MV_viele_states4'
x = [i + 1 for i in range(len(scores))]
plotLearning(x=x, scores=scores, epsilons=eps_history, name=filename)

#  Simples Modell PPO
pfad_p = 'tmp\ppo'
# score_0 = '\ppp_performance_train_15_1.txt'
score_1 = '\ppo_performance_train_16_20220916_1.txt'
score_2 = '\ppo_performance_train_17_1.txt'
score_3 = '\ppo_performance_train_17_2.txt'
score_4 = '\ppo_performance_train_17_3.txt'

scores = get_data(pfad_p, score_1)
filename = ergebnisse + '\\Alles\PPO-MV_wenig_states1'
x = [i + 1 for i in range(len(scores))]
plotLearning(x=x, scores=scores, name=filename)

scores = get_data(pfad_p, score_2)
filename = ergebnisse + '\\Alles\PPO-MV_wenig_states2'
x = [i + 1 for i in range(len(scores))]
plotLearning(x=x, scores=scores, name=filename)

scores = get_data(pfad_p, score_3)
filename = ergebnisse + '\\Alles\PPO-MV_wenig_states3'
x = [i + 1 for i in range(len(scores))]
plotLearning(x=x, scores=scores, name=filename)

scores = get_data(pfad_p, score_4)
filename = ergebnisse + '\\Alles\PPO-MV_wenig_states4'
x = [i + 1 for i in range(len(scores))]
plotLearning(x=x, scores=scores, name=filename)

#  Komplexes Modell PPO
pfad_p_v = 'tmp\ppo_viele_states'
score_1 = '\ppo_performance_train_17_4.txt'
score_2 = '\ppo_performance_train_17_5.txt'
#score_3 = '\ppo_performance_train_17_6.txt'

scores = get_data(pfad_p_v, score_1)
filename = ergebnisse + '\\Alles\PPO-MV_viele_states1'
x = [i + 1 for i in range(len(scores))]
plotLearning(x=x, scores=scores, name=filename)

scores = get_data(pfad_p_v, score_2)
filename = ergebnisse + '\\Alles\PPO-MV_viele_states2'
x = [i + 1 for i in range(len(scores))]
plotLearning(x=x, scores=scores, name=filename)
'''
scores = get_data(pfad_p_v, score_3)
filename = ergebnisse + '\\PPO_komplex\PPO-MV_viele_states3'
x = [i + 1 for i in range(len(scores))]
plotLearning(x=x, scores=scores, name=filename)


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
'''
