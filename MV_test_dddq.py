import gym
import numpy as np
from agents.dueling_ddqn_torch import Agent
from utils import plotLearning
from ps_environment import Environment
import numpy as np
from plantsim.plantsim import Plantsim
from agents.ppo_torch import PPOAgent
from utils import plot_learning_curve
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
pfad = 'C:\\Users\leonv\Documents\Programmierungen_Studium\PlantSimulationRL\simulations'
# pfad = 'D:\\Studium\Projekt\Methodenvergleich\PlantSimulationRL\simulations'

# model = pfad + '\Methodenvergleich_20220911_mitLager.spp'
# model = pfad + '\Methodenvergleich_20220913_real.spp'
# model = pfad + '\Methodenvergleich_20220915_Naiver_reward_wenig_states.spp'
# model = pfad + '\Methodenvergleich_20220915_Naiver_reward.spp'
model = pfad + '\Methodenvergleich_20220916_Naiver_reward_wenig_states.spp'

if __name__ == '__main__':
    plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell', model=model,
                        socket=None, visible=False)
    env = Environment(plantsim)
    num_games = 30
    load_checkpoint = True  # False True

    actions = env.problem.get_all_actions()
    observation = env.reset()
    env.problem.plantsim.execute_simtalk("GetCurrentState")
    env.problem.get_current_state()
    test = env.problem.state
    decay = 0  # (1+0.08)/(300*num_games)

    agent = Agent(gamma=0.99, epsilon=0.0, lr=0.0005,  # 5e-4,
                  input_dims=[len(test)], n_actions=len(actions), mem_size=50000, eps_min=0.0,
                  batch_size=64, eps_dec=decay, replace=20,
                  chkpt_dir='tmp/dddq/dueling_ddqn_very_naive_few_states_2')  # eps_dec=2e-5 eps_dec=0.99993

    if load_checkpoint:
        agent.load_models()

    filename = 'tmp\dddq\Dueling-DDQN-MV_17_wenig_states_test.png'
    scores = []
    eps_history = []
    Lieferterminabweichung = []
    n_steps = 0
    best_score = 0  # 107 # 420  # 271.424

    for i in range(num_games):
        if i > 0:
            env.reset()
        done = False
        observation = None  # env.reset()
        score = 0
        count = 1
        step = 0
        while not done:
            step += 1
            if observation is None:
                env.problem.plantsim.execute_simtalk("GetCurrentState")
                current_state = env.problem.get_current_state()
                observation = current_state.to_state()
            action = agent.choose_action(observation)
            a = env.problem.actions[action]
            env.problem.act(a)
            current_state = env.problem.get_current_state()
            observation_ = current_state.to_state()
            reward = env.problem.get_reward(current_state)
            if reward > 0 and not done:
                count += 1
            score += reward
            done = env.problem.is_goal_state(current_state)
            # print("Step " + str(step) + ": " + a + " - Reward: " + str(reward) + " - finished: " + str(
            #     count - 1) + "\n")  # + " - " + str(round((step / count), 3)) +

            # agent.store_transition(observation, action,
            #                       reward, observation_, int(done))
            # agent.learn()

            observation = observation_
        scores.append(score)
        avg_score = np.mean(scores[max(0, i - 100):(i + 1)])
        if score > best_score: #and agent.epsilon < 0.9:
            #agent.save_models()
            best_score = score

        eps_history.append(agent.epsilon)
        print('episode: ', i, ' -- score %.1f ' % score,
              ' -- average score %.1f' % avg_score,
              ' -- best score %.1f' % best_score,
              'epsilon %.5f' % agent.epsilon)
        # if i > 0 and i % 10 == 0:
        Lieferterminabweichung.append(env.problem.plantsim.get_value('Verspätung'))

    x = [i + 1 for i in range(num_games)]
    plotLearning(x, scores, eps_history, filename)
    with open("tmp\dddq\dddq_performance_train_17_scores.txt", "w") as output:
        output.write(str(scores))
    with open("tmp\dddq\dddq_performance_train_17_epsilon.txt", "w") as output:
        output.write(str(eps_history))
    with open("tmp\dddq\dddq_performance_train_17_verspaetung.txt", "w") as output:
        output.write(str(Lieferterminabweichung))
