from time import sleep

from ps_environment import Environment
import numpy as np
from plantsim.plantsim import Plantsim
from agents.ppo_torch import PPOAgent
from utils import plot_learning_curve
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# pfad = 'D:\\Studium\Projekt\Methodenvergleich\PlantSimulationRL\simulations'
pfad = 'C:\\Users\leonv\Documents\Programmierungen_Studium\PlantSimulationRL\simulations'
# model = pfad + '\Methodenvergleich_20220909_mitLager.spp'
speicherort = 'tmp\MV_ppo_16'

model = pfad + '\Methodenvergleich_20220916_Naiver_reward_wenig_states.spp'
model = pfad + '\Methodenvergleich_20220915_Naiver_reward_wenig_states.spp'
# model = pfad + '\Methodenvergleich_20220914_real_mitTyp.spp'
# model = pfad + '\Methodenvergleich_20220913_real.spp'
# model = pfad + '\Methodenvergleich_20220911_mitLager.spp'
if __name__ == '__main__':
    plantsim = Plantsim(version='22.1', license_type='Educational', path_context='.Modelle.Modell', model=model,
                        socket=None, visible=False)
    env = Environment(plantsim)  # env = gym.make('InvertedPendulumBulletEnv-v0')
    # Nachkommastelle an N anfügen, um das Lernen auszuschalten
    N = 200  # 30 to 5000 steps between training # or done [line 96]
    batch_size = 64  # 4 to 4096
    n_epochs = 10  # 3 to 30 epochs in training
    alpha = 0.0005  # 0.0003
    '''N = 20000000  # 30 to 5000 steps between training # or done [line 96]
    batch_size = 128  # 4 to 4096
    n_epochs = 5  # 3 to 30 epochs in training
    alpha = 0.0003  # 0.0003'''

    actions = env.problem.get_all_actions()
    observation = env.reset()
    env.problem.plantsim.execute_simtalk("GetCurrentState")
    env.problem.get_current_state()
    test = env.problem.state

    agent = PPOAgent(input_dims=[len(test)], env=env.problem,
                     n_actions=len(actions), batch_size=batch_size,
                     alpha=alpha, n_epochs=n_epochs, speicherort=speicherort)

    max_iterations = 200
    filename = 'PPO_training.png'
    figure_file = 'tmp/' + filename
    best_score = -16  # 716 # 822 # 3932
    performance_train = []
    learn_iters = 0
    avg_score = 0
    n_steps = 0
    load_checkpoint = True  # True False
    save_changes = True

    if load_checkpoint:
        agent.load_models()

    for i in range(max_iterations):
        env.reset()
        action = None
        s_new = None
        done = False
        score = 0
        step = 0
        count = 1
        # current_state = env.problem.get_current_state()
        observation = None  # current_state.to_state()
        # while True:
        while not done:
            if observation is None:
                env.problem.plantsim.execute_simtalk("GetCurrentState")
                current_state = env.problem.get_current_state()
                observation = current_state.to_state()
            # if step > 4000:  # Reset des Agenten bei zu großer trajectory
            #    break
            # step = 0
            # count = 1
            # if load_checkpoint:
            #    agent.load_models()
            # env.reset()
            # current_state = env.problem.get_current_state()
            # observation = current_state.to_state()
            step += 1
            action, prob, val = agent.choose_action(observation)
            a = env.problem.actions[action]
            env.problem.act(a)
            # sleep(0.001)
            # env.problem.plantsim.execute_simtalk("GetCurrentState")
            current_state = env.problem.get_current_state()
            observation_ = current_state.to_state()
            reward = env.problem.get_reward(current_state)
            done = env.problem.is_goal_state(current_state)
            if reward > 0 and not done:
                count += reward
            # print("Step " + str(step) + ": " + a + " - Reward: " + str(reward) + " - finished: " + str(
            #    count - 1) + "\n")  # + " - " + str(round((step / count), 3)) +
            n_steps += 1
            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0 or done:
                print('---------learning--------')
                agent.learn()
                #if done:
                    #agent.memory.clear_memory()
                learn_iters += 1
            observation = observation_
            # print(observation)

        if done:
            performance_train.append(score)
            avg_score = np.mean(performance_train[-100:])

            with open("tmp\ppo_performance_train_16.txt", "w") as output:
                output.write(str(performance_train))

            if score > best_score:
                best_score = score
                if save_changes:
                    agent.save_models()

            '''if len(performance_train) > 0:
                if np.max(performance_train[-1]) < best_score:
                    #agent.memory.clear_memory()
                    agent.load_models()'''

            print('///Episode', i, '-- Return %.1f' % score, '-- avg Return %.1f' % avg_score,
                  '-- best score %.1f' % best_score,
                  '-- time_steps:', n_steps, '-- learning_steps:', learn_iters)
            print(
                '//////////////////////////////////////////////////////////////////////////////////////////////////// \n '
                '//////////////////////////////////////////////////////////////////////////////////////////////////// \n '
                '//////////////////////////////////////////////////////////////////////////////////////////////////// \n '
                '//////////////////////////////////////////////////////////////////////////////////////////////////// \n ')

        if done and i % 20 == 0:
            x = [i + 1 for i in range(len(performance_train))]
            plot_learning_curve(x, performance_train, figure_file)

    x = [i + 1 for i in range(len(performance_train))]
    plot_learning_curve(x, performance_train, figure_file)
