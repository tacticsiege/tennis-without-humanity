import numpy as np
import time
import os
from collections import namedtuple, deque

import torch

from agent_utils import save_ddpg

def train_ddpg(env, agent, num_agents,
                n_episodes=2000, max_t=1000, 
                print_every=5, goal_score=30, score_window_size=100,
                keep_training=False):
    is_solved = False
    total_scores = []
    total_scores_deque = deque(maxlen=score_window_size)
    steps_for_eps = []
    steps_deque = deque(maxlen=int(score_window_size/2))
    t_step = 0

    # save parameters before starting training
    save_ddpg(agent.name, params=agent.params, verbose=True)
    
    print(f'\r\nTraining started for \'{agent.name}\'...')
    training_start_time = time.time()
    for i_episode in range(1, n_episodes+1):
        # Reset Env and Agent
        env_info = env.reset(train_mode=True)[agent.brain_name]     # reset the environment
        states = env_info.vector_observations                       # get the current state (for each agent)
        scores = np.zeros(num_agents)                               # initialize the score (for each agent)
        agent.reset()
        
        start_time = time.time()
        steps = 0
        for t in range(max_t):
            t_step += 1
            actions = agent.act(states)
            
            env_info = env.step(actions)[agent.brain_name]      # send all actions to the environment
            next_states = env_info.vector_observations          # get next state (for each agent)
            rewards = env_info.rewards                          # get reward (for each agent)
            dones = env_info.local_done                         # see if episode finished
            
            for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                agent.step(state, action, reward, next_state, done) # send actions to the agent
            
            scores += env_info.rewards                          # update the score (for each agent)
            states = next_states                                # roll over states to next time step
            
            agent.start_learn(t_step)                              # start learning, if conditions are met
            steps += 1
            if np.any(dones):                                   # exit loop if episode finished
                break

        # track progress
        steps_for_eps.append(steps)
        steps_deque.append(steps)
        mean_score = np.mean(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        total_scores_deque.append(max_score)
        total_scores.append(max_score)
        total_average_score = np.mean(total_scores_deque)
        duration = time.time() - start_time

        if i_episode % print_every == 0:
            save_ddpg(agent.name, 
                actor_weights=agent.actor_local.state_dict(), 
                critic_weights=agent.critic_local.state_dict(),
                scores=total_scores, i_checkpoint=i_episode)

            recent_steps = np.mean(steps_deque)
            print(
                '\rEpisode {}\tAvg. Score: {:.2f}\tAvg. Steps: {:.2f}\tTotal Time: {:.3f}m'
                .format(i_episode, total_average_score, recent_steps, (time.time()-training_start_time)/60))
        else:
            print(
                '\rEpisode {}\tMean: {:.2f}, Max: {:.2f}, Steps: {}, Time: {:.2f}s'
                .format(i_episode, mean_score, max_score, steps, duration), end='')
            
        if total_average_score >= goal_score and i_episode >= score_window_size and not is_solved:
            is_solved = True
            print(
                '\r\nEnvironment solved in {} episodes! Total Average score: {:.2f}'.format(
                i_episode, total_average_score))
            print('\rTotal Duration: {:.2f}m\n'.format(
                (time.time() - training_start_time)/ 60.0))
            
            save_ddpg(agent.name, 
                actor_weights=agent.actor_local.state_dict(), 
                critic_weights=agent.critic_local.state_dict(),
                scores=total_scores, verbose=True)
            
            if keep_training:
                print('\r\nContinuing training...')
            else:
                return total_scores

        # bail if we're halfway through eps with barely any learning
        if total_average_score < goal_score * 0.15 and i_episode > int(n_episodes/2) and not keep_training:
            print("\r\nAgent barely learned anything halfway through training, let's bail!\n")
            save_ddpg(agent.name, 
                actor_weights=agent.actor_local.state_dict(), 
                critic_weights=agent.critic_local.state_dict(),
                scores=total_scores, verbose=True)
            return total_scores
        
        # bail if we're 1/3 through and stuck at 0
        if total_average_score < 0.001 and i_episode > int(n_episodes/3) and not keep_training:
            print("\r\nAgent stuck at 0 after 1/3 training, bail!\n")
            save_ddpg(agent.name, 
                actor_weights=agent.actor_local.state_dict(), 
                critic_weights=agent.critic_local.state_dict(),
                scores=total_scores, verbose=True)
            return total_scores
    
    # finished all episodes
    print('\r\nCompleted training on {} episodes.'.format(n_episodes))
    print('\rAverage Score for last {} episodes: {:.2f}\tGoal: {}'.format(
        score_window_size, np.mean(total_scores_deque), goal_score))
    print('\rTotal Duration: {:.2f}m\n'.format((time.time() - training_start_time)/ 60.0))

    save_ddpg(agent.name, 
        actor_weights=agent.actor_local.state_dict(), 
        critic_weights=agent.critic_local.state_dict(),
        scores=total_scores)

    return total_scores
