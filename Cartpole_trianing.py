#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().system('pip install stable-baselines3[extra]')


# In[2]:


#!get_ipython().system('pip install gym')


# In[3]


#get_ipython().system('pip show gym')


# In[4]:


import sys
sys.path.append('c:\\Users\\Hp EliteBook\\Desktop\\Reinforcement_learning')


# In[5]:


import os
import gym 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


# In[6]:


environment_name = "CartPole-v0"


# In[8]:


env = gym.make(environment_name, render_mode= "rgb_array")


# In[9]:


#this is just for us to understand the environment and how it works.
episodes = 5
for episode in range(1, episodes + 1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        result = env.step(action)  # Store all returned values in a single variable
        n_state, reward, done, _ = result[:4]  # Unpack the first four values
        score += reward
    
    print('Episode: {} Score: {}'.format(episode, score))

env.close()


# In[15]:


#Understanding the enviroment 
# 0-push cart to left, 1-push cart to the right
env.action_space.sample()


# In[16]:


#understanding the enviroment
# [cart position, cart velocity, pole angle, pole angular velocity]
env.observation_space.sample()


# In[43]:


#make directory
log_path=os.path.join('Training', 'Logs')


# In[44]:


log_path


# In[45]:


#Training a RL model
env = gym.make(environment_name) # calling the environemt
env = DummyVecEnv([lambda: env]) #wrapping the environment in a vectorized enviroment
model = PPO('MlpPolicy', env, verbose = 1, tensorboard_log=log_path) #Mlp -using a nn layers, multi layer perceptron, verbose is about logging the result


# In[46]:


#to know the parameters you can train on 
#get_ipython().run_line_magic('pinfo', 'PPO')


# In[47]:


#TO make the agent learn 
model.learn(total_timesteps=20000)


# In[54]:


#save and reload  model
PPO_path = os.path.join('Training', 'Saved Models', 'PPO_model')


# model.save(PPO_path)

# In[55]:


model.save(PPO_path)


# In[53]:
model=PPO.load(PPO_path, env=env)

evaluate_policy(model, env, n_eval_episodes=10, render=False) # the reward for the 10 episodes is 200


# In[37]:


env.close()
#Test model
obs = env.reset()
while True:
    action, _states = model.predict(obs)   #now using the model 
    obs, rewards, done, info = env.step(action)
    env.render()
    if done: 
        print('info', info)
        break
env.close()