import os
import sys
import gym
from stable_baselines3 import A2C, PPO, DQN

epochs = 25     # epochs * 10,000 timesteps per =250,000 (default)
if len(sys.argv) >= 2:
    if int(sys.argv[1]) >= 1 and int(sys.argv[1]) <= 100:
        epochs = int(sys.argv[1]) + 1
    else:
        raise Exception("ERROR: invalid # of epochs provided! (1 >= epochs <= 100")
        sys.exit(1)

logsPath = os.path.join(os.getcwd(), 'logs')
ppoModelsPath = os.path.join(os.getcwd(), 'models', 'PPO')
a2cModelsPath = os.path.join(os.getcwd(), 'models', 'A2C')
dqnModelsPath = os.path.join(os.getcwd(), 'models', 'DQN')

if not os.path.exists(logsPath):    os.makedirs(logsPath)
if not os.path.exists(ppoModelsPath):   os.makedirs(ppoModelsPath)
if not os.path.exists(a2cModelsPath):   os.makedirs(a2cModelsPath)
if not os.path.exists(dqnModelsPath):   os.makedirs(dqnModelsPath)
     
env = gym.make('CartPole-v1')    # continuous
env.reset()

ppoModel = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logsPath)   # Proximal Policy Optimization (Default for OpenAI)
a2cModel = A2C('MlpPolicy', env, verbose=1, tensorboard_log=logsPath)   # A2C, or Advantage Actor Critic
dqnModel = DQN('MlpPolicy', env, verbose=1, tensorboard_log=logsPath)   # DQN, or Deep Q Learning

timesteps = 10000
for i in range(1, epochs):
    ppoModel.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="PPO")
    ppoModel.save("%s/%s" % (ppoModelsPath, timesteps*i))
    
    a2cModel.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="A2C")
    a2cModel.save("%s/%s" % (a2cModelsPath, timesteps*i))
    
    dqnModel.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="DQN")
    dqnModel.save("%s/%s" % (dqnModelsPath, timesteps*i))

env.close()