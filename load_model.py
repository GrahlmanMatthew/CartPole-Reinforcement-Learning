import os
import sys
import gym
from stable_baselines3 import A2C, PPO, DQN 

modelPath = os.path.join(os.getcwd(), 'models')

algorithms = ["A2C", "PPO", "DQN"]
modelType = None

if len(sys.argv) == 3 and sys.argv[1] in algorithms and os.path.exists(os.path.join(modelPath, sys.argv[1], sys.argv[2])):
    modelType = sys.argv[1]
    modelPath = os.path.join(modelPath, modelType, sys.argv[2])
else:
    raise Exception("ERROR: missing arguments! Please specify the algorithm then the model file (e.g. PPO 20000.zip)")
   
env = gym.make('CartPole-v1')    # continuous
env.reset()

# Changing model type based on user input
model = None
if modelType == "PPO":  model = PPO.load(modelPath, env=env)
if modelType == "A2C":  model = A2C.load(modelPath, env=env)
if modelType == "DQN":  model = DQN.load(modelPath, env=env)
   
# Run 5 simulations of the model 
for i in range(1, 5):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

env.close()