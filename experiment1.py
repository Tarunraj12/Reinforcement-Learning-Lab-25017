import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

print("TASK 1: ENVIRONMENT SETUP")
print(f"Gymnasium Version : {gym.__version__}")
print(f"NumPy Version     : {np.__version__}")
print(f"Matplotlib Version: {plt.matplotlib.__version__}")
print("Status            : Successfully installed and imported libraries.\n")

print("TASK 2: CREATE AND INITIALIZE RL ENVIRONMENT")
env = gym.make("CartPole-v1")
observation, info = env.reset()

print("Created Environment : CartPole-v1")
print(f"Initial Observation : {observation}")
print(f"Environment Info    : {info}\n")

print("TASK 3: EXPLORE OBSERVATION AND ACTION SPACES")
print(f"Observation Space          : {env.observation_space}")
print(f"Action Space               : {env.action_space}")
print(f"Type of Observation Space  : {type(env.observation_space)}")
print(f"Number of Possible Actions : {env.action_space.n}\n")

print("TASK 4: EXECUTE A RANDOM AGENT")


header = f"{'Step':^6} | {'Selected Action':^17} | {'Reward':^8} | {'Episode Status':^16} | {'Observation':^45}"
divider = "-" * len(header)

print(divider)
print(header)
print(divider)

done = False
step_count = 0
cumulative_reward = 0.0

while not done:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    
    step_count += 1
    cumulative_reward += reward
    
    if terminated:
        status_text = "Terminated"
    elif truncated:
        status_text = "Truncated"
    else:
        status_text = "In Progress"
        
    obs_formatted = "[" + ", ".join(f"{x:+.4f}" for x in observation) + "]"
    action_desc = f"{action} ({'Push Right' if action == 1 else 'Push Left'})"
    
    print(f"{step_count:^6} | {action_desc:^17} | {reward:^8.1f} | {status_text:^16} | {obs_formatted:^45}")

print(divider)

env.close()

print("\n--- EPISODE SUMMARY ---")
print(f"Total Number of Steps : {step_count}")
print(f"Cumulative Reward     : {cumulative_reward}")