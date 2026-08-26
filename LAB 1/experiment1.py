import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

def print_section_header(title):
    width = 95
    print("=" * width)
    print(f"{title.center(width)}")
    print("=" * width)

def print_step_header():
    header = f"{'Step':^6} | {'Selected Action':^18} | {'Reward':^8} | {'Cum. Reward':^12} | {'Status':^12} | {'Cart Pos':^10} | {'Cart Vel':^10} | {'Pole Ang':^10} | {'Pole Vel':^10}"
    divider = "-" * len(header)
    print(divider)
    print(header)
    print(divider)
    return divider

def main():
  
    print("TASK 1: ENVIRONMENT SETUP & SYSTEM CHECK")
    print(f"  [+] Gymnasium Version  : {gym.__version__}")
    print(f"  [+] NumPy Version      : {np.__version__}")
    print(f"  [+] Matplotlib Version : {plt.matplotlib.__version__}")
    print("  [+] System Status      : Successfully verified libraries and system environment.\n")


    print("TASK 2: CREATE AND INITIALIZE RL ENVIRONMENT")
    env = gym.make("CartPole-v1")
    observation, info = env.reset(seed=42)
    
    print(f"  [+] Created Environment : CartPole-v1")
    print(f"  [+] Initial State Vector: {observation}")
    print("  [+] State Vector Components Breakdown:")
    print(f"      - [0] Cart Position     : {observation[0]:+.4f} (meters)")
    print(f"      - [1] Cart Velocity     : {observation[1]:+.4f} (m/s)")
    print(f"      - [2] Pole Angle        : {observation[2]:+.4f} rad ({np.degrees(observation[2]):+.2f} deg)")
    print(f"      - [3] Pole Velocity Tip : {observation[3]:+.4f} (rad/s)")
    print(f"  [+] Environment Info    : {info}\n")

    print("TASK 3: EXPLORE OBSERVATION AND ACTION SPACES")
    print(f"  [+] Observation Space Structure:")
    print(f"      - Space Type       : {type(env.observation_space).__name__}")
    print(f"      - Space Dimensions : {env.observation_space.shape}")
    print(f"      - Data Type        : {env.observation_space.dtype}")
    print(f"      - Bounds (Min/Max) :")
    print(f"          Cart Position  : [{env.observation_space.low[0]:.2f}, {env.observation_space.high[0]:.2f}]")
    print(f"          Cart Velocity  : [{env.observation_space.low[1]}, {env.observation_space.high[1]}]")
    print(f"          Pole Angle     : [{env.observation_space.low[2]:.4f}, {env.observation_space.high[2]:.4f}] rad")
    print(f"          Pole Velocity  : [{env.observation_space.low[3]}, {env.observation_space.high[3]}]")
    print(f"  [+] Action Space Structure:")
    print(f"      - Space Type       : {type(env.action_space).__name__}")
    print(f"      - Total Actions    : {env.action_space.n}")
    print(f"      - Action Mapping   : Action 0 -> Push Cart Left | Action 1 -> Push Cart Right\n")


    print("TASK 4: STEP-BY-STEP SIMULATION OF RANDOM AGENT")
    divider = print_step_header()

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
            
        action_desc = f"{action} ({'Push Right' if action == 1 else 'Push Left'})"
        
        cart_pos = f"{observation[0]:+.4f}"
        cart_vel = f"{observation[1]:+.4f}"
        pole_ang = f"{observation[2]:+.4f}"
        pole_vel = f"{observation[3]:+.4f}"
        
        print(f"{step_count:^6} | {action_desc:^18} | {reward:^8.1f} | {cumulative_reward:^12.1f} | {status_text:^12} | {cart_pos:^10} | {cart_vel:^10} | {pole_ang:^10} | {pole_vel:^10}")

    print(divider)
    env.close()

    print("EPISODE SUMMARY & METRICS")
    print(f"  [+] Total Steps Executed : {step_count}")
    print(f"  [+] Total Return (Reward): {cumulative_reward}")
    term_reason = "Pole Angle exceeded +/-12 deg or Cart Position exceeded +/-2.4" if terminated else "Reached maximum step limit (500 steps)"
    print(f"  [+] Termination Reason   : {term_reason}")
    print("=" * 95 + "\n")

if __name__ == "__main__":
    main()