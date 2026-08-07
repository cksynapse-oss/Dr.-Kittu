import numpy as np
from agent import HabitCoachAgent

if __name__ == "__main__":
    agent = HabitCoachAgent(lr_A=0.1, lr_B=0.1)

    # Simulate a "true" user state and generate observations
    n_steps = 24
    true_state = np.random.choice(agent.n_states)  # random start
    prev_qs = agent.D.copy()
    prev_action = 0  # do_nothing

    print("=== Active Inference Coach Simulation ===\n")
    for t in range(n_steps):
        # --- Generate observation from true state ---
        obs = {}
        if np.random.rand() < 0.9:  # 90% chance we get screen data
            p_screen = agent.A[0][true_state, :]
            obs['screen'] = np.random.choice(agent.n_screen, p=p_screen)
        else:
            obs['screen'] = None

        if np.random.rand() < 0.9:
            p_steps = agent.A[1][true_state, :]
            obs['steps'] = np.random.choice(agent.n_steps, p=p_steps)
        else:
            obs['steps'] = None

        if np.random.rand() < 0.3:  # self-report only 30% of the time
            p_mood = agent.A[2][true_state, :]
            obs['mood'] = np.random.choice(agent.n_mood, p=p_mood)
        else:
            obs['mood'] = None

        # --- Agent perceives and plans ---
        action = agent.act(obs)

        # --- Learning from the transition ---
        if t > 0:
            agent.learn(obs, prev_qs, prev_action)

        # --- Update true state according to action taken ---
        trans_probs = agent.B[action][true_state, :]
        true_state = np.random.choice(agent.n_states, p=trans_probs)

        # --- Logging ---
        belief_str = ", ".join([f"{n}:{agent.qs[i]:.2f}" for i,n in enumerate(agent.state_names)])
        obs_str = ""
        if obs['screen'] is not None:
            obs_str += f"scr={agent.obs_screen[obs['screen']]}, "
        if obs['steps'] is not None:
            obs_str += f"stp={agent.obs_steps[obs['steps']]}, "
        if obs['mood'] is not None:
            obs_str += f"mood={agent.obs_mood[obs['mood']]}"
        else:
            obs_str += "mood=missing"

        print(f"t={t:2d} | true={agent.state_names[true_state]:8s} | action={agent.action_names[action]:12s} | belief: {belief_str}")
        print(f"      | obs: {obs_str}\n")

        # Save for next iteration
        prev_qs = agent.qs.copy()
        prev_action = action

    print("Simulation finished.")
    # Example save
    agent.save("coach_model.npz")
