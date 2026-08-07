# Dr. Kittu: Personal Adaptive Habit & Mood Coach

A privacy-first, on-device mobile app backend that learns your daily rhythms using Active Inference. It infers your hidden mental/physical state from phone sensor data and occasional check-ins, and then chooses tiny, well-timed interventions to gently steer you toward your preferred well-being state.

## Why Active Inference?
- **Closed loop**: Observes (screen time, step count, mood), infers hidden states (stressed, anxious, focused, tired, calm), and acts (suggests interventions).
- **Sample efficient**: Learns a useful model of your patterns in days, not months.
- **Uncertainty-aware**: Explores gently when unsure (e.g. epistemic actions).
- **Privacy by design**: Runs entirely on-device with a tiny model footprint.

## Tech Stack
- **Python 3.9+**
- **pymdp**: Active inference library for discrete state spaces
- **NumPy**: Matrix operations

## Getting Started (Simulation)

To test the core perception-action loop with a simulated user:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the simulation:
   ```bash
   python simulate.py
   ```
   This will run a 24-step simulation, logging the agent's beliefs, actions, and observations. The learned model is saved to `coach_model.npz`.

## Project Structure

- `agent.py`: The core Active Inference agent (`HabitCoachAgent`). Handles perception, planning, and learning using Dirichlet updates.
- `simulate.py`: A self-contained script to simulate a day with the agent, generating synthetic sensor readings.
- `requirements.txt`: Python dependencies.

## Next Steps for Android Integration

To deploy this agent into an Android app, you can use **Chaquopy** to embed Python natively:

1. **Setup Chaquopy** in your Android Studio project to include Python support.
2. **Copy `agent.py`** into `app/src/main/python/`.
3. **Collect Sensor Data (Kotlin side)**: Use `UsageStatsManager` for screen time and `Health Connect` for steps.
4. **Foreground Service**: Create a Kotlin service that periodically (e.g., every 15 mins) collects sensor data, passes it as a Python dictionary to `agent.act()`, and fires a notification if an intervention is suggested.
