# dqn-agent
### The following explains each line of code in the ev_charging_environment.py file

### Imports and Environment Registration

```python
import gym
from gym import spaces
import numpy as np
import math

class EVChargingEnv(gym.Env):
    def __init__(self):
        super(EVChargingEnv, self).__init__()
```

1. **Imports**: 
    - `gym` and `spaces` are imported from the OpenAI Gym library, which is used to create and manage environments for reinforcement learning.
    - `numpy` (imported as `np`) is used for numerical operations.
    - `math` is imported but not used in the provided code.

2. **Class Definition and Initialization**:
    - A new class `EVChargingEnv` is defined, which inherits from `gym.Env`.
    - The `__init__` method initializes the environment by calling the parent class initializer `super(EVChargingEnv, self).__init__()` and then setting up the environment.

### Environment Initialization

```python
        self.state_size = 4
        self.action_size = 3
        self.total_waiting_time = 0.0
        self.discount_per_hour = 0.005
        self.max_discount = 0.5
```

3. **State and Action Size**:
    - `self.state_size = 4`: The state size is 4, indicating that the state is represented by 4 variables.
    - `self.action_size = 3`: The action size is 3, indicating that there are 3 possible actions: idle (0), conventional charging (1), and fast charging (2).

4. **Discount Factors**:
    - `self.total_waiting_time`: Keeps track of the total waiting time.
    - `self.discount_per_hour`: The discount factor per hour of waiting.
    - `self.max_discount`: The maximum discount that can be applied.

### Action and Observation Space

```python
        self.action_space = spaces.Discrete(self.action_size)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)
        self.availability_index = np.random.rand(24)
```

5. **Action Space**:
    - `self.action_space = spaces.Discrete(self.action_size)`: Defines a discrete action space with 3 possible actions.

6. **Observation Space**:
    - `self.observation_space = spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)`: Defines a continuous observation space with values ranging from 0 to 1, and the shape is defined by `self.state_size` (4).

7. **Availability Index**:
    - `self.availability_index = np.random.rand(24)`: Initializes an array with 24 random values representing the availability index for each hour of the day.

### State Initialization

```python
        self.state = None
        self.reset()
```

8. **State Initialization**:
    - `self.state`: The initial state is set to `None`.
    - `self.reset()`: Calls the `reset` method to initialize the state.

### Reset Method

```python
    def reset(self):
        SoC = np.random.rand()
        hour = np.random.randint(0, 24)
        minute = np.random.randint(0, 60)
        charging_mode = 0
        self.availability_index = np.random.rand(24)
        self.state = np.array([SoC, hour/24.0, minute/60.0, charging_mode])
        return self.state
```

9. **Reset Method**:
    - `SoC = np.random.rand()`: Initializes the state of charge (SoC) with a random value between 0 and 1.
    - `hour = np.random.randint(0, 24)`: Initializes the hour with a random integer between 0 and 23.
    - `minute = np.random.randint(0, 60)`: Initializes the minute with a random integer between 0 and 59.
    - `charging_mode = 0`: Initializes the charging mode to 0 (idle).
    - `self.availability_index = np.random.rand(24)`: Resets the availability index with new random values.
    - `self.state = np.array([SoC, hour/24.0, minute/60.0, charging_mode])`: Constructs the state array by normalizing hour and minute values.
    - `return self.state`: Returns the initial state.

### Discount Factor Calculation

```python
    def calculate_discount_factor(self):
        if 1 - self.discount_per_hour * self.total_waiting_time > 1:
            return 1.0
        elif 1 - self.discount_per_hour * self.total_waiting_time < self.max_discount:
            return self.max_discount
        else:
            return 1 - self.discount_per_hour * self.total_waiting_time
```

10. **Calculate Discount Factor**:
    - This method calculates a discount factor based on the total waiting time, ensuring it stays between 1 and `self.max_discount`.

### Step Method

```python
    def step(self, action):
        SoC, hour, minute, charging_mode = self.state
        tp = 0.25   # time period
        P_c = 7     # conventional charging power
        P_f = 22    # fast charging power
        Q_b = 22    # battery capacity
```

11. **Step Method**:
    - The `step` method takes an action as input and updates the state accordingly.

12. **State Variables**:
    - `SoC, hour, minute, charging_mode = self.state`: Unpacks the current state into individual variables.
    - `tp = 0.25`: Defines the time period.
    - `P_c = 7`: Defines the conventional charging power.
    - `P_f = 22`: Defines the fast charging power.
    - `Q_b = 22`: Defines the battery capacity.

### Action Application and State Update

```python
        if action == 0:  # Idle
            charging_mode = 0
        elif action == 1:  # Conventional charging
            charging_mode = 1
            SoC += (tp * P_c)/(100 * Q_b)
        elif action == 2:  # Fast charging
            charging_mode = 2
            SoC += (tp * P_f)/(100 * Q_b)
```

13. **Action Application**:
    - Updates the `charging_mode` based on the action.
    - If `action` is 1 or 2, it also updates the `SoC` based on the charging power and battery capacity.

### Time Progression

```python
        minute *= 60
        hour *= 24
        minute += 15
        if minute >= 60:
            minute = 0
            hour += 1
        if hour >= 24:
            hour = 0
```

14. **Time Update**:
    - Converts `hour` and `minute` to their original scales, adds 15 minutes, and then normalizes back to the 0-1 range.

### State Update

```python
        normalized_charging_mode = charging_mode / 2.0
        self.state = np.array([SoC, hour / 24.0, minute / 60.0, normalized_charging_mode])
```

15. **State Update**:
    - Updates the state array with the new `SoC`, `hour`, `minute`, and `normalized_charging_mode`.

### Reward Calculation

```python
        C_t = [[0, 4.6], [1, 4.5], [2, 4.5], [3, 4.4], [4, 4.5], [5, 4.6], [6, 4.6], [7, 4.7],
                         [8, 4.9], [9, 4.9], [10, 5.0],[11, 4.9], [12, 5.0], [13, 6.9], [14, 6.6], [15, 7.2],
                         [16, 7.2], [17, 8.1], [18, 10.6], [19, 8.4], [20, 6.9], [21, 6.7], [22, 5.7], [23, 5.0]]

        Wi = np.random.normal(0.05, 0.0075)     # idle waiting cost
        Wc = 0.5 * Wi   # charging waiting cost
        k = self.calculate_discount_factor()
```

16. **Reward Calculation Variables**:
    - `C_t`: Time-of-use tariff rates for each hour of the day.
    - `Wi`: Randomly generated idle waiting cost.
    - `Wc`: Charging waiting cost.
    - `k`: Discount factor calculated by `calculate_discount_factor`.

### Reward Calculation Based on Action

```python
        Ux = np.interp(self.availability_index[int(hour)], [0, 0.25, 0.5, 0.75, 1], [-2, -1, 0, 1, 2])

        if charging_mode == 0:  # Idle
            r1 = -k * (tp * P_c * C_t[int(hour)][1] + tp * Wc + tp * Wi)
            reward = r1  # Idle waiting cost

        elif charging_mode == 1:  # Conventional charging
            r1 = -(tp * P_c * C_t[int(hour)][1] +

 tp * Wc)
            r2 = Ux
            reward = r1 + r2  # Charging cost
        elif charging_mode == 2:  # Fast charging
            r1 = -(tp * P_f * C_t[int(hour)][1] + tp * Wc)
            r2 = Ux
            reward = r1 + r2  # Higher charging cost
```

17. **Utility Factor**:
    - `Ux`: Interpolated utility factor based on the availability index.

18. **Reward Calculation**:
    - Calculates the reward based on the current `charging_mode`.
    - For idle, it includes waiting costs.
    - For conventional and fast charging, it includes charging costs and the utility factor.

### Penalty for Overcharging

```python
        if SoC > 1.0:
            SoC = 1.0
            reward -= 1.0
```

19. **Overcharging Penalty**:
    - If `SoC` exceeds 1.0, it is capped at 1.0 and a penalty is applied to the reward.

### Check for Termination

```python
        done = SoC >= 1.0  # Done if fully charged
        self.total_waiting_time += tp
        
        return self.state, reward, done, {}
```

20. **Check for Termination**:
    - The episode is done if `SoC` is fully charged (`SoC >= 1.0`).

21. **Update Waiting Time**:
    - Increments `self.total_waiting_time` by `tp`.

22. **Return Statement**:
    - Returns the updated state, reward, done flag, and an empty dictionary (info).

### Render Method

```python
    def render(self, mode='human', close=False):
        SoC, hour, minute, charging_mode = self.state
        availability = self.availability_index[int(hour * 24)]
        print(f"State of Charge: {SoC:.2f}, Time: {int(hour * 24)}:{int(minute * 60)}, Charging Mode: {charging_mode * 2}, index: {availability}")
```

23. **Render Method**:
    - Displays the current state, including `SoC`, time, charging mode, and availability index.

### Environment Registration

```python
# Register the environment
gym.envs.registration.register(
    id='EVCharging-v0',
    entry_point=EVChargingEnv,
)
```

24. **Environment Registration**:
    - Registers the `EVChargingEnv` environment with Gym, making it available for use by specifying its ID and entry point.





## The following explains the dqn-agent.py file:

### Imports and Environment Registration

```python
import random
import os
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

from gym.envs.registration import register

register(
    id='EVCharging-v0',
    entry_point='ev_charging_env:EVChargingEnv',
)
```

1. **Imports**: 
    - Various libraries are imported:
        - `random`, `os`, and `numpy` for random number generation, file operations, and numerical operations, respectively.
        - `gym` for reinforcement learning environment.
        - `deque` from `collections` for memory buffer.
        - `tensorflow` and related modules for neural network operations.

2. **TensorFlow Logging Configuration**:
    - Configures TensorFlow to reduce logging verbosity to minimize unnecessary log messages.

3. **Environment Registration**:
    - Registers the custom `EVChargingEnv` environment with Gym.

### Constants and Agent Class

```python
EPISODES = 1000

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.last_action = None
```

4. **Constants**:
    - `EPISODES = 1000`: Defines the number of training episodes.

5. **DQNAgent Class Initialization**:
    - Initializes the DQN agent with several attributes:
        - `state_size` and `action_size` define the size of the state and action spaces.
        - `memory` is a deque used to store experience tuples.
        - `gamma` is the discount factor.
        - `epsilon` is the exploration rate, with `epsilon_min` and `epsilon_decay` controlling its decay.
        - `learning_rate` for the neural network optimizer.
        - `model` is the neural network model created by `_build_model`.
        - `last_action` stores the last action taken by the agent.

### Neural Network Model

```python
    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
```

6. **Model Definition**:
    - Defines a simple feedforward neural network with three layers:
        - Input layer with 24 neurons and ReLU activation.
        - Hidden layer with 24 neurons and ReLU activation.
        - Output layer with `action_size` neurons and linear activation.
    - Compiles the model with mean squared error loss and Adam optimizer.

### Memory and Action Selection

```python
    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if self.last_action == 0:
            action = 1
        elif np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
        else:
            act_values = self.model.predict(state)
            action = np.argmax(act_values[0])
        self.last_action = action
        return action
```

7. **Memory Storage**:
    - `memorize` method stores experience tuples in the memory buffer.

8. **Action Selection**:
    - `act` method determines the action based on:
        - If the last action was idle, the next action must be conventional charging.
        - If a random number is less than `epsilon`, it selects a random action (exploration).
        - Otherwise, it uses the model to predict Q-values and selects the action with the highest Q-value (exploitation).

### Training and Replay

```python
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
```

9. **Replay Memory**:
    - `replay` method samples a minibatch from memory and updates the Q-values using the Bellman equation.
    - The target value is the reward plus the discounted maximum future reward, unless the episode is done.
    - The model is trained on the target values for the selected actions.
    - `epsilon` is decayed after each training step.

### Saving and Loading Weights

```python
    def load(self, name):
        if os.path.exists(name):
            self.model.load_weights(name)
        else:
            print(f"File {name} does not exist.")

    def save(self, name):
        self.model.save_weights(name)
```

10. **Model Weights Management**:
    - `load` method loads model weights from a file if it exists.
    - `save` method saves model weights to a file.

### Main Training Loop

```python
if __name__ == "__main__":
    env = gym.make('EVCharging-v0')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    agent.load("./save/evcharging-dqn.weights.h5")
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        cumulative_reward = 0
        action_counts = np.zeros(action_size)

        for time in range(1440):  # 1440 minutes in a day
            env.render()
            action = agent.act(state)
            action_counts[action] += 1
            next_state, reward, done, _ = env.step(action)
            cumulative_reward += reward
            print(f"Each reward: {reward}")
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.memorize(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{EPISODES}, score: {time}, e: {agent.epsilon:.2}, Cumulative Reward: {cumulative_reward:.2f}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        print(f"Episode: {e} Action Distribution: {action_counts}")
        if e % 10 == 0:
            agent.save("./save/evcharging-dqn.weights.h5")
            print("Saved")
```

11. **Main Loop**:
    - Creates the environment and initializes the agent.
    - Loads existing weights if available.
    - Runs for `EPISODES` number of episodes.

12. **Episode Loop**:
    - Resets the environment and reshapes the state.
    - Tracks cumulative reward and action counts.
    - For each time step in a day (1440 minutes):
        - Renders the environment.
        - Selects and performs an action.
        - Updates the cumulative reward.
        - Stores the experience in memory.
        - Checks for the end of the episode.
        - Trains the agent using experience replay if memory size exceeds the batch size.

13. **Periodic Saving**:
    - Saves model weights every 10 episodes.

This code implements a DQN agent to learn optimal charging strategies in an EV charging environment by interacting with the environment, storing experiences, and training a neural network to predict Q-values.
