class EVCharger:
    def __init__(self, trained_model):
        self.trained_model = trained_model

    def charge_ev(self, state):
        action = self.trained_model.select_action(state)
        # Perform action (e.g., start charging)
        return action

    def update_model(self, state, action, reward, next_state):
        # Update the model (if necessary) based on feedback from the environment
        self.trained_model.update_Q(state, action, reward, next_state)

# Example usage:
# Load the trained model (trained_model is an instance of the DRLAgent class)
trained_model = DRLAgent(state_space, action_space, num_episodes, Tc_max, SoCf)
trained_model.train()

# Deploy the trained model in an EV charging station
ev_charger = EVCharger(trained_model)

# Example scenario:
incoming_ev_state = (50, 10, 30, 1)  # Example state of the incoming EV
action = ev_charger.charge_ev(incoming_ev_state)
print(f"Charging action: {action}")

# After observing the outcome, provide feedback to update the model
reward = calculate_reward(incoming_ev_state, action)  # Example reward calculation
next_state = update_state(incoming_ev_state, action)  # Example state transition
ev_charger.update_model(incoming_ev_state, action, reward, next_state)
