class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.global_states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.global_states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def __len__(self):
        return len(self.rewards)