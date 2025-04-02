import json
import os
import random

from .state import State


class Q_State(State):
    '''Augments the game state with Q-learning information'''

    def __init__(self, string):
        super().__init__(string)

        # key stores the state's key string (see notes in _compute_key())
        self.key = self._compute_key()

    def _compute_key(self):
        '''
        Returns a key used to index this state.

        The key should reduce the entire game state to something much smaller
        that can be used for learning. When implementing a Q table as a
        dictionary, this key is used for accessing the Q values for this
        state within the dictionary.
        '''

        # this simple key uses the 3 object characters above the frog
        # and combines them into a key string
        return ''.join([
            self.get(self.frog_x - 1, self.frog_y - 1) or '_',
            self.get(self.frog_x - 1, self.frog_y) or '_',
            self.get(self.frog_x, self.frog_y - 1) or '_',
            self.get(self.frog_x, self.frog_y + 1) or '_',
            self.get(self.frog_x + 1, self.frog_y) or '_',
            self.get(self.frog_x + 1, self.frog_y - 1) or '_',
        ])

    def reward(self):
        '''Returns a reward value for the state.'''

        if self.at_goal:
            return self.score
        elif self.is_done:
            return -10
        else:
            return 0


class Agent:

    def __init__(self, train=None):
        # train is either a string denoting the name of the saved
        # Q-table file, or None if running without training
        self.train = train

        # q is the dictionary representing the Q-table
        self.q = {}

        # name is the Q-table filename
        # (you likely don't need to use or change this)
        self.name = train or 'q'

        # path is the path to the Q-table file
        # (you likely don't need to use or change this)
        self.path = os.path.join(os.path.dirname(
            os.path.realpath(__file__)), 'train', self.name + '.json')

        self.load()
        
        self.prev_s = None
        self.prev_a = None

    def load(self):
        '''Loads the Q-table from the JSON file'''
        try:
            with open(self.path, 'r') as f:
                self.q = json.load(f)
            if self.train:
                print('Training {}'.format(self.path))
            else:
                print('Loaded {}'.format(self.path))
        except IOError:
            if self.train:
                print('Training {}'.format(self.path))
            else:
                raise Exception('File does not exist: {}'.format(self.path))
        return self

    def save(self):
        '''Saves the Q-table to the JSON file'''
        with open(self.path, 'w') as f:
            json.dump(self.q, f)
        return self

    def choose_action(self, state_string):
        '''
        Returns the action to perform.

        This is the main method that interacts with the game interface:
        given a state string, it should return the action to be taken
        by the agent.
        '''

        ALPHA = 0.1
        GAMMA = 0.9
        EPSILON = 0.1
        
        s = Q_State(state_string)
        
        # https://stackoverflow.com/questions/3203099/percentage-chance-to-make-action
        if random.random() < EPSILON or s.key not in self.q or not self.q[s.key]:
            a = random.choice(State.ACTIONS)
        else:

            action_to_take = list(self.q[s.key].keys())[0]
            max_qvalue = float('-inf')

            for action, value in self.q[s.key].items():
                if value > max_qvalue:
                    max_qvalue = value
                    action_to_take = action

            a = action_to_take
        
        if self.train:

            if self.prev_s is None:
                if s.key not in self.q:
                        self.q[s.key] = {'u': 0, 'd': 0, 'l': 0, 'r': 0, '_': 0}
            else:
                if self.prev_s.key not in self.q:
                        self.q[self.prev_s.key] = {'u': 0, 'd': 0, 'l': 0, 'r': 0, '_': 0}
                if s.key not in self.q:
                        self.q[s.key] = {'u': 0, 'd': 0, 'l': 0, 'r': 0, '_': 0}
            
                # Bellman Equation
                self.q[self.prev_s.key][self.prev_a] = ((1 - ALPHA) * self.q[self.prev_s.key][self.prev_a]) + (ALPHA * (s.reward() + GAMMA * max(self.q[s.key].values())))
                
            self.save()

        self.prev_s, self.prev_a = s, a

        return a
