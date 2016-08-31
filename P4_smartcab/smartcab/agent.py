import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import itertools

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.list_of_actions = [ None, 'forward', 'left', 'right'] 
        self.state = [] #we will use a list of tuples of states to map the indexes
        self.next_waypoint = None
        self.number_of_states= len(self.list_of_actions)*2*2*2*2# 4 waypoints or actions * 2 light states * 2 oncoming  traffic states * 2 left traffic states  * 2 right  traffic states 
        self.R = np.zeros(shape=[self.number_of_states, 4]) #we have 4 actions
        self.q = np.zeros(shape=[self.number_of_states, 4])
        #self.q = np.empty(shape=[self.number_of_states,self.number_of_states])
        self.state_counter = 0
        self.gamma = .8 #random value
		
    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.color = 'red'  # override color
        #self.state = []
        self.next_waypoint = None
        
    def update(self, t):
		# Gather inputs
		self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
		inputs = self.env.sense(self)
		deadline = self.env.get_deadline(self)
		
		# TODO: Update state
		this_state= (inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'], self.next_waypoint)
		
		if not this_state in self.state:
			self.state.append(this_state)
		print self.state
			
		#if not self.state  or not (inputs) in self.state: # if dictionary if empty : initialize it, if it has not the state: add it and increase the counter.
			#self.state[(inputs)] = self.state_counter
			#self.state_counter += 1
		
		# TODO: Select action according to your policy
		action = random.choice(self.list_of_actions)
		
		# Execute action and get reward
		reward = self.env.act(self, action)
		#self.R.resize(self.R.shape[0]+1, self.R.shape[1]+1)
		self.R[self.state.index(this_state)][self.list_of_actions.index(action)] = reward
		#print self.R
		# TODO: Learn policy based on state, action, reward
		
		#Q = reward + Gamma * np.max[[ in list_of_actions]]	
		#print self.q
		temp = []
		for (next_state,next_action) in list(itertools.product(range(self.number_of_states),self.list_of_actions)):
			temp.append( self.q[next_state][self.list_of_actions.index(next_action)]) 
		self.q[self.state.index(this_state)][self.list_of_actions.index(action)] = reward + self.gamma* np.max(temp)
				# we iterate over the possible combinations of states and actions, since any future combination is possible 
		print self.q[range(len(self.state))][:]
		print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
