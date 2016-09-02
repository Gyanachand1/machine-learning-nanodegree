import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import itertools
import pdb # for debugging
import os 
import os.path
import csv
import pandas as pd

class LearningAgent(Agent):
	"""An agent that learns to drive in the smartcab world."""

	def __init__(self, env):
		super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
		self.color = 'red'  # override color
		self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
		# TODO: Initialize any additional variables here
		self.valid_actions = [ None, 'forward', 'left', 'right'] 
		self.state = [] #we will use a list of tuples of states to map the indexes
		self.list_of_states = []
		self.next_waypoint = None
		self.number_of_states= 5
		self.R = np.zeros(shape=[self.number_of_states, 4]) #we have 4 actions
		self.q = np.zeros(shape=[self.number_of_states, 4])
		self.policy = np.zeros([self.number_of_states, 1], dtype = int)
		#self.q = np.empty(shape=[self.number_of_states,self.number_of_states])
		self.state_counter = 0
		self.alpha = .5 # learning rate		
		self.gamma = .1 #discount factor
		self.epsilon = .01
		self.data = pd.DataFrame({'alpha' : self.alpha, 'gamma' : self.gamma, 'epsilon' : self.epsilon,'successful': 0,'infractions' : 0, 'Q': [self.q], 'R': [self.R]})
		print self.data

		
	def reset(self, destination=None):
		self.planner.route_to(destination)
		# TODO: Prepare for a new trip; reset any variables here, if required
		self.color = 'red'  # override color
		self.state = []
		self.next_waypoint = None
		
	def return_state(self, inputs, next_waypoint):
		'''Builds a state tuple'''
		action = next_waypoint #dejar como next_way_point porque ha cambiado la variable
		state = 0
		if action == 'forward':
			if inputs['light'] != 'green':
				state = 1
		elif action == 'left':
			if inputs['light'] == 'green' and (inputs['oncoming'] == None or inputs['oncoming'] == 'left'):
				state = 2
			else:
				state = 3
		elif action == 'right':
			if inputs['light'] == 'green' or inputs['left'] != 'forward':
				state = 4
			else:
				state = 5
		return state
	
	def store_states(self,this_state):
		'''Creates a list with the states that have appeared up to now (light,oncoming,right,left,next_waypoint)'''
		if not this_state in self.list_of_states:
			self.list_of_states.append(this_state)
			
	def get_reward(self, reward):
		if reward == -1.:
			self.data['infractions']+= 1
		elif reward== 12. or reward == 9.5 or reward== 9. or reward ==10. :
			sefl.data['successfull'] += 1
	def update(self, t):
		# Gather inputs
		self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
		inputs = self.env.sense(self)
		deadline = self.env.get_deadline(self)
		
		# TODO: Update state
		this_state =  self.return_state(inputs, self.next_waypoint)
		self.store_states(this_state)
		self.state.append(self.list_of_states.index(this_state))
		
			#print self.list_of_states
		#print self.state
	
	
		# TODO: Select action according to your policy
		action = self.select_action()
		
		# Execute action and get reward
		reward = self.env.act(self, action)
		self.R[self.state[-1]][self.valid_actions.index(action)] = reward
		#print self.R[range(len(self.list_of_states))][:]
		
		# TODO: Learn policy based on state, action, reward
		self.update_q(action,t+1.)
		#print self.q[range(len(self.list_of_states))][:]
		
		#we select the maximum argument from the list of possible actions for the state
		self.get_policy()

		#print self.policy[range(len(self.list_of_states))]
				
		print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

	def update_q(self, action, t):
		'''Update the Q matrix'''
		reward = self.R[self.state[-1]][self.valid_actions.index(action)] 
		next_inputs = self.env.sense(self)
		next_waypoint = self.planner.next_waypoint() 
		next_state = self.return_state(next_inputs, next_waypoint)
		temp = []
		self.store_states(next_state)
		print next_state, self.list_of_states.index(next_state) 
		temp= [self.q[self.list_of_states.index(next_state)][self.valid_actions.index(next_action)] for next_action in self.valid_actions] 
		# First approach : Q_hat(s,a) += alpha(r+gamma*max(abs(Q_hat(s',a')-Q_hat(s,a))))
		self.alpha = 1./t
		self.q[self.state[-1]][self.valid_actions.index(action)]+= self.alpha* (reward +  self.gamma*np.max(temp-(1./self.gamma)*self.q[self.state[-1]][self.valid_actions.index(action)]))
				# we iterate over the possible combinations of states and actions, since any future combination is possible 
		#print self.q[range(len(self.list_of_states))][:]
		#self.list_of_states.pop() # we delete the state we have added just to not make confussion for the update script	
	def select_action(self):
		if len(self.state)==1: # if we are in the first iteration 
		#1: random action
			action = random.choice(self.valid_actions)
		#1 next waypoint can be a good direction to start
			action= self.next_waypoint
		else:
		#2_using policy to take an action
			probabilities = [1.-self.epsilon  , self.epsilon/float(len(self.valid_actions)) , self.epsilon/float(len(self.valid_actions)) , self.epsilon/float(len(self.valid_actions)), self.epsilon/float(len(self.valid_actions))]
			#print probabilities
			#probabilities = [1-e 0.25*e 0.25*e 0.25*e 0.25*e]
			these_actions = []						
			these_actions = [self.valid_actions[self.policy[self.state[-2]]]]  + self.valid_actions
			#print these_actions
			action =  np.random.choice(these_actions, 1, p= probabilities)[0]#we access the state previous to the present one in order to choose the action to take 
		return action
		
	def get_policy(self):
		next_inputs = self.env.sense(self)
		next_waypoint = self.planner.next_waypoint() 
		next_state = self.return_state(next_inputs, next_waypoint)
		self.policy[self.state[-1]]= np.argmax([self.q[self.list_of_states.index(next_state)][self.valid_actions.index(next_action)] for next_action in self.valid_actions]) 
		return self.policy[self.state[-1]]
		
	def write_state_to_csv(self):
		"""Write a line to the output CSV file"""
		# MODIFICATION
		self.data['Q'] = [self.q]
		self.data['R'] = [self.R]

		try:
			df = pd.read_csv('outcome.csv')
			pdb.set_trace()
			df = pd.merge(df, self.data, how='right', on=['successful', 'infractions', 'Q', 'R'])
		except IOError:
			with open("outcome.csv", "w"):
			# now you have an empty file already
				df = self.data
				pass  # or write something to it already

		df.to_csv('outcome.csv')
		for filename, variable in zip(['list_of_states','reward','Q'] , [self.list_of_states, self.R, self.q]):
			output_file = open(filename+"_alpha_{}_gamma_{}_epsilon_{}.csv".format(self.alpha, self.gamma, self.epsilon), 'wb')# append row to previous ones
			writer = csv.writer(output_file, delimiter='\n')
			# example row in the CSV file
			writer.writerow(variable)#, self.q, self.R ))

def remove_file(filename):
	try:
		os.remove(filename)
	except OSError:
		pass
		
def run():
	"""Run the agent for a finite number of trials."""
	#pdb.set_trace()
	
	#os.remove(os.path.join(os.path.join(os.path.dirname(__file__), os.pardir), 'output.csv')) #reset statistics
	#os.remove(os.path.join(os.path.join(os.path.dirname(__file__), os.pardir), 'simulation.csv')) #reset statistics 
	list_of_files = ['simulation.csv', 'output.csv', 'list_of_states.csv','reward.csv','Q.csv']
	for filename in list_of_files:
		remove_file(os.path.join(os.path.join(os.path.dirname(__file__), os.pardir), filename))
	# Set up environment and agent
	e = Environment()  # create environment (also adds some dummy traffic)
	a = e.create_agent(LearningAgent)  # create agent
	e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
	sim = Simulator(e, update_delay=0.1, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

	sim.run(n_trials=2)#run for a specified number of trials
	a.write_state_to_csv()
	
	# NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
	
	#pdb.set_trace()



if __name__ == '__main__':
    run()
