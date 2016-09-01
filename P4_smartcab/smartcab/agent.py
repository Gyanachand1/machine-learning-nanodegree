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
		self.valid_actions = [ None, 'forward', 'left', 'right'] 
		self.state = [] #we will use a list of tuples of states to map the indexes
		self.list_of_states = []
		self.next_waypoint = None
		self.number_of_states= (len(self.valid_actions)**4)*2 # 4 waypoints or actions * 2 light states * 4 oncoming  traffic states * 4 left traffic states  * 4 right  traffic states 
																#  next_waypoint, left, right and oncoming can take values 'None' , 'left', 'right' or 'forward'
		self.R = np.zeros(shape=[self.number_of_states, 4]) #we have 4 actions
		self.q = np.zeros(shape=[self.number_of_states, 4])
		self.policy = np.zeros([self.number_of_states, 1], dtype = int)
		#self.q = np.empty(shape=[self.number_of_states,self.number_of_states])
		self.state_counter = 0
		self.gamma = .8 #random value
		self.alpha = 0.7
		self.epsilon = .001
		
	def reset(self, destination=None):
		self.planner.route_to(destination)
		# TODO: Prepare for a new trip; reset any variables here, if required
		self.color = 'red'  # override color
		self.state = []
		self.next_waypoint = None
		
	def build_state(self, inputs, next_waypoint):
		'''Builds a state tuple'''
		return (inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'], next_waypoint)
	
	def store_states(self,this_state):
		'''Creates a list with the states that have appeared up to now (light,oncoming,right,left,next_waypoint)'''
		if not this_state in self.list_of_states:
			self.list_of_states.append(this_state)
			
	
	def update(self, t):
		# Gather inputs
		self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
		inputs = self.env.sense(self)
		deadline = self.env.get_deadline(self)
		
		# TODO: Update state
		this_state =  self.build_state(inputs, self.next_waypoint)
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
		next_state = self.build_state(next_inputs, next_waypoint)
		temp = []
		self.store_states(next_state)
		temp= [self.q[self.list_of_states.index(next_state)][self.valid_actions.index(next_action)] for next_action in self.valid_actions] 
		# First approach : Q_hat(s,a) += alpha(r+gamma*max(abs(Q_hat(s',a')-Q_hat(s,a))))
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
		next_state = self.build_state(next_inputs, next_waypoint)
		self.policy[self.state[-1]]= np.argmax([self.q[self.list_of_states.index(next_state)][self.valid_actions.index(next_action)] for next_action in self.valid_actions]) 
		return self.policy[self.state[-1]]
def run():
	"""Run the agent for a finite number of trials."""

    # Set up environment and agent
	e = Environment()  # create environment (also adds some dummy traffic)
	a = e.create_agent(LearningAgent)  # create agent
	e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
	sim = Simulator(e, update_delay=0.1, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

	sim.run(n_trials=50)  # run for a specified number of trials
	# NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
	print "CONCLUSION REPORT"
	print "WINS: {}".format(e.wins)
	print "LOSSES: {}".format(e.losses)
	print "INFRACTIONS: {}".format(e.infractions)


if __name__ == '__main__':
    run()
