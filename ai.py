# AI for Self Driving Car

# Importing the libraries

import numpy as np 
import random
import os

import torch
import torch.nn as nn ## Neural Networks
import torch.nn.functional as F ## Functions present in the Neural Networks
import torch.optim as optim ## Optimizer
import torch.autograd as autograd # Pytorch's automatic differentiation engine that powers Neural Networks training
from torch.autograd import Variable ## Take notes of autograd and Pytorch variable 

# Creating the architecture of the Neural Network

class Network(nn.Module): ## Inheritance -- the Class Network inherits it's parent class called Module in torch.nn
## Module - base class for all neural networks
    def __init__(self, input_size, nb_action):  ## Constructor - input_size == No of neurons in the input layer typically 5 - 3 signals of the car, + orientation , - orientationnb_action == no of neurons in the output layer typically 3 - moving straight, moving left, moving right
    ##Using Super() to invoke the parent class constructor ## super(ClassName, obj name).name_of_the_function()"""
        super(Network, self).__init__() 
        self.input_size = input_size
        self.nb_action = nb_action
        ## Full connections of the neurons
        ## Since there is only one hidden layer, Two full connection
        ## All the input neurons to the input layer and all the neurons in the hidden layer to the output neurons
        self.fc1 = nn.Linear(input_size, 30) ## Paramaters -- the number of neurons in the input layer and no of neurons in the hidden layer
        self.fc2 = nn.Linear(30, nb_action)
    ## This forward function will return the qvalues of the state and used rectified linear unit as an activation function relu
    def forward(self, state):
        x = F.relu(self.fc1(state)) ## X -- Hidden Neurons activation layer(rectifier Function)
        q_values = self.fc2(x)
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):
    ## 2 Process
    ## Add a new event to the memory
    ## Ensures to have 100 transitions(event)
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        ##event -- last state, new State, last action and last reward
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size)) 
        ## Taking randon samples from self.memory of precise batch size
        ## if list = ((1,2,3),(4,5,6)) zip(*list) = ((1.4),(2,3),(5,6))
        ## Why Zip -- to split the (1,2,3) - state1,action1,reward1 to state1,state2 | action1,action2 | reward1,reward2
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        ## Input size == no of neurons in the input layer
        ## nb_action == no of neurons on the output layer
        ## gamma == delay coefficient or discounting factor
        self.gamma = gamma
        self.reward_window = []
        ## reward window -- sliding window of the mean of last 100 rewards increasing with time 
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        ## self.model.parameters() -- the parameters of the neural network ; lr - learning rate -- should be low so that the AI can learn correctly
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        ## last state = vector of 5 dimensions - three signals , + orientation  and minus orientation
        ## with one fake dimension depicting the batch_size that is going to be the first dimension using unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
    
    ## what action to be performed by the AI
    def select_action(self, state):
        ## state
        ## Output of the neural network(3) is gonna be the input to the select_action 
        ## So the output of the neural netowrk purely depends on the input of the neural network
        ## Input of the neural Networks -- Last state of the AI
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        ## volatile = true --> make sure that we don't have the gradient descent associated with state
        ## will save some memory -- high performace
        ## T -- Temperature Parameter will modulate how the neural netowrk shoud be sure which action should be performed
        ## T closer to zero == the neural network will be less sure to playing that action
        ## T higher == the neural network will be more sure to playing that action
        action = probs.multinomial(num_samples=1)
        ## probs.multinomial --> will take the random prob dist from probs
        ## Now action contains the fake batch dimension so to get rid of it
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph = True)
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    ## Save the brain of the car
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    ## Loads the brain of the car
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
