import random
import typing
import torch
import time

class Transition(typing.NamedTuple):
    prev_state: typing.Union[torch.cuda.FloatTensor]
    prev_action: int
    prev_reward: float
    cur_state: typing.Union[torch.cuda.FloatTensor]
    end_transition: bool

class ReplayMemory(object):
    capacity: int
    memory: typing.List[Transition]
    position: int
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def push(self, transition: Transition):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> typing.Iterable[Transition]:
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

import gym
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import torchvision.transforms as transforms
import PIL

class SimpleAtariNet(nn.Module):
    def __init__(self):
        super(SimpleAtariNet, self).__init__()
        # Assume input is size 140x600
        self.conv0 = nn.Conv2d(3, 16, 12, stride=(2, 8))
        self.conv1 = nn.Conv2d(16, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.lin1 = nn.Linear(1280, 512)
        self.lin2 = nn.Linear(512, 2)
    
    def forward(self, x):
        x = functional.relu(self.conv0(x))
        x = functional.relu(self.conv1(x))
        x = functional.relu(self.conv2(x))
        x = functional.relu(self.conv3(x))
        x = functional.relu(self.lin1(x.view(-1, 1280)))
        x = self.lin2(x)
        return x

def test_net():
    assert SimpleAtariNet()(autograd.Variable(torch.ones(1, 3, 140, 600))).size() == (1, 2)

#test_net()

class CartPoleDQNContextSimple(object):
    def __init__(self, memory_size: int, cuda: bool, pause_length=None):
        self.pause_length = pause_length
        self.cuda = cuda
        self.env = gym.make('CartPole-v1')
        self.replay_memory = ReplayMemory(memory_size)
        self.net = SimpleAtariNet()
        self.targetnet = SimpleAtariNet()
        self.targetnet.load_state_dict(self.net.state_dict())
        if self.cuda:
            self.net.cuda()
            self.targetnet.cuda()
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=0.00025, weight_decay = 0.001)
        self.batch_size = 32
        self.c = 10000
        self.resize = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.ToTensor(),
             transforms.Lambda(lambda x: x[:, 160:300, :])])
        self.discount_factor = 0.99
        self.num_steps_with_current_target = 0
        self.init_epsilon = 0.1
        self.final_epsilon = 0.9
        self.anneal_length = 1000000
        self.num_steps = 0
        self.frame_skip_frequency = 4
    
    def do_action(self, action: int):
        for i in range(self.frame_skip_frequency):
            _, reward, done, _ = self.env.step(action)
            if self.pause_length != None:
                time.sleep(self.pause_length)
                self.env.render()
            if done:
                break
        return reward, done
        
    def run_episode(self):
        self.env.reset()
        prev_screen = self.get_screen()
        prev_state = self.get_screen()
        prev_action = self.choose_action(prev_state)
        prev_reward, done = self.do_action(prev_action)
        while not done:
            cur_screen = self.get_screen()
            cur_state = cur_screen - prev_screen
            
            cur_epsilon = self.init_epsilon + (self.final_epsilon - self.init_epsilon)*self.num_steps/self.anneal_length
            if random.random() > 1 - cur_epsilon:
                cur_action = self.env.action_space.sample()
            else:
                cur_action = self.choose_action(cur_state)
            
            cur_reward, done = self.do_action(cur_action)
            
            if done:
                cur_reward = -1 # Modified because of problem with constant reward. 
            else:
                cur_reward = 0
            
            # Add transition.
            self.replay_memory.push(Transition(prev_state=prev_state,
                                               prev_action=prev_action,
                                               prev_reward=prev_reward,
                                               cur_state=cur_state,
                                               end_transition=False))
            
            # If we're done, we also need to add the next Transition with an end state.
            if done:
                self.replay_memory.push(Transition(prev_state=cur_state,
                                                   prev_action=cur_action,
                                                   prev_reward=cur_reward,
                                                   cur_state=self.get_screen(),
                                                   end_transition=True))
            # Now that we've added transitions, we can sample self.batch_size number of replay transitions
            # and perform a step in the optimizer.
            cur_batch_size = min(len(self.replay_memory), self.batch_size)
            transitions = self.replay_memory.sample(cur_batch_size)
            states, actions, rewards, next_states, end_transitions = zip(*transitions)
            
            states = autograd.Variable(torch.cat([state.unsqueeze(0) for state in states])) # Add batch dim to each sample.
            rewards = autograd.Variable(torch.cat([torch.Tensor([reward]) for reward in rewards]))
            end_transitions = autograd.Variable(torch.Tensor(end_transitions).float())
            next_states = autograd.Variable(torch.cat([next_state.unsqueeze(0) for next_state in next_states]))
            
            actions_mask = torch.zeros(cur_batch_size, 2).byte()
            for sample_index, action in enumerate(actions):
                actions_mask[sample_index, action] = 1
            
            if self.cuda:
                states = states.cuda()
                next_states = next_states.cuda()
                rewards = rewards.cuda()
                end_transitions = end_transitions.cuda()
                actions_mask = actions_mask.cuda()
            
            target_qvalues = self.targetnet(next_states)[actions_mask]
            if self.cuda:
                target_qvalues = target_qvalues.cuda()
            current_qvalues, _ = self.net(states).max(1)
            
            error = torch.sum(
                (rewards + self.discount_factor*target_qvalues*end_transitions - current_qvalues)**2)/cur_batch_size
            self.optimizer.zero_grad()
            error.backward()
            self.optimizer.step()
            self.num_steps_with_current_target += 1
            if self.num_steps_with_current_target == self.c:
                self.targetnet.load_state_dict(self.net.state_dict())
                self.num_steps_with_current_target = 0
            
            # After all that is done, we reset our context variables for the next iteration.
            prev_screen = cur_screen
            prev_state = cur_state
            prev_action = cur_action
            prev_reward = cur_reward
            
            self.num_steps += 1
    
    def run_episode_without_training(self):
        self.env.reset()
        prev_screen = self.get_screen()
        prev_state = self.get_screen()
        prev_action = self.choose_action(prev_state)
        prev_reward, done = self.do_action(prev_action)
        while not done:
            cur_screen = self.get_screen()
            cur_state = cur_screen - prev_screen
            cur_action = self.choose_action(cur_state)
            cur_reward, done = self.do_action(cur_action)
            prev_screen = cur_screen
            prev_state = cur_state
            prev_action = cur_action
            prev_reward = cur_reward
    
    def get_screen(self):
        screen = self.resize(self.env.render(mode='rgb_array')).float()/255.0
        
        return screen
    
    def choose_action(self, observation: torch.Tensor) -> typing.Tuple[autograd.Variable, int]:
        """Returns the action chosen by the network."""
        network_input = autograd.Variable(observation.unsqueeze(0))
        if self.cuda:
            network_input = network_input.cuda()
        value, index = self.net(network_input).max(1)
        return index.data[0]

def test_context():
    CartPoleDQNContextSimple(2000, True).run_episode()
    CartPoleDQNContextSimple(2000, False).run_episode()
    CartPoleDQNContextSimple(2000, True).run_episode_without_training()
    CartPoleDQNContextSimple(2000, False).run_episode_without_training()

#test_context()

context = CartPoleDQNContextSimple(2000, True, pause_length=0.1)
trained_net = torch.load('./cartpole-net-v2')
context.net.load_state_dict(trained_net.state_dict())
context.run_episode_without_training()

