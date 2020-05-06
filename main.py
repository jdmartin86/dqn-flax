import numpy as np
import matplotlib.pyplot as plt

from functools import partial

import jax
from jax import numpy as jnp, random, jit, lax, vmap, grad

import flax
from flax import nn, optim

import random

from environment import *

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = jnp.expand_dims(state, 0)
        next_state = jnp.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return {'state': jnp.concatenate(state), 'action':jnp.asarray(action), 
                'reward':jnp.asarray(reward), 
                'next_state':jnp.concatenate(next_state), 'done':jnp.asarray(done)}
    
    def __len__(self):
        return len(self.buffer)

#@title Network definition
class DQN(flax.nn.Module):
  """DQN network."""
  def apply(self, x, num_actions):
    x = flax.nn.Conv(x,features=32, kernel_size=(8,8), strides=(4,4))
    x = flax.nn.relu(x)
    x = flax.nn.Conv(x,features=64, kernel_size=(4,4), strides=(2,2))
    x = flax.nn.relu(x)
    x = flax.nn.Conv(x,features=64, kernel_size=(3,3), strides=(1,1))
    x = flax.nn.relu(x)
    x = x.reshape((x.shape[0], -1))
    x = flax.nn.Dense(x, features=512)
    x = flax.nn.relu(x)
    x = flax.nn.Dense(x, features=num_actions)
    return x

epsilon_start = 0.1
epsilon_final = 0.1
epsilon_decay = 50

epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * jnp.exp(-1. * frame_idx / epsilon_decay)

def flatnonzero(a):
  return jnp.nonzero(jnp.ravel(a))[0]

def rand(key, num_actions):
  return jax.random.randint(key, (1,), 0, num_actions)[0]

def rand_argmax(a):
  return np.random.choice(jnp.nonzero(jnp.ravel(a == jnp.max(a)))[0])
  
@jit
def policy(key, x, model, epsilon, num_actions):
  prob = jax.random.uniform(key)
  q = jnp.squeeze(model(jnp.expand_dims(x, axis=0)))
  rnd = partial(rand, num_actions=num_actions)
  a = jax.lax.cond(prob < epsilon, key, rnd, q, jnp.argmax)
  return a

@vmap
def q_learning_loss(q, target_q, action, action_select, reward, done, gamma=0.9):
  td_target = reward + gamma*(1.- done)*target_q[action_select]
  td_error = jax.lax.stop_gradient(td_target) - q[action]
  return td_error**2

@jit
def train_step(optimizer, target_model, batch):
  def loss_fn(model):
    q = model(batch['state'])
    done = batch['done']
    target_q = target_model(batch['next_state'])
    action_select = model(batch['next_state']).argmax(-1)
    return jnp.mean(q_learning_loss(q, target_q, batch['action'], action_select, 
                                    batch['reward'], batch['done']))
    
  loss, grad = jax.value_and_grad(loss_fn)(optimizer.target)
  optimizer = optimizer.apply_gradient(grad)
  return optimizer, loss

def run_experiment(num_episodes=1000,
                   num_steps=50,
                   batch_size=32,
                   replay_size=1000,
                   target_update_frequency=10,
                   gamma=0.9):
  # Create environment
  env = ImageGridworld()
  replay_buffer = ReplayBuffer(replay_size)

  #logging
  ep_losses = []
  ep_returns = []
  key = jax.random.PRNGKey(0)

  # Build and initialize the action selection and target network.
  num_actions = env.action_space.n
  state = env.reset()

  module = DQN.partial(num_actions=num_actions)
  _, initial_params = module.init(key, jnp.expand_dims(state, axis=0))
  model = nn.Model(module, initial_params)
  target_model = nn.Model(module, initial_params)

  # Build and initialize optimizer.
  optimizer = optim.Adam(1e-3).create(model)

  for n in range(num_episodes):
    state = env.reset()

    # Initialize statistics
    ep_return = 0.

    for t in range(num_steps):
      # Generate an action from the agent's policy.
      epsilon = 0.1#epsilon_by_frame(n)
      action = policy(key, state, optimizer.target, epsilon, num_actions)  

      # Step the environment.
      next_state, reward, done, _ = env.step(int(action))

      # Tell the agent about what just happened.      
      replay_buffer.push(state, action, reward, next_state, done)
      ep_return += reward

      # Update the value model when there's enough data.
      if len(replay_buffer) > batch_size:
        batch = replay_buffer.sample(batch_size)
        optimizer, loss = train_step(optimizer, target_model, batch)
        ep_losses.append(float(loss))

      #Update Target model parameters
      if t % target_update_frequency == 0:
        target_model = target_model.replace(params=optimizer.target.params)

      # Terminate episode when absorbing state reached.
      if done: break

      # Cycle the state
      state = next_state

    # Update episodic statistics
    ep_returns.append(ep_return)
    
    if n % 100 == 0:
      print("Episode #{}, Return {}, Loss {}".format(n, ep_return, loss))

  return optimizer.target, ep_returns, ep_losses

qfunc, ep_returns, ep_losses = run_experiment()

plt.plot(ep_returns)

plt.plot(ep_losses)
