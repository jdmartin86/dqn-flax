import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from typing import Callable, Mapping, NamedTuple, Tuple, Sequence

# Gym
import gym
from gym.spaces import Discrete
from gym.spaces import Box

# Images
from skimage.transform import resize
from skimage.color import rgb2gray

# @title Environment Configs
"""
'S' : starting point

'F' or '.': free space (room 1)
'W' or 'x': wall
'H' or 'o': hole (terminates episode)
'G' : goal
'K' : key for door
'D' : locked door


"""

MAPS = {
    "smallway":
    [
        "GD.S.K"
    ],
    "hallway": [
        "GD...S...K"
    ],
    "room": [
        "GW..S",
        ".W...",
        ".D...",
        ".W...",
        ".W..K",
    ],
    "large-room": ["GW...S"]+3*[".W...."]+[".D...."]+3*[".W...."]+[".W...K"],
    "large-room-nodoor": ["GW...S"]+3*[".W...."]+["......"]+3*[".W...."]+[".W...K"],
    "hallway-nodoor": [
        "G....S...K"
    ],
    "room-nodoor": [
        "GW..S",
        ".W...",
        ".....",
        ".W...",
        ".W..K",
    ],
    "four-rooms": [
     "...W..G",
     ".......",
     "...W...",
     "W.WWW.W",
     "...W...",
     ".S.....",
     "...W...",
    ]

}

# @title Environments
class KeyEnv(gym.Env):
    """
    'S' : starting point

    'F' or '.': free space (room 1)
    'W' or 'x': wall
    'H' or 'o': hole (terminates episode)
    'G' : goal
    'K' : key for door
    'D' : locked door


    """

    def __init__(self, desc='4x4',start_random=False):
        if isinstance(desc, str):
            desc = MAPS[desc]
        desc = np.array(list(map(list, desc)))
        desc[desc == '.'] = 'F'
        desc[desc == 'o'] = 'H'
        desc[desc == 'x'] = 'W'
        self.desc = desc
        self.start_random = start_random
        self.n_row, self.n_col = desc.shape
        (start_x,), (start_y,) = np.nonzero(desc == 'S')
        self.possible_start_states = np.nonzero(np.logical_or(desc == 'S', desc == 'F'))

        self.start_state = start_x * self.n_col + start_y
        self.state = self.start_state
        self._P = None
        self._R = None

    def has_key(self,state):
        return (state // (self.n_col * self.n_row)) == 1
    
    def get_xy(self,state):
        state = state % (self.n_col * self.n_row)
        x = state // self.n_col
        y = state % self.n_col
        return x,y
    
    def get_state_number(self, x, y, key):
        return key*(self.n_col * self.n_row) + x * self.n_col + y

    def get_tile_type(self,state):
        x,y = self.get_xy(state)
        return self.desc[x][y]
    
    
    def reset(self):
        if self.start_random:
            i = np.random.choice(len(self.possible_start_states[0]))
            x,y = self.possible_start_states[0][i],self.possible_start_states[1][i]
            self.state = x * self.n_col + y
        else: 
            self.state = self.start_state
        return self.start_state

    @staticmethod
    def action_from_direction(d):
        """
        Return the action corresponding to the given direction. This is a helper method for debugging and testing
        purposes.
        :return: the action index corresponding to the given direction
        """
        return dict(
            left=0,
            down=1,
            right=2,
            up=3
        )[d]

    def step(self, action):
        """
        action map:
        0: left
        1: down
        2: right
        3: up
        :param action: should be a one-hot vector encoding the action
        :return:
        """
        possible_next_states = self.get_possible_next_states(self.state, action)

        probs = [x[1] for x in possible_next_states]
        next_state_idx = np.random.choice(len(probs), p=probs)
        next_state = possible_next_states[next_state_idx][0]

        next_state_type = self.get_tile_type(next_state)

        if next_state_type == 'H':
            done = True
            reward = 0
        elif next_state_type in ['F','S','D','K']:
            done = False
            reward = 0
        elif next_state_type == 'G':
            done = True
            reward = 1
        else:
            raise NotImplementedError
        self.state = next_state
        done = False
        return self.state, reward, False, dict()

    def transition_distribution(self,state,action):
        return self.get_possible_next_states(state,action)

    def P(self):
      if self._P is not None:
        return self._P
      self._P = np.zeros((self.observation_space.n, self.action_space.n, self.observation_space.n))
      for s in range(self.observation_space.n):
        for a in range(self.action_space.n):
          for s_next, p in self.transition_distribution(s, a):
            self._P[s, a, s_next] += p
      return self._P

    def reward(self,state,action):
        (next_state,p), = self.transition_distribution(state,action)
        return 1 if self.get_tile_type(next_state) == 'G' else 0
    def R(self):
      if self._R is not None:
        return self._R
      self._R = np.zeros((self.observation_space.n, self.action_space.n))
      for s in range(self.observation_space.n):
        for a in range(self.action_space.n):
          self._R[s,a] = self.reward(s,a)
      return self._R

    def get_possible_next_states(self, state, action):
        """
        Given the state and action, return a list of possible next states and their probabilities. Only next states
        with nonzero probabilities will be returned
        :param state: start state
        :param action: action
        :return: a list of pairs (s', p(s'|s,a))
        """
        # assert self.observation_space.contains(state)
        # assert self.action_space.contains(action)

        has_key = self.has_key(state)
        x,y = self.get_xy(state) 

        coords = np.array([x, y])

        increments = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])
        next_coords = np.clip(
            coords + increments[action],
            [0, 0],
            [self.n_row - 1, self.n_col - 1]
        )

        next_has_key = has_key or self.get_tile_type(self.get_state_number(next_coords[0],next_coords[1],False)) == 'K'

        next_state = self.get_state_number(next_coords[0],next_coords[1],next_has_key)

        state_type = self.get_tile_type(state)
        next_state_type = self.get_tile_type(next_state)

        if next_state_type == 'W' or state_type == 'H' or state_type == 'G' or ((state_type == 'D' or next_state_type=='D') and not has_key):
            return [(state, 1.)]
        else:
            return [(next_state, 1.)]

    @property
    def action_space(self):
        return Discrete(4)

    @property
    def observation_space(self):
        return Discrete(2 * self.n_row * self.n_col)

    def set_state(self,state):
        self.state = state


    def plot(self, state=None, ax=None):
        if state is None:
          state = self.state
        has_key = self.has_key(state)
        colors = {'S':(255,255,255,255),
                'F': (255,255,255,255),
                'W': (0,0,0,255),
                'H': (255,0,0,255),
                'G':(0,255,0,255),
                'K':((255,255,255,100) if has_key else (218,165,32,255)),
                'D':((212,170,144,100) if has_key else (139,69,19,255)),
            }
        import matplotlib.pyplot as plt
        if ax is None:
          ax = plt.gca()

        for i,line in enumerate(self.desc):
            for j,value in enumerate(line):
                startY = self.n_row-i
                startX = j
                xs = np.linspace(startX,startX+1,10) 
                ax.fill_between(xs,startY-1,startY,color=np.array([colors[value]]*10)/255)
        x,y = self.get_xy(state)
        ax.scatter([y+0.5],[self.n_row-x-0.5],s=600, c='b')
        ax.set_xlim(0,self.n_col)
        ax.set_ylim(0,self.n_row)
        
    
    def visualize_value(self, values, ax=None, text=True):
        import matplotlib.pyplot as plt
        import matplotlib.colors
        import matplotlib.cm as cm
        if ax is None:
          ax = plt.gca()

        if not text:
            ax.text = lambda *args, **kwargs: None

        norm = matplotlib.colors.Normalize(vmin=min(values), vmax=max(values), clip=True)
        cmap = cm.Blues_r
        values = np.array(values).flatten()
        assert len(values) == self.n_col*self.n_row*2
        iterator = iter(values)

        for i,line in enumerate(self.desc):
            for j,value in enumerate(line):
                startY = self.n_row-i
                startX = j
                act_value = next(iterator) 
                color = cmap(norm(act_value))
                if value == 'W':
                    color = (0,0,0,1)
                xs = np.linspace(startX,startX+1,10) 
                ax.fill_between(xs,startY-1,startY,color=color)
                ax.text(startX+0.4,startY-0.5,'%.3g'%act_value)
        
        for i,line in enumerate(self.desc):
            for j,value in enumerate(line):
                startY = self.n_row-i
                startX = j+self.n_col+1
                act_value = next(iterator) 
                color = cmap(norm(act_value))
                if value == 'W':
                    color = (0,0,0,1)
                xs = np.linspace(startX,startX+1,10) 
                ax.fill_between(xs,startY-1,startY,color=color)
                ax.text(startX+0.4,startY-0.5,'%.3g'%act_value)
        

        ax.set_xlim(0,self.n_col*2+1)
        ax.set_ylim(0,self.n_row)


    def visualize_qvalue(self,qvalues,text=True,display_format='%.2g', ax=None, bounds=None):
        import matplotlib.pyplot as plt
        import matplotlib.colors
        import matplotlib.cm as cm
        if ax is None:
          ax = plt.gca()

        if not text:
            ax.text = lambda *args, **kwargs: None
        qvalues = np.array(qvalues).flatten()
        assert len(qvalues) == self.n_col*self.n_row*2*4
        iterator = iter(qvalues)
        if bounds:
            vmin,vmax = bounds
        else:
            vmin,vmax = min(qvalues),max(qvalues)
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        cmap = cm.Blues_r


        for i,line in enumerate(self.desc):
            for j,value in enumerate(line):
                startY = self.n_row-i
                startX = j
                act_values = [next(iterator) for _ in range(4)]
                colors = [cmap(norm(act_value)) for act_value in act_values]

                if value == 'W':
                    colors = [(0,0,0,1)]*4

                xs = np.linspace(0,1,10)
                top_v = np.maximum.reduce([xs,1-xs])
                
                ax.fill_between(startX+xs,startY-1+xs,startY-xs,where=[True]*5+[False]*5,color=colors[0])
                ax.text(startX+0.1,startY-0.5,display_format%act_values[0])

                ax.fill_between(startX+xs,startY-1,startY-top_v,color=colors[1])
                ax.text(startX+0.3,startY-0.8,display_format%act_values[1])

                ax.fill_between(startX+xs,startY-1+xs,startY-xs,where=[False]*5+[True]*5,color=colors[2])
                ax.text(startX+0.7,startY-0.5,display_format%act_values[2])


                ax.fill_between(startX+xs,startY-1+top_v,startY,color=colors[3])
                ax.text(startX+0.3,startY-0.2,display_format%act_values[3])

                location = [(startX+0.1,startY-0.5),(startX+0.3,startY-0.8),(startX+0.7,startY-0.5),(startX+0.3,startY-0.2)][np.argmax(act_values)]
                ax.text(*location,s='*')
        for i,line in enumerate(self.desc):
            for j,value in enumerate(line):
                startY = self.n_row-i
                startX = j + self.n_col + 1
                act_values = [next(iterator) for _ in range(4)]
                colors = [cmap(norm(act_value)) for act_value in act_values]

                if value == 'W':
                    colors = [(0,0,0,1)]*4

                xs = np.linspace(0,1,10)
                top_v = np.maximum.reduce([xs,1-xs])
                
                ax.fill_between(startX+xs,startY-1+xs,startY-xs,where=[True]*5+[False]*5,color=colors[0])
                ax.text(startX+0.1,startY-0.5,display_format%act_values[0])

                ax.fill_between(startX+xs,startY-1,startY-top_v,color=colors[1])
                ax.text(startX+0.3,startY-0.8,display_format%act_values[1])

                ax.fill_between(startX+xs,startY-1+xs,startY-xs,where=[False]*5+[True]*5,color=colors[2])
                ax.text(startX+0.7,startY-0.5,display_format%act_values[2])


                ax.fill_between(startX+xs,startY-1+top_v,startY,color=colors[3])
                ax.text(startX+0.3,startY-0.2,display_format%act_values[3])
                location = [(startX+0.1,startY-0.5),(startX+0.3,startY-0.8),(startX+0.7,startY-0.5),(startX+0.3,startY-0.2)][np.argmax(act_values)]
                ax.text(*location,s='*')
        ax.set_xlim(0,self.n_col*2+1)
        ax.set_ylim(0,self.n_row)

def test():
    e = KeyEnv("room")
    #e.plot()
    v = np.random.randn(200)
    e.visualize_qvalue(v,False)

e = KeyEnv("room-nodoor")
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
actions = [1, 1, 1, 1, 0, 0, 3, 3, 0, 0, 3, 3, 3]

for a, ax in zip(actions, axes.flatten()):
  e.plot(ax=ax)
  e.step(a)
  ax.set_yticks([])
  ax.set_xticks([])

#@title Wrapper for environment
def fig2data(fig):
    """ Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it.
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer (fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = (w,h,4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll (buf,3,axis=2)[..., :3]
    return buf


class ImageGridworld(KeyEnv):
  def __init__(self, desc='room', start_random=False, image_size=84, grayscale=False):
    super().__init__(desc, start_random)
    self._image_size = 84

    nS = super().observation_space.n
    nC = 1 if grayscale else 3
    self._observation_space = Box(low=0.0, high=1.0, shape=(image_size, image_size, nC), dtype=np.float32)

    self._image_states = np.empty((nS, image_size, image_size, nC))
    for i in range(nS):
      fig, ax = plt.subplots(figsize=(3, 3))
      self.plot(state=i, ax=ax)
      plt.axis('off')
      image = fig2data(fig)
      plt.close()
      im_resized = resize(image, (self._image_size, self._image_size))
      if grayscale:
        im_resized = rgb2gray(im_resized)
      self._image_states[i] = im_resized
      print('{}/{} rendered'.format(i, nS), end='\r')

  @property
  def observation_space(self):
    return self._observation_space

  def step(self, action):
    internal_state, reward, done, info = super().step(action)
    info['state'] = internal_state
    state = self._image_states[internal_state]
    return state, reward, done, info

  def reset(self):
    internal_state = super().reset()
    return self._image_states[internal_state]
