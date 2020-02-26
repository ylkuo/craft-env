from enum import IntEnum
from gym import spaces
from PIL import Image
from skimage.measure import block_reduce
from skimage.transform import resize
from skimage.util import pad
from utils import Index, pad_slice

import argparse
import copy
import gym
import numpy as np
import os
import pygame
import time
import yaml


def neighbors(pos, width, height, dir=None):
    x, y = pos
    neighbors = []
    if x > 0 and (dir is None or dir == CraftWorldEnv.Actions.left):
        neighbors.append((x-1, y))
    if y > 0 and (dir is None or dir == CraftWorldEnv.Actions.down):
        neighbors.append((x, y-1))
    if x < width - 1 and (dir is None or dir == CraftWorldEnv.Actions.right):
        neighbors.append((x+1, y))
    if y < height - 1 and (dir is None or dir == CraftWorldEnv.Actions.up):
        neighbors.append((x, y+1))
    return neighbors


def nears(pos, width, height):
    x, y = pos
    neighbors = []
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            nx = x + dx
            ny = y + dy
            if nx == x and ny == y:
                continue
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            neighbors.append((nx, ny))
    return neighbors


class Cookbook(object):
    def __init__(self, recipes_path):
        with open(recipes_path) as recipes_f:
            recipes = yaml.load(recipes_f, Loader=yaml.FullLoader)
        self.index = Index()
        self.environment = set(self.get_index(e) for e in recipes["environment"])
        self.primitives = set(self.get_index(p) for p in recipes["primitives"])
        self.recipes = {}
        # set up recipes from yaml file
        if 'recipes' in recipes.keys():
            for output, inputs in recipes["recipes"].items():
                d = {}
                for inp, count in inputs.items():
                    if "_" in inp:  # special keys
                        d[inp] = count
                    else:
                        d[self.get_index(inp)] = count
                self.recipes[self.get_index(output)] = d
        kinds = self.environment | self.primitives | set(self.recipes.keys())
        self.n_kinds = len(self.index)
        # get indices
        self.grabbable_indices = [i+1 for i in range(self.n_kinds)
                                  if i+1 not in self.environment]
        self.workshop_indices = [item for item in self.environment if 'recycle' in self.index.get(item)]

    def primitives_for(self, goal):
        out = {}

        def insert(kind, count):
            assert kind in self.primitives
            if kind not in out:
                out[kind] = count
            else:
                out[kind] += count

        for ingredient, count in self.recipes[goal].items():
            if not isinstance(ingredient, int):
                assert ingredient[0] == "_"
                continue
            elif ingredient in self.primitives:
                insert(ingredient, count)
            else:
                sub_recipe = self.recipes[ingredient]
                n_needed = count
                expanded = self.primitives_for(ingredient)
                for k, v in expanded.items():
                    insert(k, v * n_needed)
        return out

    def get_index(self, item):
        return self.index.index(item)

    def __str__(self):
        out_str = ''
        for item in self.index:
            out_str = '{}{}\t{}\n'.format(out_str, item, self.get_index(item))
        return out_str


class CraftGui(object):
    def __init__(self, env, width, height, is_headless=False,
                 width_px=300, height_px=300, target_fps=None,
                 img_path='images/',
                 caption='CraftWorld Simulator'):
        if is_headless:
            os.environ['SDL_VIDEODRIVER'] = 'dummy'
        self._env = env
        self._width = width
        self._height = height
        self._cell_width = width_px / width
        self._cell_height = height_px / height
        # load icon images
        self._sprites = {}
        for item in env.cookbook.index:
            if 'none' in item:
                continue
            if item == 'boundary':
                for direction in ['left', 'right', 'top', 'bottom']:
                    filename = img_path + item + '_' + direction + '.png'
                    self._sprites[item  + '_' + direction] = \
                        pygame.image.load(filename)
            else:
                self._sprites[item] = pygame.image.load(img_path + item + '.png')
        for direction in ['left', 'right', 'up', 'down']:
            filename = img_path + 'robot_' + direction + '.png'
            self._sprites['robot_' + direction] = pygame.image.load(filename)
        # pygame related
        self._target_fps = target_fps
        self._screen = pygame.display.set_mode((width_px, height_px), 0, 32)
        pygame.display.set_caption(caption)
        self._clock = pygame.time.Clock()

    def move(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    obs, reward, done, _ = self._env.step(self._env.actions.up)
                elif event.key == pygame.K_DOWN:
                    obs, reward, done, _ = self._env.step(self._env.actions.down)
                elif event.key == pygame.K_LEFT:
                    obs, reward, done, _ = self._env.step(self._env.actions.left)
                elif event.key == pygame.K_RIGHT:
                    obs, reward, done, _ = self._env.step(self._env.actions.right)
                elif event.key == pygame.K_SPACE:
                    obs, reward, done, _ = self._env.step(self._env.actions.use)
                    self._env.print_inventory()
                else:
                    continue
                print('reward: {}, done: {}'.format(reward, done))

    def draw(self, move_first=False):
        if move_first:
            self.move()
        bg_color = (255, 255, 255)
        self._screen.fill(bg_color)
        row = 0
        cell_size = (int(self._cell_width-1), int(self._cell_height-1))
        for y in reversed(range(self._height)):
            for x in range(self._width):
                px_x = x*self._cell_width
                px_y = row*self._cell_height
                rect = pygame.Rect(px_x, px_y, self._cell_width, self._cell_height)
                pygame.draw.rect(self._screen, (200,200,200), rect, 1)
                if self._env.grid[x, y, :].any() or (x, y) == self._env.pos:
                    thing = self._env.grid[x, y, :].argmax()
                    if (x, y) == self._env.pos:

                        if self._env.dir == self._env.actions.left:
                            picture = pygame.transform.scale(self._sprites['robot_left'], cell_size)
                        elif self._env.dir == self._env.actions.right:
                            picture = pygame.transform.scale(self._sprites['robot_right'], cell_size)
                        elif self._env.dir == self._env.actions.up:
                            picture = pygame.transform.scale(self._sprites['robot_up'], cell_size)
                        elif self._env.dir == self._env.actions.down:
                            picture = pygame.transform.scale(self._sprites['robot_down'], cell_size)
                    elif thing == self._env.cookbook.get_index("boundary"):
                        if row == 0:
                            picture = pygame.transform.scale(self._sprites['boundary_top'], cell_size)
                        elif row == self._height - 1:
                            picture = pygame.transform.scale(self._sprites['boundary_bottom'], cell_size)
                        elif x == 0:
                            picture = pygame.transform.scale(self._sprites['boundary_left'], cell_size)
                        elif x == self._width - 1:
                            picture = pygame.transform.scale(self._sprites['boundary_right'], cell_size)
                    else:
                        picture = pygame.transform.scale(self._sprites[self._env.cookbook.index.get(thing)],
                                                         cell_size)
                    self._screen.blit(picture, (px_x, px_y))
            row += 1
        pygame.display.update()
        if self._target_fps is not None:
            self._clock.tick(self._target_fps)


class CraftWorldEnv(gym.Env):
    class Actions(IntEnum):
        down  = 0 # move down
        up    = 1 # move up
        left  = 2 # move left
        right = 3 # move right
        use   = 4 # use

    def __init__(self, recipe_path,
                 init_pos, init_dir, grid,
                 width=10, height=10,
                 window_width=7, window_height=7,
                 time_limit=10, use_gui=True,
                 target_fps=None, is_headless=False):
        self.cookbook = Cookbook(recipe_path)
        # environment rl parameter
        self.actions = CraftWorldEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))
        # set up gui
        self._use_gui = use_gui
        if use_gui:
            self.gui = CraftGui(self, width, height,
                                is_headless=is_headless,
                                target_fps=target_fps)
        if use_gui:
            self.observation_space = spaces.Tuple((
                spaces.Box(low=0, high=255,
                           shape=(80, 80, 3),
                           dtype=np.uint8),
                spaces.Box(low=0, high=time_limit,
                           shape=(self.cookbook.n_kinds+2+4, ),
                           dtype=np.float32)))
        else:
            self.n_features = \
                2 * window_width * window_height * (self.cookbook.n_kinds+1) + \
                self.cookbook.n_kinds + 4 + 1
            self.observation_space = spaces.Box(low=0, high=time_limit,
                                                shape=(self.n_features, ),
                                                dtype=np.float32)
        self.time_limit = time_limit
        # set the environment
        self._width = width
        self._height = height
        self._window_width = window_width
        self._window_height = window_height
        self._init_pos = init_pos
        self._init_dir = init_dir
        self._init_grid = grid
        self.pos = copy.deepcopy(init_pos)
        self.dir = copy.deepcopy(init_dir)
        self.grid = copy.deepcopy(grid)
        self.inventory = np.zeros(self.cookbook.n_kinds)
        # start the first game
        self.reset()

    def load(self, data):
        self._init_grid = data[0]
        self._init_pos = data[1]
        self._init_dir = data[2]
        self.reset()

    def get_data(self):
        return self._init_grid, self._init_pos, self._init_dir

    def step(self, action):
        x, y = self.pos
        n_dir = action
        if action == self.actions.left:
            dx, dy = (-1, 0)
        elif action == self.actions.right:
            dx, dy = (1, 0)
        elif action == self.actions.up:
            dx, dy = (0, 1)
        elif action == self.actions.down:
            dx, dy = (0, -1)
        elif action == self.actions.use:
            dx, dy = (0, 0)
            n_dir = self.dir
        else:  # not supported move
            raise ValueError('Not supported action')
        # move
        self.dir = n_dir
        x = self.pos[0] + dx
        y = self.pos[1] + dy
        if not self.grid[x, y, :].any():
            self.pos = (x, y)
        # take `use` action
        if action == self.actions.use:
            success = False
            for nx, ny in neighbors(self.pos, self._width, self._height, self.dir):
                here = self.grid[nx, ny, :]
                if not self.grid[nx, ny, :].any():
                    continue
                assert here.sum() == 1
                thing = here.argmax()
                if not(thing in self.cookbook.grabbable_indices or \
                        thing in self.cookbook.workshop_indices):
                    continue
                if thing in self.cookbook.grabbable_indices:
                    self.inventory[thing] += 1
                    self.grid[nx, ny, thing] = 0
                    success = True
                elif thing in self.cookbook.workshop_indices:
                    workshop = self.cookbook.index.get(thing)
                    for output, inputs in self.cookbook.recipes.items():
                        if inputs["_at"] != workshop:
                            continue
                        for i in inputs.keys():
                            if i == '_at':
                                continue
                            self.inventory[i] -= inputs[i]
                        success = True
                break
        # TODO: Add your own rule of the game and the reward
        done = False
        reward = 1
        if self._use_gui:
            self.gui.draw()
        return self.feature(), reward, done, {}

    def feature(self):
        x, y = self.pos
        # position features
        pos_feats = np.asarray(self.pos).astype(np.float32)
        pos_feats[0] /= self._width
        pos_feats[1] /= self._height
        # direction features
        dir_features = np.zeros(4)
        dir_features[self.dir] = 1
        if self._use_gui:
            hw = int(self._window_width / 2)
            hh = int(self._window_height / 2)
            img_str = pygame.image.tostring(self.gui._screen, 'RGB')
            img = Image.frombytes('RGB', self.gui._screen.get_size(), img_str)
            cell_width = int(self.gui._cell_width); cell_height = int(self.gui._cell_height)
            px_x = x * cell_width; px_y = (self._height - y - 1) * cell_height
            px_hw = int(hw * cell_width); px_hh = int(hh * cell_height)
            img_padded = pad(img, pad_width=((px_hw, px_hw),
                                             (px_hh, px_hh),
                                             (0,0)),
                             mode='constant')
            new_x = px_hw + px_x; new_y = px_hh + px_y
            out_img = img_padded[new_y-px_hh:new_y+px_hh+cell_height, \
                                 new_x-px_hw:new_x+px_hw+cell_width]
            out_img = out_img.reshape((px_hh*2+cell_height, px_hw*2+cell_width, 3))
            out_img = resize(out_img, [80, 80, 3],
                             preserve_range=True, anti_aliasing=True)
            out_values = np.concatenate((self.inventory, pos_feats, dir_features))
            features = {0: out_img.astype(np.uint8), 1: out_values}
        else:
            hw = int(self._window_width / 2)
            hh = int(self._window_height / 2)
            bhw = int((self._window_width * self._window_width) / 2)
            bhh = int((self._window_height * self._window_height) / 2)

            grid_feats = pad_slice(self.grid, (x-hw, x+hw+1), 
                    (y-hh, y+hh+1))
            grid_feats_big = pad_slice(self.grid, (x-bhw, x+bhw+1),
                    (y-bhh, y+bhh+1))
            grid_feats_big_red = block_reduce(grid_feats_big,
                    (self._window_width, self._window_height, 1), func=np.max)

            features = np.concatenate((grid_feats.ravel(),
                    grid_feats_big_red.ravel(), self.inventory, 
                    dir_features, [0]))
            assert len(features) == self.n_features
        return features

    def reset(self):
        self._seq = []
        self._last_states = set()
        self.inventory = np.zeros(self.cookbook.n_kinds)
        self.pos = copy.deepcopy(self._init_pos)
        self.dir = copy.deepcopy(self._init_dir)
        self.grid = copy.deepcopy(self._init_grid)
        if self._use_gui:
            self.gui.draw()
        return self.feature()

    def visualize(self):
        s = ''
        for y in reversed(range(self._height)):
            for x in range(self._width):
                if not (self.grid[x, y, :].any() or (x, y) == self.pos):
                    ch = ' '
                else:
                    thing = self.grid[x, y, :].argmax()
                    if (x, y) == self.pos:
                        if self.dir == self.actions.left:
                            ch = "<"
                        elif self.dir == self.actions.right:
                            ch = ">"
                        elif self.dir == self.actions.up:
                            ch = "^"
                        elif self.dir == self.actions.down:
                            ch = "v"
                    elif thing == self.cookbook.get_index("boundary"):
                        ch = 'X'
                    else:
                        ch = chr(97+thing)
                s += ch
            s += '\n'
        print(s)

    def print_inventory(self):
        print('Current inventory items:')
        for item, count in enumerate(self.inventory):
            if count > 0:
                print('{}: {}'.format(self.cookbook.index.get(item), count))
        print('----------')


def get_alphabets(recipe_path):
    cookbook = Cookbook(recipe_path)
    alphabets = []
    for item in cookbook.index:
        if 'none' in item or 'recycle' in item:
            continue
        alphabets.append(item)
    return alphabets


def random_free(grid, rand, width, height):
    pos = None
    while pos is None:
        (x, y) = (rand.randint(width), rand.randint(height))
        if grid[x, y, :].any():
            continue
        # check if nearby is occupied
        ns = neighbors((x,y), width, height)
        occupied = 0
        for n in ns:
            if grid[n[0], n[1], :].any():
                occupied += 1
        if occupied == len(ns):
            continue
        pos = (x, y)
    return pos


def sample_craft_env(args, width=10, height=10, env_data=None):
    cookbook = Cookbook(args.recipe_path)
    rand = np.random.RandomState()
    # generate grid
    grid = np.zeros((width, height, cookbook.n_kinds+1))
    i_bd = cookbook.index["boundary"]
    grid[0, :, i_bd] = 1
    grid[width-1:, :, i_bd] = 1
    grid[:, 0, i_bd] = 1
    grid[:, height-1:, i_bd] = 1
    if env_data is None:
        # ingredients
        for primitive in cookbook.primitives:
            for i in range(2):
                (x, y) = random_free(grid, rand, width, height)
                grid[x, y, primitive] = 1
        # generate crafting stations
        for env in cookbook.environment:
            if env == cookbook.get_index('boundary'):
                continue
            (x, y) = random_free(grid, rand, width, height)
            grid[x, y, env] = 1
        # generate init pos
        init_pos = random_free(grid, rand, width, height)
        init_dir = rand.randint(4)
    else:
        grid, init_pos, init_dir = env_data
    # return the env
    return CraftWorldEnv(args.recipe_path,
                         init_pos, init_dir, grid,
                         width=width, height=height,
                         use_gui=args.use_gui,
                         is_headless=args.is_headless,
                         time_limit=args.num_steps,
                         target_fps=args.target_fps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Craft world')
    args = parser.parse_args()
    args.recipe_path = 'craft_recipes_basic.yaml'
    args.num_steps = 25
    args.target_fps = 60
    args.use_gui = True
    args.is_headless = False
    env = sample_craft_env(args)
    while True:
        env.gui.draw(move_first=True)
        feature = env.feature()
        img = Image.fromarray(feature[0])
        img.save('tmp_images/feature.png')
