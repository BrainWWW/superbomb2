from pommerman.agents.docker_agent import *
from pommerman.agents import SimpleAgent
import pommerman
import numpy as np

def featurize(obs):
    # TODO: history of n moves?
    board = obs['board']
    board = np.array(board)

    # convert board items into bitmaps
    maps = [board == i for i in range(10)]
    maps.append(np.array(obs['bomb_blast_strength']))
    maps.append(np.array(obs['bomb_life']))

    # duplicate ammo, blast_strength and can_kick over entire map
    #创建一个由常数填充的数组,第一个参数是数组的形状，第二个参数是数组中填充的常数。
    maps.append(np.full(board.shape, obs['ammo']))
    maps.append(np.full(board.shape, obs['blast_strength']))
    maps.append(np.full(board.shape, obs['can_kick']))

    # add my position as bitmap
    position = np.zeros(board.shape)
    position[tuple(obs['position'])] = 1
    maps.append(position)

    # add teammate
    if obs['teammate'] is not None:
        maps.append(board == obs['teammate'])
    else:
        maps.append(np.zeros(board.shape))

    # add enemies
    enemies = [board == e for e in obs['enemies']]
    maps.append(np.any(enemies, axis=0))

    #assert len(maps) == NUM_CHANNELS
    return np.stack(maps, axis=2)

def available_action(state):
    # 可行动的动作为0，不可行动的动作为-9999
    position = state['position']
    board = np.array(state['board'])
    x = position[0]
    y = position[1]
    if state['can_kick']:
        avail_path = [0,3,6,7,8]
    else:
        avail_path = [0,6,7, 8]
    action = [0, -9999, -9999, -9999, -9999, 0]
    if state['ammo'] == 0:
        action[-1] = -9999
    if (x - 1) >= 0:
        if board[x-1,y] in avail_path:
            #可以往上边走
            action[1] = 0
    if (x + 1) <= 10:
        if board[x+1, y] in avail_path:
            #可以往下边走
            action[2] = 0
    if (y - 1) >= 0:
        if board[x,y-1] in avail_path:
            #可以往坐边走
            action[3] = 0
    if (y + 1) <= 10:
        if board[x,y+1] in avail_path:
            #可以往右边走
            action[4] = 0
    return action

agent_list = [
    SimpleAgent(),
    SimpleAgent(),
    SimpleAgent(),
    SimpleAgent()
]

env = pommerman.make('PommeTeamCompetition-v0', agent_list)
state = env.reset()
s = state[0]
print(s)
print(s['board'].shape)
obs = json.dumps(state[0], cls=utility.PommermanJSONEncoder)
# print(obs)
# print(type(obs))
observation = json.loads(obs)
print(observation)
print(len(observation['board']))
print(np.array((observation['board'])))
print(np.array((observation['board'])).shape)

print(available_action(observation))
# print(len(observation['bomb_blast_strength']))
#
# map = featurize(observation)
# print(map.shape)
# print(map[0])
