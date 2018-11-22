from pommerman.agents.docker_agent import *
from pommerman.agents import SimpleAgent
import pommerman
import numpy as np

def featurize(obs):
    # TODO: history of n moves?
    board = obs['board']

    # convert board items into bitmaps
    maps = [board == i for i in range(10)]
    maps.append(obs['bomb_blast_strength'])
    maps.append(obs['bomb_life'])

    # duplicate ammo, blast_strength and can_kick over entire map
    #创建一个由常数填充的数组,第一个参数是数组的形状，第二个参数是数组中填充的常数。
    maps.append(np.full(board.shape, obs['ammo']))
    maps.append(np.full(board.shape, obs['blast_strength']))
    maps.append(np.full(board.shape, obs['can_kick']))

    # add my position as bitmap
    position = np.zeros(board.shape)
    position[obs['position']] = 1
    maps.append(position)

    # add teammate
    if obs['teammate'] is not None:
        maps.append(board == obs['teammate'].value)
    else:
        maps.append(np.zeros(board.shape))

    # add enemies
    enemies = [board == e.value for e in obs['enemies']]
    maps.append(np.any(enemies, axis=0))

    #assert len(maps) == NUM_CHANNELS
    return np.stack(maps, axis=2)


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
# obs = json.dumps(state[0], cls=utility.PommermanJSONEncoder)
# # print(obs)
# # print(type(obs))
# observation = json.loads(obs)
# print(observation)
# print(len(observation['board']))
# print(np.array((observation['board'])))
# print(np.array((observation['board'])).shape)
#
# print(len(observation['bomb_blast_strength']))

map = featurize(s)
print(map.shape)
print(map[:][0][0].shape)
