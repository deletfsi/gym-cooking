# Recipe planning
from recipe_planner.stripsworld import STRIPSWorld
import recipe_planner.utils as recipe
from recipe_planner.recipe import *

# Delegation planning
from delegation_planner.bayesian_delegator import BayesianDelegator

# Navigation planning
import navigation_planner.utils as nav_utils

# Other core modules
from utils.interact import interact
from utils.world import World
from utils.core import *
from utils.agent import SimAgent
from misc.game.gameimage import GameImage
from utils.agent import COLORS

import copy
import networkx as nx
import numpy as np
from itertools import combinations, permutations, product
from collections import namedtuple

import gym
from gym import error, spaces, utils
from gym.utils import seeding


CollisionRepr = namedtuple("CollisionRepr", "time agent_names agent_locations")


class OvercookedEnvironment(gym.Env):
    """Environment object for Overcooked."""

    def __init__(self, arglist):
        self.arglist = arglist
        self.t = 0
        self.set_filename()

        # For visualizing episode.
        self.rep = []

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False

    def get_repr(self):
        '''
       (
            # --- world representation ---
            (ObjectRepr(name='ChoppedLettuce', location=(4, 3), is_held=True),),
            (ObjectRepr(name='Plate', location=(1, 5), is_held=False),),
            (ObjectRepr(name='FreshTomato', location=(5, 1), is_held=False),),
            # --- agent representation ---
            AgentRepr(name='agent-1', location=(2, 2), holding='None'),
            AgentRepr(name='agent-2', location=(4, 3), holding='ChoppedLettuce')
        )
        '''
        return self.world.get_repr() + tuple([agent.get_repr() for agent in self.sim_agents])



    def __str__(self):
        # Print the world and agents.
        ''' # self.rep 是类似 [['-', '-', 't', '-'], [' ', '1', ' '], ...] 的二维列表

        # 1. 内层 map: `map(lambda y: y + ' ', x)`
        #    - 对 self.rep 中的每一行 `x` (例如 ['-', '-', 't', '-']) 进行处理。
        #    - `lambda y: y + ' '` 这个函数会给行中的每个元素 `y` (例如 '-') 后面加上一个空格。
        #    - 结果：对于第一行，会得到一个迭代器，产生 '-', ' ', '-', ' ', 't', ' ', '-', ' '

        # 2. 内层 ''.join(...)
        #    - 将上面加了空格的元素连接成一个字符串。
        #    - 结果：对于第一行，会得到字符串 "- - t - "

        # 3. 外层 list(map(lambda x: ..., self.rep))
        #    - 将步骤 1 和 2 应用到 self.rep 的每一行。
        #    - 结果 (_display) 会是一个包含格式化后每行字符串的列表：
        #      ['- - t - - ', '  1   /  ', '- * p   2 ']
        _display = list(map(lambda x: ''.join(map(lambda y: y + ' ', x)), self.rep))

        # 4. 外层 '\n'.join(_display)
        #    - 用换行符 ('\n') 将列表 _display 中的所有行字符串连接起来。
        #    - 最终返回一个多行字符串。
        
        - - t - -
        1   /
        - * p   2

        '''
        _display = list(map(lambda x: ''.join(map(lambda y: y + ' ', x)), self.rep))
        return '\n'.join(_display)

    def __eq__(self, other):
        return self.get_repr() == other.get_repr()

    def __copy__(self):
        new_env = OvercookedEnvironment(self.arglist)
        new_env.__dict__ = self.__dict__.copy()
        new_env.world = copy.copy(self.world)
        new_env.sim_agents = [copy.copy(a) for a in self.sim_agents]
        new_env.distances = self.distances

        # Make sure new objects and new agents' holdings have the right pointers.
        for a in new_env.sim_agents:
            if a.holding is not None:
                a.holding = new_env.world.get_object_at(
                        location=a.location,
                        desired_obj=None,
                        find_held_objects=True)
        return new_env

    def set_filename(self):
        self.filename = "{}_agents{}_seed{}".format(self.arglist.level,\
            self.arglist.num_agents, self.arglist.seed)
        model = ""
        if self.arglist.model1 is not None:
            model += "_model1-{}".format(self.arglist.model1)
        if self.arglist.model2 is not None:
            model += "_model2-{}".format(self.arglist.model2)
        if self.arglist.model3 is not None:
            model += "_model3-{}".format(self.arglist.model3)
        if self.arglist.model4 is not None:
            model += "_model4-{}".format(self.arglist.model4)
        self.filename += model

    def load_level(self, level, num_agents):
        '''这个方法的核心功能是从一个 .txt 格式的关卡文件中读取信息，并在模拟世界 (self.world) 中构建对应的地图布局、对象、食谱和智能体初始状态
        地图布局 (Phase 1): 使用特定字符表示不同的格子类型和初始物品。
            -: 工作台 (Counter)
            /: 切菜板 (Cutboard)
            *: 递送点 (Delivery)
            (空格): 地板 (Floor)
            t, l, o, p: 分别代表放在工作台上的番茄 (Tomato)、生菜 (Lettuce)、洋葱 (Onion)、盘子 (Plate)。
        
        食谱列表 (Phase 2): 每行一个食谱类的名称 (例如 Salad, SimpleTomato)。
        智能体初始位置 (Phase 3): 每行一个智能体的初始坐标 x y (例如 2 1)。
        
        
        '''
        x = 0
        y = 0
        with open('utils/levels/{}.txt'.format(level), 'r') as file:
            # Mark the phases of reading.
            phase = 1
            for line in file:
                line = line.strip('\n')
                if line == '':
                    phase += 1

                # Phase 1: Read in kitchen map.
                elif phase == 1:
                    for x, rep in enumerate(line):
                        # Object, i.e. Tomato, Lettuce, Onion, or Plate.
                        if rep in 'tlop':
                            counter = Counter(location=(x, y))
                            obj = Object(
                                    location=(x, y),
                                    contents=RepToClass[rep]())
                            counter.acquire(obj=obj)
                            self.world.insert(obj=counter)
                            self.world.insert(obj=obj)
                        # GridSquare, i.e. Floor, Counter, Cutboard, Delivery.
                        elif rep in RepToClass:
                            newobj = RepToClass[rep]((x, y))
                            self.world.objects.setdefault(newobj.name, []).append(newobj)
                        else:
                            # Empty. Set a Floor tile.
                            f = Floor(location=(x, y))
                            self.world.objects.setdefault('Floor', []).append(f)
                    y += 1
               
                # Phase 2: Read in recipe list.
                 # ========= Phase 2: 读取食谱列表 =========
                elif phase == 2:
                    # 假设读到行 "Salad"
                    # globals()[line]() 会查找名为 "Salad" 的类并调用其构造函数 Salad()
                    # 创建的食谱对象被添加到 self.recipes 列表中
                    self.recipes.append(globals()[line]())

                # Phase 3: Read in agent locations (up to num_agents).
                 # ========= Phase 3: 读取智能体初始位置 =========
                elif phase == 3:
                    if len(self.sim_agents) < num_agents:
                        loc = line.split(' ')
                        sim_agent = SimAgent(
                                name='agent-'+str(len(self.sim_agents)+1),
                                id_color=COLORS[len(self.sim_agents)],
                                location=(int(loc[0]), int(loc[1])))
                        self.sim_agents.append(sim_agent)

        self.distances = {}
        self.world.width = x+1
        self.world.height = y
        self.world.perimeter = 2*(self.world.width + self.world.height)


    def reset(self):
        self.world = World(arglist=self.arglist)
        self.recipes = []
        self.sim_agents = []
        self.agent_actions = {}
        self.t = 0

        # For visualizing episode.
        self.rep = []

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False

        # Load world & distances.
        self.load_level(
                level=self.arglist.level,
                num_agents=self.arglist.num_agents)
        
        ''' 生成所有可能的子任务 (Subtasks)
        #    调用 run_recipes() (这个方法通常在 OvercookedEnvironment 或其父类中定义，
        #    它会使用 recipe_planner 来分析加载的食谱，生成完成食谱所需的所有子任务列表)
        #    这对应论文中的 T (set of sub-tasks) 
        #    示例：对于 Salad 食谱，all_subtasks 可能包含 [Get(Tomato), Get(Lettuce),
        #    Get(Plate), Chop(Tomato), Chop(Lettuce), Merge(Tomato, Plate),
        #    Merge(Lettuce, Plate), Merge(Tomato-Plate, Lettuce), Deliver(...)] 等
        '''
        self.all_subtasks = self.run_recipes()
        
        '''保存初始状态的副本作为 "上一步的观察" (obs_tm1)
        #    这对于 Bayesian Delegation 的推理很重要，因为它需要 t-1 时刻的状态
        #    来计算 t-1 时刻采取动作的似然度。在 reset 时刻，obs_tm1 就是初始状态。'''
        self.world.make_loc_to_gridsquare()
        self.world.make_reachability_graph()
        self.cache_distances()
        self.obs_tm1 = copy.copy(self)

        '''初始化图像处理/记录功能 (如果启用)'''
        if self.arglist.record or self.arglist.with_image_obs:
            self.game = GameImage(
                    filename=self.filename,
                    world=self.world,
                    sim_agents=self.sim_agents,
                    record=self.arglist.record)
            self.game.on_init()
            if self.arglist.record:
                self.game.save_image_obs(self.t)

        return copy.copy(self)

    def close(self):
        return

    def step(self, action_dict):
        # Track internal environment info.
        self.t += 1
        print("===============================")
        print("[environment.step] @ TIMESTEP {}".format(self.t))
        print("===============================")

        # Get actions.
        for sim_agent in self.sim_agents:
            sim_agent.action = action_dict[sim_agent.name]

        # Check collisions.
        self.check_collisions()
        
        ''' 保存 "上一步观察"：*在碰撞检测之后，实际执行动作之前* 保存环境状态副本。
        #    这非常重要，因为 Bayesian Delegation 的信念更新 P(a_{t-1}|s_{t-1}, ta)
        #    需要的是 *实际执行* 的动作 a_{t-1} (碰撞处理后的) 和
        #    执行动作 *之前* 的状态 s_{t-1} (即这里的 obs_tm1)。'''
        self.obs_tm1 = copy.copy(self)

        # Execute.
        '''执行导航/交互：根据最终确定的动作 (可能被碰撞修改过) 更新世界状态。
        #    execute_navigation() 内部会为每个智能体调用 interact(agent, world)。
        #    interact() 会根据智能体的动作和目标格子的状态来改变智能体的位置、
        #    持有物状态，以及世界中物品的状态 (例如拾取、放下、合并、切割)。'''
        self.execute_navigation()

        # Visualize.
        self.display()
        self.print_agents()
        if self.arglist.record:
            self.game.save_image_obs(self.t)

        # Get a plan-representation observation.  
        new_obs = copy.copy(self)
        # Get an image observation
        image_obs = self.game.get_image_obs()

        '''检查回合是否结束并计算奖励 ,  done() 方法会检查是否所有必要的 Deliver 子任务都完成了，
        或者是否达到最大时间步。'''
        done = self.done()
        
        ''' reward() 方法通常很简单，如果回合成功结束 (self.successful 为 True)，返回 1，否则返回 0。'''
        reward = self.reward()
        
        
        info = {"t": self.t, "obs": new_obs,
                "image_obs": image_obs,
                "done": done, "termination_info": self.termination_info}
        
        '''返回 Gym 标准的 step 输出'''
        return new_obs, reward, done, info









    def done(self):
        """检查当前回合是否应该结束。

        回合结束的条件有两个：
        1. 达到最大时间步限制。
        2. 所有必要的 'Deliver' 子任务都已完成。

        Returns:
            bool: 如果回合应该结束，则返回 True，否则返回 False。
        """
        
        # Done if the episode maxes out
        if self.t >= self.arglist.max_num_timesteps and self.arglist.max_num_timesteps:
            self.termination_info = "Terminating because passed {} timesteps".format(
                    self.arglist.max_num_timesteps)
            self.successful = False   # 超时意味着未成功完成任务目标
            return True


        ''' 断言：确保 self.all_subtasks (在 reset 时生成) 中至少包含一个 Deliver 类型的子任务。
        # 这是为了保证环境总有一个明确的“完成”目标。
        # 如果没有 Deliver 任务，环境可能永远不会正常结束。'''
        assert any([isinstance(subtask, recipe.Deliver) for subtask in self.all_subtasks]), "no delivery subtask"


        ''' # 条件2: 检查是否所有 Deliver 子任务都已完成
        # 遍历在 reset() 时通过 run_recipes() 生成的所有子任务列表 self.all_subtasks'''
        # Done if subtask is completed.
        for subtask in self.all_subtasks:
            # Double check all goal_objs are at Delivery.
            if isinstance(subtask, recipe.Deliver):
                _, goal_obj = nav_utils.get_subtask_obj(subtask)

                delivery_loc = list(filter(lambda o: o.name=='Delivery', self.world.get_object_list()))[0].location
                
                ''' 获取当前世界中所有 goal_obj (目标物品) 实例的位置列表
                # self.world.get_all_object_locs 会查找所有匹配 goal_obj 的对象实例的位置'''
                goal_obj_locs = self.world.get_all_object_locs(obj=goal_obj)
                
                ''' 如果没有任何一个 goal_obj 在 delivery_loc，意味着这个 Deliver 任务尚未完成'''
                if not any([gol == delivery_loc for gol in goal_obj_locs]):
                    self.termination_info = ""
                    self.successful = False
                    return False

        self.termination_info = "Terminating because all deliveries were completed"
        self.successful = True
        return True

    def reward(self):
        """计算并返回当前时间步的奖励。

        这是一个稀疏奖励函数：只有在回合成功结束时才给予奖励。
        对应论文 [cite: 141] 中提到的 R (reward function)。

        Returns:
            int: 成功则返回 1，否则返回 0。
            
        检查 self.successful 标志 (由 done() 方法在成功完成时设置为 True)
        """
        return 1 if self.successful else 0

    def print_agents(self):
        for sim_agent in self.sim_agents:
            sim_agent.print_status()

    def display(self):
        self.update_display()
        print(str(self))

    def update_display(self):
        self.rep = self.world.update_display()
        for agent in self.sim_agents:
            x, y = agent.location
            self.rep[y][x] = str(agent)


    def get_agent_names(self):
        return [agent.name for agent in self.sim_agents]

    def run_recipes(self):
        """Returns different permutations of completing recipes.
            为当前加载的食谱生成所有必要的子任务。

        它使用 STRIPSWorld 类来处理食谱规划，找出完成食谱目标
        所需的一系列子任务（STRIPS 动作）。
        这对应论文 [cite: 145, 148, 403] 中提到的将高级目标分解为子任务的过程。

        Returns:
            list: 包含完成所有加载食谱所需的所有子任务对象的列表。
                  注意：如果存在多个食谱或完成单个食谱有多种最短路径，
                  这里返回的是所有这些路径中涉及到的 *所有* 子任务的集合（去重可能由后续逻辑处理，
                  但这里的代码是直接扁平化）。
                  
                  
        """
        
        '''  # 1. 初始化 STRIPSWorld 对象
        #    - self.world: 传入当前的环境世界状态，STRIPSWorld 会从中提取初始状态谓词
        #      (例如，哪些物品是 Fresh 的)。
        #    - self.recipes: 传入在 load_level 中加载的食谱对象列表 (例如 [SimpleTomato()])。
        #      STRIPSWorld 会从食谱对象中获取目标谓词 (例如 Delivered(Plate-Tomato))
        #      和完成该食谱允许使用的 STRIPS 动作 (例如 Get, Chop, Merge, Deliver)。'''
        self.sw = STRIPSWorld(world=self.world, recipes=self.recipes)
       
       
        # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions
        ''' # 2. 调用 STRIPSWorld 的 get_subtasks 方法进行规划
        #    - max_path_length: 限制规划器搜索的动作序列（子任务路径）的最大长度。
        #    - get_subtasks 内部会进行类似图搜索的规划 (论文附录提到使用 BFS [cite: 402])，
        #      寻找从初始状态到达目标状态的最短动作序列(或所有最短序列)。
        #    - 返回的 subtasks 是一个列表，其中每个元素是对应一个食谱的一条或多条
        #      最短子任务路径 (每条路径也是一个列表)。
        #      例如，如果只有一个食谱且只有一条最短路径，可能是 [[task1, task2, task3]]
        #      如果有多个食谱，可能是 [[recipe1_task1,...], [recipe2_task1,...]]'''
        subtasks = self.sw.get_subtasks(max_path_length=self.arglist.max_num_subtasks)
        
       
        '''
        例子 1：只有一个食谱，且只有一条最短路径

        假设 self.recipes 只包含 SimpleTomato，并且 get_subtasks 找到的最短路径只有一条：
        SimpleTomato =>  [Get('Tomato'), Get('Plate'), Chop('Tomato'), Merge('Tomato', 'Plate'), Deliver('Plate-Tomato')]
        
        
        例子 2：有多个食谱，或者一个食谱有多条不同的最短路径 : SimpleTomato 和 SimpleLettuce ,
        并且 get_subtasks 为每个食谱找到了一条路径：
        
        # 为了简洁，我们用缩写表示子任务对象
            t1 = Get('Tomato')
            t2 = Get('Plate') # 注意盘子可能在两条路径中都需要
            t3 = Chop('Tomato')
            t4 = Merge('Tomato', 'Plate')
            t5 = Deliver('Plate-Tomato')

            l1 = Get('Lettuce')
            l2 = Get('Plate') # 同上
            l3 = Chop('Lettuce')
            l4 = Merge('Lettuce', 'Plate')
            l5 = Deliver('Plate-Lettuce')

            # get_subtasks 返回的 subtasks 可能是这样的结构：
            subtasks = [
                [t1, t2, t3, t4, t5],  # Path for SimpleTomato
                [l1, l2, l3, l4, l5]   # Path for SimpleLettuce
            ]

            # 最终结果 all_subtasks 会是：
            # [t1, t2, t3, t4, t5, l1, l2, l3, l4, l5]
            # 即：
            # [Get('Tomato'), Get('Plate'), Chop('Tomato'), Merge('Tomato', 'Plate'), Deliver('Plate-Tomato'), 
            # Get('Lettuce'), Get('Plate'), Chop('Lettuce'), Merge('Lettuce', 'Plate'), Deliver('Plate-Lettuce')]

            注意 需要重复  Get('Plate') ， 不可以合并哦
        ''' 
        all_subtasks = [subtask for path in subtasks for subtask in path]
        print('Subtasks:', all_subtasks, '\n')
        return all_subtasks






    def get_AB_locs_given_objs(self, subtask, subtask_agent_names, start_obj, goal_obj, subtask_action_obj):
        """Returns list of locations relevant for subtask's Merge operator.

        See Merge operator formalism in our paper, under Fig. 11:
        https://arxiv.org/pdf/2003.11778.pdf
        
        
        根据给定的子任务和相关对象，返回执行该子任务所需的两组相关位置列表 (A_locs, B_locs)。

        这对应于论文图11中描述的 Merge 操作符形式化 [cite: 146]。

        Args:
            subtask (Action): 当前要执行的子任务对象 (例如 Chop('Tomato'), Deliver('Plate-Tomato')).
            subtask_agent_names (tuple): 被分配执行此子任务的智能体名称元组 (例如 ('agent-1',), ('agent-1', 'agent-2')).
            start_obj (Object or list): 子任务开始时涉及的物品对象或对象列表。
                                        (由 nav_utils.get_subtask_obj 返回)
            goal_obj (Object): 子任务完成时期望得到的物品对象。
                               (由 nav_utils.get_subtask_obj 返回)
            subtask_action_obj (GridSquare): 执行子任务需要交互的固定对象 (例如 Cutboard, Delivery)。
                                          (由 nav_utils.get_subtask_action_obj 返回)

        Returns:
            tuple: 包含两个列表的元组 (A_locs, B_locs)。
                   A_locs: 主要物品的可能位置列表。
                   B_locs: 工具或目标地点的位置列表。
        """
         

        """For Merge operator on Chop subtasks, we look at objects that can be
        chopped and the cutting board objects.
        --- 情况 1: 子任务是 Chop (例如 Chop('Tomato')) ---
        对于 Chop 任务，A 是需要被切的物品 (番茄)，B 是切菜板"""
        if isinstance(subtask, recipe.Chop):
            
            # A: Object that can be chopped.
            '''  A 位置: 需要被切的物品 (start_obj，例如新鲜番茄) 的所有可能位置。
            # 包括：
            # 1. self.world.get_object_locs(obj=start_obj, is_held=False):
            #    在世界上所有 *未被持有* 的 start_obj 的位置。
            # 2. list(map(... filter(...))):
            #    找到所有 *被分配到这个任务* (名字在 subtask_agent_names 中)
            #    并且 *正拿着* start_obj 的智能体的位置 '''
            A_locs = self.world.get_object_locs(obj=start_obj, is_held=False) + list(map(lambda a: a.location,\
                list(filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj, self.sim_agents))))

            # B: Cutboard objects.
            '''B 位置: 执行动作所需的对象 (subtask_action_obj，即 Cutboard) 的所有位置'''
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)



        #  --- 情况 2: 子任务是 Deliver (例如 Deliver('Plate-Tomato')) ---
        # 对于 Deliver 任务，A 是需要被递送的物品 (盘子+番茄)，B 是递送点。
        # For Merge operator on Deliver subtasks, we look at objects that can be
        # delivered and the Delivery object 
        elif isinstance(subtask, recipe.Deliver):
            # B: Delivery objects.
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)

            # A: Object that can be delivered.
            ''' # A 位置: 需要被递送的物品 (start_obj，例如 Plate-Tomato) 的所有可能位置。
            # 计算方式与 Chop 类似：包括世界上未被持有的 + 被分配的智能体持有的。'''
            A_locs = self.world.get_object_locs(
                    obj=start_obj, is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj, self.sim_agents))))
            '''# 特殊处理：从 A_locs 中移除那些 *已经是* 递送点的位置。 
                因为如果物品已经在递送点，就不需要再把它“递送”到那里了。'''
            A_locs = list(filter(lambda a: a not in B_locs, A_locs))

        # For Merge operator on Merge subtasks, we look at objects that can be
        # combined together. These objects are all ingredient objects (e.g. Tomato, Lettuce).
          # --- 情况 3: 子任务是 Merge (例如 Merge('ChoppedTomato', 'Plate')) ---
        # 对于 Merge 任务，A 是第一个要合并的物品，B 是第二个要合并的物品。
        # start_obj 在这里是一个包含两个物品对象的列表，例如 [ChoppedTomatoObject, PlateObject]  
        
            # --- Merge 示例 ---
            # 假设: subtask=Merge('ChoppedTomato', 'Plate'), subtask_agent_names=('agent-1', 'agent-2')
            #       start_obj=[ChoppedTomatoObject, PlateObject]
            #       一个 ChoppedTomato 在 (0, 6) (未被持有)
            #       一个 Plate 在 (6, 6) (未被持有)
            #       agent-1 在 (1,1)，手上没东西
            #       agent-2 在 (5,5)，手上拿着一个 Plate
            # 计算 A_locs (ChoppedTomato):
            #   world.get_object_locs(ChoppedTomato, False) -> [(0,6)]
            #   filter(...) -> agent-1 或 agent-2 没拿 ChoppedTomato -> []
            #   A_locs = [(0,6)]
            # 计算 B_locs (Plate):
            #   world.get_object_locs(Plate, False) -> [(6,6)]
            #   filter(...) -> agent-2 拿着 Plate -> [agent-2]
            #   map(...) -> [(5,5)]
            #   B_locs = [(6,6)] + [(5,5)] = [(6,6), (5,5)]
            # 返回: ([(0,6)], [(6,6), (5,5)])          
        elif isinstance(subtask, recipe.Merge):
            '''  A 位置: 第一个物品 (start_obj[0]) 的所有可能位置 (世界 + 持有)。 '''
            A_locs = self.world.get_object_locs(
                    obj=start_obj[0], is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj[0], self.sim_agents))))
            
            '''# B 位置: 第二个物品 (start_obj[1]) 的所有可能位置 (世界 + 持有)'''
            B_locs = self.world.get_object_locs(
                    obj=start_obj[1], is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj[1], self.sim_agents))))

        else:
            return [], []

        return A_locs, B_locs




    def get_lower_bound_for_subtask_given_objs(
            self, subtask, subtask_agent_names, start_obj, goal_obj, subtask_action_obj):
        """
        计算给定子任务 (subtask) 在当前状态下，由指定智能体 (subtask_agent_names) 执行时，
        涉及到的相关对象之间的最短路径距离下界，并加入持有无关物品的惩罚。

        这个函数估算的是完成子任务的 *理论最小成本*，主要用于启发式搜索或计算先验概率，
        判断一个任务分配方案的“容易程度”。返回值越小越好。
        
        
        Return the lower bound distance (shortest path) under this subtask between objects."""
         # 断言：确保分配给任务的智能体数量不超过2个。当前代码只支持单智能体或双智能体协作。
        assert len(subtask_agent_names) <= 2, 'passed in {} agents but can only do 1 or 2'.format(len(agents))

        '''  --- 计算持有惩罚 (Holding Penalty) ---
        # 如果智能体在执行任务时，手里拿着与当前任务无关的东西，会增加额外的成本。
        # 例如，要去拿番茄 (Get Tomato)，但手里却拿着盘子。 
        # Calculate extra holding penalty if the object is irrelevant.'''
        holding_penalty = 0.0
        for agent in self.sim_agents:
            if agent.name in subtask_agent_names:
                 # 检查该智能体是否正持有物品 (agent.holding 不是 None)
                if agent.holding is not None:
                    # 对于 Merge 任务，智能体需要持有其中一个合并物品，所以不计算持有惩罚。
                    if isinstance(subtask, recipe.Merge):
                        continue
                    else:
                        '''# 检查智能体持有的物品是否 *既不是* 任务的起始物品 (start_obj)
                        # *也不是* 任务的目标物品 (goal_obj)。
                        # 例如，执行 Chop(Tomato)，start_obj 是 FreshTomato, goal_obj 是 ChoppedTomato。
                        # 如果智能体拿着 Plate，则 Plate != FreshTomato 且 Plate != ChoppedTomato，需要惩罚。
                        # 如果智能体拿着 FreshTomato，则 FreshTomato == start_obj，不需要惩罚。
                        # 如果智能体拿着 ChoppedTomato，则 ChoppedTomato == goal_obj，不需要惩罚 (虽然这可能意味着任务已完成或接近完成)。'''
                        if agent.holding != start_obj and agent.holding != goal_obj:
                            '''   如果持有的是无关物品，增加 1.0 的惩罚。这相当于增加了一步移动或交互的成本。'''
                            holding_penalty += 1.0
        
        
        ''' 
        --- 处理双智能体情况下的惩罚 ---
        # 如果是两个智能体协作，并且两个智能体都拿着无关物品，上面的循环会计算出 holding_penalty = 2.0。
        # 这里将其限制为最大 1.0，意味着即使两个人都拿错了东西，总惩罚也只是 1.0。
        # 目的是避免过度惩罚双人协作的情况。
        # Account for two-agents where we DON'T want to overpenalize.'''
        holding_penalty = min(holding_penalty, 1)

        # Get current agent locations.
        agent_locs = [agent.location for agent in list(filter(lambda a: a.name in subtask_agent_names, self.sim_agents))]
       
        '''# 调用 get_AB_locs_given_objs 函数，获取执行该子任务所需的两组关键位置列表：
        # A_locs: 通常是起始物品的可能位置 (例如，新鲜番茄在哪？可能在地上，可能在某个智能体手上)。
        # B_locs: 通常是工具或目标地点的位置 (例如，砧板在哪？递送点在哪？另一个要合并的物品在哪？)。
        # 这个函数内部会处理单/双智能体、不同任务类型 (Chop/Deliver/Merge) 的情况。'''
        A_locs, B_locs = self.get_AB_locs_given_objs(
                subtask=subtask,
                subtask_agent_names=subtask_agent_names,
                start_obj=start_obj,
                goal_obj=goal_obj,
                subtask_action_obj=subtask_action_obj)

        '''  # --- 计算并返回最终的成本下界 ---
        # 调用 self.world.get_lower_bound_between 函数计算核心的距离下界。
        # 这个函数会考虑智能体的当前位置 (agent_locs)、A物品的可能位置 (A_locs)、B物品/地点的可能位置 (B_locs)，
        # 以及子任务类型 (subtask)，利用预先计算好的 reachability_graph (可达性图) [cite: 440] 找到完成交互所需的最短路径距离。
        # 对于双智能体 Merge 任务，它还会特殊处理，考虑两个智能体分别移动到 A 和 B 的情况 [cite: 441]。
        # # Add together distance and holding_penalty.'''
        return self.world.get_lower_bound_between(
                subtask=subtask,
                agent_locs=tuple(agent_locs),
                A_locs=tuple(A_locs),
                B_locs=tuple(B_locs)) + holding_penalty






    def is_collision(self, agent1_loc, agent2_loc, agent1_action, agent2_action):
        """
        判断两个智能体根据它们各自的动作是否会发生碰撞。
        碰撞可能发生在智能体之间，或者智能体与世界中的固定障碍物之间。

        Args:
            agent1_loc (tuple): 智能体1的当前位置 (x, y)。
            agent2_loc (tuple): 智能体2的当前位置 (x, y)。
            agent1_action (tuple): 智能体1尝试执行的动作，通常是 (dx, dy)，例如 (0, 1), (-1, 0), (0, 0)。
            agent2_action (tuple): 智能体2尝试执行的动作。

        Returns:
            list: 一个包含两个布尔值的列表 [execute1, execute2]。
                  execute1 为 True 表示 agent1 的动作可以执行，False 表示因碰撞无法执行。
                  execute2 为 True 表示 agent2 的动作可以执行，False 表示因碰撞无法执行。
        """ 
        # Tracks whether agents can execute their action.
        execute = [True, True]

        # 计算智能体1尝试移动后的目标位置
        # Collision between agents and world objects.
        agent1_next_loc = tuple(np.asarray(agent1_loc) + np.asarray(agent1_action))
        if self.world.get_gridsquare_at(location=agent1_next_loc).collidable:
            # Revert back because agent collided.   如果目标格子是障碍物，智能体1无法移动到那里。
            agent1_next_loc = agent1_loc

        agent2_next_loc = tuple(np.asarray(agent2_loc) + np.asarray(agent2_action))
        if self.world.get_gridsquare_at(location=agent2_next_loc).collidable:
            # Revert back because agent collided.
            agent2_next_loc = agent2_loc

        # Inter-agent collision. 智能体之间的碰撞 
        # 两个智能体尝试移动到同一个最终位置 
        if agent1_next_loc == agent2_next_loc:
            if agent1_next_loc == agent1_loc and agent1_action != (0, 0):
                execute[1] = False
            elif agent2_next_loc == agent2_loc and agent2_action != (0, 0):
                execute[0] = False
            else:
                execute[0] = False
                execute[1] = False

        # Prevent agents from swapping places. 两个智能体尝试互换位置
        elif ((agent1_loc == agent2_next_loc) and
                (agent2_loc == agent1_next_loc)):
            execute[0] = False
            execute[1] = False
        return execute





    def check_collisions(self):
        """Checks for collisions and corrects agents' executable actions.

        Collisions can either happen amongst agents or between agents and world objects."""
        execute = [True for _ in self.sim_agents]

        # Check each pairwise collision between agents.
        for i, j in combinations(range(len(self.sim_agents)), 2):
            agent_i, agent_j = self.sim_agents[i], self.sim_agents[j]
            exec_ = self.is_collision(
                    agent1_loc=agent_i.location,
                    agent2_loc=agent_j.location,
                    agent1_action=agent_i.action,
                    agent2_action=agent_j.action)

            # Update exec array and set path to do nothing.
            if not exec_[0]:
                execute[i] = False
            if not exec_[1]:
                execute[j] = False

            '''# 如果 exec_ 不全是 True (即至少有一个智能体无法执行动作)，说明发生了碰撞。 
            Track collisions.'''
            if not all(exec_):
                collision = CollisionRepr(
                        time=self.t,
                        agent_names=[agent_i.name, agent_j.name],
                        agent_locations=[agent_i.location, agent_j.location])
                self.collisions.append(collision)

        print('\nexecute array is:', execute)

        # Update agents' actions if collision was detected.
        for i, agent in enumerate(self.sim_agents):
            '''  如果智能体 i 在 execute 列表中被标记为 False (即发生了碰撞，其动作无法执行)
             将该智能体的动作强制修改为 (0, 0)，即原地不动。'''
            if not execute[i]:
                agent.action = (0, 0)
            print("{} has action {}".format(color(agent.name, agent.color), agent.action))




    def execute_navigation(self):
        for agent in self.sim_agents:
            interact(agent=agent, world=self.world)
            self.agent_actions[agent.name] = agent.action




    def cache_distances(self):
        """Saving distances between world objects.
        在环境初始化时预先计算并缓存 (cache) 所有重要的静态格子（地板、各种柜台、递送点、切菜板）
        之间的最短路径距离。这样做是为了在后续的规划过程中（例如，在 E2E_BRTDP 中计算 Q 值或 V 值的下界，
        或者在 BayesianDelegator 中计算空间先验）能够快速查询任意两个关键位置之间的距离，
        而不需要每次都重新进行路径搜索，从而提高规划效率。
        高效的距离计算对于模型中的规划部分 (如 BRTDP ) 和启发式估计 (如用于计算先验概率的 Vt(s)的下界 ) 至关重要
        source_objs , dest_objs = Floor + 柜台、切板、递送点等
        """
        counter_grid_names = [name for name in self.world.objects if "Supply" in name or "Counter" in name or "Delivery" in name or "Cut" in name]
        # Getting all source objects.
        source_objs = copy.copy(self.world.objects["Floor"])
        for name in counter_grid_names:
            source_objs += copy.copy(self.world.objects[name])
        # Getting all destination objects.
        dest_objs = source_objs

        # From every source (Counter and Floor objects),
        # calculate distance to other nodes.
        for source in source_objs:
            self.distances[source.location] = {}
            # Source to source distance is 0.
            self.distances[source.location][source.location] = 0
            for destination in dest_objs:
                '''# --- 确定接近源/目标格子的可能“边” ---
                # reachability_graph 中的节点是 (location, edge_action) 对。
                # edge_action 表示从哪个方向接近或离开这个 location。
                # - 对于不可碰撞的格子 (如 Floor)，智能体可以直接“在”上面，所以接近它的边只有 (0, 0)。
                # - 对于可碰撞的格子 (如 Counter)，智能体不能进入，只能从相邻格子接近它，
                #   因此可能的接近方向是所有导航动作 World.NAV_ACTIONS (上/下/左/右)。
                # # Possible edges to approach source and destination.'''
                source_edges = [(0, 0)] if not source.collidable else World.NAV_ACTIONS
                destination_edges = [(0, 0)] if not destination.collidable else World.NAV_ACTIONS
                # Maintain shortest distance.
                shortest_dist = np.inf
                for source_edge, dest_edge in product(source_edges, destination_edges):
                    try:
                        dist = nx.shortest_path_length(self.world.reachability_graph, (source.location,source_edge), (destination.location, dest_edge))
                        # Update shortest distance.
                        if dist < shortest_dist:
                            shortest_dist = dist
                    except:
                        continue
                # Cache distance floor -> counter.
                self.distances[source.location][destination.location] = shortest_dist

        # Save all distances under world as well.
        self.world.distances = self.distances

