# Recipe planning
from recipe_planner.stripsworld import STRIPSWorld
import recipe_planner.utils as recipe_utils
from recipe_planner.utils import *

# Delegation planning
from delegation_planner.bayesian_delegator import BayesianDelegator

# Navigation planner
from navigation_planner.planners.e2e_brtdp import E2E_BRTDP
import navigation_planner.utils as nav_utils

# Other core modules
from utils.core import Counter, Cutboard
from utils.utils import agent_settings

import numpy as np
import copy
from termcolor import colored as color
from collections import namedtuple

AgentRepr = namedtuple("AgentRepr", "name location holding")

# Colors for agents.
COLORS = ['blue', 'magenta', 'yellow', 'green']


class RealAgent:
    """Real Agent object that performs task inference and plans."""

    def __init__(self, arglist, name, id_color, recipes):
        self.arglist = arglist
        self.name = name
        self.color = id_color
        self.recipes = recipes

        # Bayesian Delegation.
        self.reset_subtasks()
        self.new_subtask = None
        self.new_subtask_agent_names = []
        self.incomplete_subtasks = []
        self.signal_reset_delegator = False
        self.is_subtask_complete = lambda w: False
        self.beta = arglist.beta
        self.none_action_prob = 0.5

        self.model_type = agent_settings(arglist, name)
        if self.model_type == "up":
            self.priors = 'uniform'
        else:
            self.priors = 'spatial'

        # Navigation planner.
        self.planner = E2E_BRTDP(
                alpha=arglist.alpha,
                tau=arglist.tau,
                cap=arglist.cap,
                main_cap=arglist.main_cap)

    def __str__(self):
        return color(self.name[-1], self.color)

    def __copy__(self):
        a = Agent(arglist=self.arglist,
                name=self.name,
                id_color=self.color,
                recipes=self.recipes)
        a.subtask = self.subtask
        a.new_subtask = self.new_subtask
        a.subtask_agent_names = self.subtask_agent_names
        a.new_subtask_agent_names = self.new_subtask_agent_names
        a.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            a.holding = copy.copy(self.holding)
        return a

    def get_holding(self):
        if self.holding is None:
            return 'None'
        return self.holding.full_name

    def select_action(self, obs):
        """Return best next action for this agent given observations."""
        sim_agent = list(filter(lambda x: x.name == self.name, obs.sim_agents))[0]
        self.location = sim_agent.location
        self.holding = sim_agent.holding
        self.action = sim_agent.action

        if obs.t == 0:
            self.setup_subtasks(env=obs)

        # Select subtask based on Bayesian Delegation.
         # 更新对其他智能体意图的信念，并决定自己下一步应该执行哪个子任务
        self.update_subtasks(env=obs)
        
        # 从 Delegator 获取最可能的子任务分配中，分配给自己的子任务和执行者
        self.new_subtask, self.new_subtask_agent_names = self.delegator.select_subtask(
                agent_name=self.name)
        
        # 根据选择的子任务 (new_subtask) 和执行者，规划具体的导航动作
        self.plan(copy.copy(obs))
        
         # self.action 已经被 plan() 方法更新为新规划出的动作
        return self.action





    def get_subtasks(self, world):
        """Return different subtask permutations for recipes."""
        self.sw = STRIPSWorld(world, self.recipes)
        # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions.
        subtasks = self.sw.get_subtasks(max_path_length=self.arglist.max_num_subtasks)
        all_subtasks = [subtask for path in subtasks for subtask in path]

        # Uncomment below to view graph for recipe path i
        # i = 0
        # pg = recipe_utils.make_predicate_graph(self.sw.initial, recipe_paths[i])
        # ag = recipe_utils.make_action_graph(self.sw.initial, recipe_paths[i])
        return all_subtasks

    def setup_subtasks(self, env):
        """
        在回合开始时 (t=0) 初始化子任务列表和贝叶斯委托器。

        Args:
            env (OvercookedEnvironment): 当前环境对象。
            
            Initializing subtasks and subtask allocator, Bayesian Delegation
        """ 
        self.incomplete_subtasks = self.get_subtasks(world=env.world)
        self.delegator = BayesianDelegator(
                agent_name=self.name,
                all_agent_names=env.get_agent_names(),
                model_type=self.model_type,
                planner=self.planner,
                none_action_prob=self.none_action_prob)

    def reset_subtasks(self):
        """  重置当前智能体正在执行的子任务相关状态。
        当子任务完成或信念需要重置时调用。
        Reset subtasks---relevant for Bayesian Delegation."""
        self.subtask = None
        self.subtask_agent_names = []
        self.subtask_complete = False




    def refresh_subtasks(self, world):
        """ 在每个时间步结束时调用，检查当前子任务是否已完成，并更新未完成子任务列表。Refresh subtasks---relevant for Bayesian Delegation."""
        # Check whether subtask is complete.
        self.subtask_complete = False
        if self.subtask is None or len(self.subtask_agent_names) == 0:
            print("{} has no subtask".format(color(self.name, self.color)))
            return
        self.subtask_complete = self.is_subtask_complete(world)
        print("{} done with {} according to planner: {}\nplanner has subtask {} with subtask object {}".format(
            color(self.name, self.color),
            self.subtask, self.is_subtask_complete(world),
            self.planner.subtask, self.planner.goal_obj))

        # Refresh for incomplete subtasks.
        if self.subtask_complete:
            if self.subtask in self.incomplete_subtasks:
                self.incomplete_subtasks.remove(self.subtask)
                self.subtask_complete = True
        print('{} incomplete subtasks:'.format(
            color(self.name, self.color)),
            ', '.join(str(t) for t in self.incomplete_subtasks))




    def update_subtasks(self, env):
        """更新贝叶斯委托器的信念状态 (概率分布 P(ta))。
        根据上一步的观察和动作进行贝叶斯更新，或在必要时重置先验。
        Update incomplete subtasks---relevant for Bayesian Delegation."""
        
        # --- 判断是否需要重置先验概率 ---
        # 条件1: 当前子任务 (self.subtask) 已经完成 (即不在 incomplete_subtasks 列表中)。
        # 条件2: Delegator 认为需要重置 (通常是可行的任务分配假设空间发生了变化)。
        #   调用 self.delegator.should_reset_priors 进行检查。
        if ((self.subtask is not None and self.subtask not in self.incomplete_subtasks)
                or (self.delegator.should_reset_priors(obs=copy.copy(env),
                            incomplete_subtasks=self.incomplete_subtasks))):
            self.reset_subtasks()
            self.delegator.set_priors(
                    obs=copy.copy(env),
                    incomplete_subtasks=self.incomplete_subtasks,
                    priors_type=self.priors)
        else:
            if self.subtask is None:
                self.delegator.set_priors(
                    obs=copy.copy(env),
                    incomplete_subtasks=self.incomplete_subtasks,
                    priors_type=self.priors)
            else:
                self.delegator.bayes_update(
                        obs_tm1=copy.copy(env.obs_tm1),
                        actions_tm1=env.agent_actions,
                        beta=self.beta)

    def all_done(self):
        """检查该智能体是否认为所有任务都已完成。
        完成的标准是：未完成子任务列表中不再有 Deliver 类型的子任务。
        Return whether this agent is all done.
        An agent is done if all Deliver subtasks are completed."""
        if any([isinstance(t, Deliver) for t in self.incomplete_subtasks]):
            return False
        return True

    def get_action_location(self):
        """Return location if agent takes its action---relevant for navigation planner."""
        return tuple(np.asarray(self.location) + np.asarray(self.action))

    def plan(self, env, initializing_priors=False):
        """ 使用导航规划器 (self.planner) 为当前选定的子任务规划下一个动作。
        处理 Level 0 和 Level 1 规划。
        Plan next action---relevant for navigation planner."""
        print('right before planning, {} had old subtask {}, new subtask {}, subtask complete {}'.format(self.name, self.subtask, self.new_subtask, self.subtask_complete))

        # Check whether this subtask is done.
        if self.new_subtask is not None:
             # 定义用于检查该新子任务是否完成的 
            self.def_subtask_completion(env=env)

        # If subtask is None, then do nothing.
        if (self.new_subtask is None) or (not self.new_subtask_agent_names):
            actions = nav_utils.get_single_actions(env=env, agent=self)
            probs = []
            for a in actions:
                if a == (0, 0):
                    probs.append(self.none_action_prob)
                else:
                    probs.append((1.0-self.none_action_prob)/(len(actions)-1))
            self.action = actions[np.random.choice(len(actions), p=probs)]
        # Otherwise, plan accordingly.
        else:
            '''如果是 'greedy' 模型或者正在初始化先验，则不考虑其他智能体 (Level 0) ''' 
            if self.model_type == 'greedy' or initializing_priors:
                other_agent_planners = {}
            else:
                # Determine other agent planners for level 1 planning.
                # Other agent planners are based on your planner---agents never
                # share planners.
                '''# 否则，进行 Level 1 规划，需要获取其他智能体的规划器模型
                如果 new_subtask 为 None (发生重置后)，则使用上一个 subtask 作为备用
                调用 Delegator 获取其他智能体的规划器（基于当前信念）'''
                backup_subtask = self.new_subtask if self.new_subtask is not None else self.subtask
                other_agent_planners = self.delegator.get_other_agent_planners(
                        obs=copy.copy(env), backup_subtask=backup_subtask)

            print("[ {} Planning ] Task: {}, Task Agents: {}".format(
                self.name, self.new_subtask, self.new_subtask_agent_names))


            '''other_agent_planners 取决于 greedy 还是其他，
            调用 E2E_BRTDP 的 get_next_action 方法进行规划'''
            action = self.planner.get_next_action(
                    env=env, subtask=self.new_subtask,
                    subtask_agent_names=self.new_subtask_agent_names,
                    other_agent_planners=other_agent_planners)

            # If joint subtask, pick your part of the simulated joint plan.
            if self.name not in self.new_subtask_agent_names and self.planner.is_joint:
                self.action = action[0] #原地不动 
            else:
                self.action = action[self.new_subtask_agent_names.index(self.name)] if self.planner.is_joint else action

        # Update subtask.
        # 将 new_subtask 和 new_subtask_agent_names 设为当前的 subtask 和 subtask_agent_names
        self.subtask = self.new_subtask
        self.subtask_agent_names = self.new_subtask_agent_names
        
         # 清空 new_subtask 状态，为下一轮选择做准备
        self.new_subtask = None
        self.new_subtask_agent_names = []

        print('{} proposed action: {}\n'.format(self.name, self.action))




    def def_subtask_completion(self, env):
        '''
        例子 1: 子任务为 Chop(Tomato)
        假设 RealAgent 的 self.new_subtask 被设置为 recipe.Chop('Tomato')。当 plan() 方法调用 def_subtask_completion(env) 时：
                
                获取相关对象:
                    nav_utils.get_subtask_obj(subtask=recipe.Chop('Tomato')) 会返回:
                    self.start_obj: 一个代表新鲜番茄 (FreshTomato) 的 Object 实例。
                    self.goal_obj: 一个代表切好的番茄 (ChoppedTomato) 的 Object 实例。
                    nav_utils.get_subtask_action_obj(subtask=recipe.Chop('Tomato')) 会返回:
                    self.subtask_action_object: 一个代表切菜板 (Cutboard) 的 GridSquare 实例。 (虽然在这个完成条件检查中不直接使用)。
                    
            进入 else 分支:   因为 self.new_subtask (即 Chop('Tomato')) 不是 Deliver 类型。
                    
                计算初始目标物品数量:
                    self.cur_obj_count = len(env.world.get_all_object_locs(obj=self.goal_obj))
                    这会计算在 当前 环境 env 中，有多少个切好的番茄 (ChoppedTomato) 对象。
                    假设在调用 def_subtask_completion 时，世界上还没有切好的番茄，那么 self.cur_obj_count 会被设置为 0。

            
                定义完成检查函数:
                    self.is_subtask_complete = lambda w: len(w.get_all_object_locs(obj=self.goal_obj)) > self.cur_obj_count
                    这里定义了一个 lambda 函数，它接受一个未来的世界状态 w 作为参数。
                    这个函数会计算在 未来世界 w 中切好的番茄 (self.goal_obj) 的总数，然后检查这个数量是否 大于 self.cur_obj_count (在这个例子中是 0)。
                    总结 (Chop): 对于 Chop(Tomato) 任务，def_subtask_completion 定义了一个检查：当未来某个时刻世界上的切好番茄数量从 0 变为 1 (或更多) 时，任务即被视为完成



        例子 2: 子任务为 Deliver('Plate-Tomato')
        
            获取相关对象:

                nav_utils.get_subtask_obj(subtask=recipe.Deliver('Plate-Tomato')) 会返回:
                self.start_obj: 一个代表装有切好番茄的盘子 (Plate-Tomato) 的 Object 实例 (这是需要递送的物品)。
                self.goal_obj: 同样是代表装有切好番茄的盘子 (Plate-Tomato) 的 Object 实例 (因为递送后物品本身不变，只是位置变了)。
                nav_utils.get_subtask_action_obj(subtask=recipe.Deliver('Plate-Tomato')) 会返回:
                self.subtask_action_object: 一个代表递送点 (Delivery) 的 GridSquare 实例。

                nav_utils.get_subtask_action_obj(subtask=recipe.Deliver('Plate-Tomato')) 会返回:
                self.subtask_action_object: 一个代表递送点 (Delivery) 的 GridSquare 实例。

            进入 if isinstance(self.new_subtask, Deliver): 分支: 因为子任务是 Deliver 类型

            计算初始目标物品在递送点的数量:
                self.cur_obj_count = len(list(filter(lambda o: o in set(env.world.get_all_object_locs(self.subtask_action_object)), env.world.get_object_locs(obj=self.goal_obj, is_held=False))))
                这会计算在 当前 环境 env 中，有多少个 未被持有 的 Plate-Tomato 对象 (self.goal_obj) 正好位于递送点 (self.subtask_action_object) 的位置上。
                假设在调用 def_subtask_completion 时，还没有 Plate-Tomato 被送到递送点，那么 self.cur_obj_count 会被设置为 0
       
            定义辅助检查函数 has_more_obj:
                self.has_more_obj = lambda x: int(x) > self.cur_obj_count
                这个 lambda 检查输入的数量 x 是否大于初始计数 self.cur_obj_count (0)。
       
            定义完成检查函数 is_subtask_complete:
                self.is_subtask_complete = lambda w: self.has_more_obj(len(list(filter(lambda o: o in set(w.get_all_object_locs(obj=self.subtask_action_object)), w.get_object_locs(obj=self.goal_obj, is_held=False)))))
                这个 lambda 函数接受一个未来的世界状态 w。
                它会计算在 未来世界 w 中，位于递送点 (self.subtask_action_object) 的、未被持有的 Plate-Tomato (self.goal_obj) 的数量。
                然后用 self.has_more_obj 检查这个数量是否 大于 初始计数 self.cur_obj_count (0)。

        '''
        # Determine desired objects.
        self.start_obj, self.goal_obj = nav_utils.get_subtask_obj(subtask=self.new_subtask)
        self.subtask_action_object = nav_utils.get_subtask_action_obj(subtask=self.new_subtask)

        # Define termination conditions for agent subtask.
        # For Deliver subtask, desired object should be at a Deliver location.
         # 对于 Deliver 任务
        if isinstance(self.new_subtask, Deliver):
            ''' # 获取当前在递送点 (subtask_action_object) 的目标物品 (goal_obj) 的数量 '''
            self.cur_obj_count = len(list(
                filter(lambda o: o in set(env.world.get_all_object_locs(self.subtask_action_object)),
                env.world.get_object_locs(obj=self.goal_obj, is_held=False))))
            self.has_more_obj = lambda x: int(x) > self.cur_obj_count
            self.is_subtask_complete = lambda w: self.has_more_obj(
                    len(list(filter(lambda o: o in
                set(env.world.get_all_object_locs(obj=self.subtask_action_object)),
                w.get_object_locs(obj=self.goal_obj, is_held=False)))))
        # Otherwise, for other subtasks, check based on # of objects.
        # 对于其他任务 (Get, Chop, Merge)
        else:
            # Current count of desired objects.
            self.cur_obj_count = len(env.world.get_all_object_locs(obj=self.goal_obj))
            # Goal state is reached when the number of desired objects has increased.
            self.is_subtask_complete = lambda w: len(w.get_all_object_locs(obj=self.goal_obj)) > self.cur_obj_count


class SimAgent:
    """Simulation agent used in the environment object."""

    def __init__(self, name, id_color, location):
        self.name = name
        self.color = id_color
        self.location = location
        self.holding = None
        self.action = (0, 0)
        self.has_delivered = False

    def __str__(self):
        return color(self.name[-1], self.color)

    def __copy__(self):
        a = SimAgent(name=self.name, id_color=self.color,
                location=self.location)
        a.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            a.holding = copy.copy(self.holding)
        return a

    def get_repr(self):
        return AgentRepr(name=self.name, location=self.location, holding=self.get_holding())

    def get_holding(self):
        if self.holding is None:
            return 'None'
        return self.holding.full_name

    def print_status(self):
        print("{} currently at {}, action {}, holding {}".format(
                color(self.name, self.color),
                self.location,
                self.action,
                self.get_holding()))

    def acquire(self, obj):
        if self.holding is None:
            self.holding = obj
            self.holding.is_held = True
            self.holding.location = self.location
        else:
            self.holding.merge(obj) # Obj(1) + Obj(2) => Obj(1+2)

    def release(self):
        self.holding.is_held = False
        self.holding = None

    def move_to(self, new_location):
        self.location = new_location
        if self.holding is not None:
            self.holding.location = new_location
