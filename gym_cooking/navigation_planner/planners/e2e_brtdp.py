# Recipe planning
from recipe_planner.utils import *

# Navigation planning
import navigation_planner.utils as nav_utils
from navigation_planner.utils import MinPriorityQueue as mpq

# Other core modules
from utils.world import World
from utils.interact import interact
from utils.core import *

from collections import defaultdict
import numpy as np
import scipy as sp
import random
from itertools import product
import copy
import time
from functools import lru_cache
from enum import Enum

class PlannerLevel(Enum):
    LEVEL1 = 1
    LEVEL0 = 0

def argmin(vector):
    e_x = np.array(vector) == min(vector)   # 如果只有一个最小值，直接返回其索引
    # 如果有多个最小值，构建一个均匀分布的概率数组,使用多项分布随机选择一个最小值的索引
    return np.where(np.random.multinomial(1, e_x / e_x.sum()))[0][0]

def argmax(vector):
    # 如果只有一个最大值，直接返回其索引
    e_x = np.array(vector) == max(vector)
     # 如果有多个最大值，构建一个均匀分布的概率数组,使用多项分布随机选择一个最大值的索引
    return np.where(np.random.multinomial(1, e_x / e_x.sum()))[0][0]

class E2E_BRTDP:
    """Bounded Real Time Dynamic Programming (BRTDP) algorithm.

    For more details on this algorithm, please refer to the original
    paper: http://www.cs.cmu.edu/~ggordon/mcmahan-likhachev-gordon.brtdp.pdf
    """

    def __init__(self, alpha, tau, cap, main_cap):
        """
         初始化 BRTDP 算法及其超参数。

        Args:
            alpha (float): BRTDP 收敛标准。当值函数上界和下界之差小于 alpha 时，认为已收敛。 
            tau (float): BRTDP 归一化常数，用于 run SampleTrial 中的终止条件检查。 
            cap (int): BRTDP 在单次 run SampleTrial 探索中的最大步数（ rollout cap）。防止无限循环或过长探索。 
            main_cap (int): BRTDP 主循环 (Main) 的最大迭代次数。
            
        Initializes BRTDP algorithm with its hyper-parameters.
        Rf. BRTDP paper for how these hyper-parameters are used in their
        algorithm.

        http://www.cs.cmu.edu/~ggordon/mcmahan-likhachev-gordon.brtdp.pdf

        Args:
            alpha: BRTDP convergence criteria.
            tau: BRTDP normalization constant.
            cap: BRTDP cap on sample trial rollouts.
            main_cap: BRTDP Main cap on its Main loop.
        """
        self.alpha = alpha
        self.tau = tau
        self.cap = cap
        self.main_cap = main_cap

        # --- 状态值函数 ---
        # v_l: 存储状态值函数的下界 V_L(s, subtask) 的字典
        # v_u: 存储状态值函数的上界 V_U(s, subtask) 的字典
        self.v_l = {}
        self.v_u = {}
        
        # --- 状态表示与环境映射 ---
        # repr_to_env_dict: 将状态的字符串/元组表示映射回对应的 OvercookedEnvironment 对象副本。
        # 这对于在规划过程中恢复完整的环境状态信息很有用。
        self.repr_to_env_dict = dict()
        self.start = None
        self.pq = mpq()
        self.actions = World.NAV_ACTIONS
        self.is_joint = False  # 标记当前是否在为多个智能体联合规划 (True) 或为单个智能体规划 (False)
        self.planner_level = PlannerLevel.LEVEL0  # Level 0 规划时，临时移除的其他智能体持有的对象
        self.cur_object_count = 0   # 用于跟踪目标对象数量，以判断子任务是否完成
        self.is_subtask_complete = lambda h: False  # 一个函数，用于检查给定世界状态 w 是否完成了当前子任务
        self.removed_object = None # Level 0 规划时，临时移除的其他智能体持有的对象
        self.goal_obj = None # 当前子任务的目标对象

        # Setting up costs for value function.
        self.time_cost = 1.0    # 对应论文中的时间惩罚 γ
        self.action_cost = 0.1   # 对应论文中的移动惩罚 ϵ 

    def __copy__(self):
        copy_ = E2E_BRTDP(
                    alpha=self.alpha, tau=self.tau,
                    cap=self.cap, main_cap=self.main_cap)
        copy_.__dict__ = self.__dict__.copy()
        return copy_

    @lru_cache(maxsize=10000)
    def T(self, state_repr, action):
        """Return next states when taking action from state.
        状态转移函数 T(s, a)。
        给定当前状态的表示 (state_repr) 和一个动作 (action)，
        模拟执行该动作后的下一个状态。

        Args:
            state_repr (tuple): 当前状态的元组表示 (来自 env.get_repr())。
            action (tuple or tuple of tuples):
                - 单智能体规划时: 一个表示动作的元组，如 (0, 1)。
                - 联合规划时: 一个包含两个智能体动作的元组，如 ((0, 1), (-1, 0))。

        Returns:
            OvercookedEnvironment: 执行动作后的下一个环境状态对象。
       
        """
        state = self.repr_to_env_dict[state_repr]
        subtask_agents = self.get_subtask_agents(env_state=state)

        # Single agent    单智能体规划 
        if not self.is_joint:
            agent = subtask_agents[0]
            sim_state = copy.copy(state)
            sim_agent = list(filter(lambda a: a.name == agent.name, sim_state.sim_agents))[0]
            sim_agent.action = action
            interact(agent=sim_agent,
                     world=sim_state.world)    # 调用 interact 函数模拟动作效果，更新 sim_state

        # Joint
        else:
            agent_1, agent_2 = subtask_agents
            sim_state = copy.copy(state)
            sim_agent_1 = list(filter(lambda a: a.name == agent_1.name, sim_state.sim_agents))[0]
            sim_agent_2 = list(filter(lambda a: a.name == agent_2.name, sim_state.sim_agents))[0]
            sim_agent_1.action, sim_agent_2.action = action
            interact(agent=sim_agent_1, world=sim_state.world)
            interact(agent=sim_agent_2, world=sim_state.world)
            
            # 断言：确保联合动作后两个智能体不在同一位置 (interact 应该处理了碰撞)
            assert sim_agent_1.location != sim_agent_2.location, 'action {} led to state {}'.format(action, sim_state.get_repr())

        # Track this state in value function and repr dict
        # if it's a new state.
        self.repr_init(env_state=sim_state)
        self.value_init(env_state=sim_state)
        return sim_state

    def get_actions(self, state_repr):
        """Returns list of possible actions from current state. 
        获取在给定状态下所有可能的有效动作。

        Args:
            state_repr (tuple): 当前状态的元组表示。

        Returns:
            list:
                - 单智能体规划: 返回一个包含单个动作元组的列表，如 [(0, 1), (-1, 0), (0, 0)]。
                - 联合规划: 返回一个包含联合动作元组的列表，如 [((0,1), (0,0)), ((1,0), (-1,0))]。
                  只包含不会导致碰撞的联合动作。 
        """
        
         # 如果当前没有分配子任务 (self.subtask is None)，智能体只能执行不动 ((0, 0))
        if self.subtask is None:
            return [(0, 0)]
        # Convert repr into an environment object. 从表示缓存中获取完整的状态对象
        state = self.repr_to_env_dict[state_repr]

         # 获取当前子任务涉及的智能体对象列表
        subtask_agents = self.get_subtask_agents(env_state=state)
        output_actions = []

        # Return single-agent actions.   单智能体动作 
        if not self.is_joint:
            agent = subtask_agents[0]
            
            # 调用 nav_utils.get_single_actions 获取该智能体所有可能的单步动作
            # (包括移动和与环境交互的可能性，以及原地不动)
            output_actions = nav_utils.get_single_actions(env=state, agent=agent)
        # Return joint-agent actions.   联合动作  
        else:
            agent_1, agent_2 = subtask_agents
            
             # 使用 itertools.product 生成两个智能体所有单步动作的笛卡尔积 ，这构成了所有可能的联合动作组合
            valid_actions = list(product(
                nav_utils.get_single_actions(env=state, agent=agent_1),
                nav_utils.get_single_actions(env=state, agent=agent_2)))
            # Only consider action to be valid if agents do not collide.
            for va in valid_actions:
                agent1, agent2 = va
                
                 # 调用环境的 is_collision 方法检查这对动作是否会导致碰撞
                execute = state.is_collision(
                        agent1_loc=agent_1.location,
                        agent2_loc=agent_2.location,
                        agent1_action=agent1,
                        agent2_action=agent2)
                
                # 只有当两个动作都不会导致碰撞时，才将该联合动作视为有效动作
                if all(execute):
                    output_actions.append(va)
                    
           # 返回有效的动作列表
        return output_actions









    def runSampleTrial(self):
        """
        get next_action 方法会在决定最终动作 之前，检查是否需要调用 Main（进而调用 runSampleTrial）来进行更多的探索和值更新。
        触发条件是 B > diff，即预测的下一步状态的不确定性 (B) 相对于起始状态的不确定性 (diff) 来说过高。   
        
        run SampleTrial 执行一次从起始状态开始的模拟轨迹探索，并在过程中和结束时更新状态值函数的上下界 (V_l , V_u)
        
        前向探索 (Forward Pass / Simulation):
            模拟智能体从当前规划起点 (self.start) 出发，遵循当前的“最优”策略（基于V_l）进行一系列决策，生成一条状态-动作轨迹。
            循环执行，每次选择一个动作 a，这个动作是基于当前状态 (modified_state) 的 Q_l 值最小的那个动作 (argmin Q_L)。
            这代表了根据当前最乐观的估计（最低成本）应该采取的行动。
            记录下经过的每一个状态 x 到 traj 栈中。
            发现新状态: 如果探索中遇到之前未访问过的状态，会调用 repr_init 和 value_init 来初始化其表示和价值边界.
        
        在线价值更新:
            更新上界 (V U ​ ): 对于当前状态 modified_state，计算所有可选动作 a 的 Q U ​ (modified_state,a) 值，
            并取其中的最小值作为新的 V U ​ (modified_state)。
            更新下界 (V L ​ ): 对于当前状态 modified_state，计算根据 V L ​ 选择出的那个最优动作 a 的 Q L ​ (modified_state,a) 值，
            并将这个值直接赋给 V L ​ (modified_state)。 
        
        智能终止探索:
            计算根据当前策略选择的动作 a 会导向的下一个状态的不确定性 B=V U ​ (s ′ ) − V L ​ (s ′ )
            计算整个规划过程起始点 (self.start) 的归一化不确定性 diff=(V U ​ (start)−V L ​ (start))/τ
            判断: 如果 B≤diff，意味着按照当前最优动作走一步后，到达的状态其价值估计的区间宽度（不确定性 B）
            相对于整个问题的初始不确定性（diff）来说已经足够小了。
            此时，继续沿着这条路径深入探索对于改善起始点的决策可能帮助不大，因此 提前终止 当前的 run SampleTrial
        
        反向更新：
            将探索过程中获得的（可能更精确的）价值信息从轨迹的末端反向传播回路径上的所有状态。这是价值迭代的关键步骤
            对于弹出的每个状态 x，重新计算其 V U ​ (x) 和 V L ​ (x)。计算方式同第 2 步中的在线更新：
            V U ​ (x)=min a ​ Q U ​ (x,a) 和 V L ​ (x)=min a ​ Q L ​ (x,a) 
            （注意这里是对所有动作取 minQ L ​ 来更新 V L ​ ，这与前向探索中只用选定动作的 Q L ​ 不同，这更像是标准的价值迭代回溯）。
        """
        start_time = time.time()
        x = self.start
        
        # 初始化一个栈，用于存储探索过程中的状态轨迹，方便后续回溯更新
        traj = nav_utils.Stack()

        # Terminating if this takes too long e.g. path is infeasible.
        counter = 0
        start_repr = self.start.get_repr()
        
         # 计算起始状态值函数上下界之差，用于后续的终止条件判断
        diff = self.v_u[(start_repr, self.subtask)] - self.v_l[(start_repr, self.subtask)]
        print("DIFF AT START: {}".format(diff))

        while True:
            counter += 1
            if counter > self.cap: # 如果探索步数超过上限，则强制跳出循环
                break
            
             # 将当前状态压入轨迹栈
            traj.push(x)

             # 获取当前环境状态 x 的表示
            x_repr = x.get_repr()

            ''' # --- Level 1 规划处理 ---
            # 获取用于规划的状态。
            # 如果是 Level 1 规划器，modified_state 会包含其他智能体最可能的动作的影响。
            # 否则 (Level 0)，modified_state 与当前状态 x 相同。
            # _get_ modified_state_with_other_agent_actions 会返回修改后的状态和预测的其他智能体动作。 
            # Get the planner state. If Planner Level is 1, then
            # modified_state will include the most likely actions of the
            # other agents. Otherwise, the modified_state will be the same
            # as state `x`.'''
            modified_state, other_agent_actions = self._get_modified_state_with_other_agent_actions(x)
            modified_state_repr = modified_state.get_repr()

            # Get available actions from this state.
            # 获取在 (可能修改后的) 状态下的所有可用动作
            actions = self.get_actions(state_repr=modified_state_repr)

            ''' # --- 值函数更新 (基于修改后的状态) ---
            # 更新上界 V_U(s): V_U(s) = min_a Q_U(s, a)
            # 计算所有可能动作 a 的 Q 上界值 (使用 v_u)，并取最小值作为新的上界
            # We pick actions based on expected state.'''
            new_upper = min([
                self.Q(state=modified_state, action=a, value_f=self.v_u)
                for a in actions])
            
            '''# 更新当前 (修改后) 状态的上界值'''
            self.v_u[(modified_state_repr, self.subtask)] = new_upper

            '''# 选择动作 (基于下界): 根据 V_L(s) 选择贪婪动作 a* = argmin_a Q_L(s, a) 
            # 计算所有动作 a 的 Q 下界值 (使用 v_l)'''
            action_index = argmin([
                self.Q(state=modified_state, action=a, value_f=self.v_l)
                for a in actions])
            a = actions[action_index]

            '''# 更新下界 V_L(s): V_L(s) = Q_L(s, a*) # 计算选中动作 a 的 Q 下界值 '''
            new_lower = self.Q(state=modified_state, action=a, value_f=self.v_l)
            self.v_l[(modified_state_repr, self.subtask)] = new_lower

            '''# --- 检查终止条件 ---
            # 计算下一个状态 s' 的预期值函数差 B = E[V_U(s') - V_L(s')]
            # 在确定性环境中，这简化为 B = V_U(T(s,a)) - V_L(T(s,a))'''
            b = self.get_expected_diff(modified_state, a)
            B = sum(b.values())
            diff = (self.v_u[(start_repr, self.subtask)] - self.v_l[(start_repr, self.subtask)])/self.tau
            if (B <= diff):
                break

            '''# --- 状态转移 ---
            # 获取执行动作 a 后的下一个状态
            # 因为环境是确定性的，get_expected_diff 返回的字典 b 只包含一个键 (下一个状态的表示)'''
            x = self.repr_to_env_dict[list(b.keys())[0]]

            ''' # --- 新状态初始化 ---
            # 如果下一个状态 x 是新遇到的，初始化其表示和值函数界限
            # Track this new state in repr dict and value function if it's new.'''
            self.repr_init(env_state=x)
            self.value_init(env_state=x)

        print("RUN SAMPLE EXPLORED {} STATES, took {}".format(len(traj), time.time()-start_time))
        
        ''' # --- 回溯更新阶段 ---
        # 当前向探索结束后，从轨迹栈中弹出状态，并对路径上的所有状态执行 Bellman 更新'''
        while not(traj.empty()):
            x = traj.pop()
            x_repr = x.get_repr()
            actions = self.get_actions(state_repr=x_repr)
            self.v_u[(x_repr, self.subtask)] = min([
                self.Q(state=x, action=a, value_f=self.v_u) for a in actions])
            self.v_l[(x_repr, self.subtask)] = min([
                self.Q(state=x, action=a, value_f=self.v_l) for a in actions])




    def main(self):
        """Main loop function for BRTDP.
        main 函数的主要目的是 迭代地优化和收紧当前规划问题 起始状态 (self.start) 的价值估计
        （具体来说是价值函数的上界 V U ​ 和下界 V L ​ ），直到这两个界限足够接近（收敛）
        """
        main_counter = 0
        start_repr = self.start.get_repr()

         # 获取起始状态的初始上界和下界值
        upper = self.v_u[(start_repr, self.subtask)]
        lower = self.v_l[(start_repr, self.subtask)]
        diff = upper - lower

        '''# --- 主循环 ---
        # 持续运行，直到满足以下任一条件：
        # 1. 起始状态的上下界之差 diff 小于或等于收敛阈值 self.alpha。
        # 2. 主循环迭代次数 mai counter 达到上限 self.main_cap。
        Run until convergence or until you max out on iteration  '''  
        while (diff > self.alpha) and (main_counter < self.main_cap):
            print('\nstarting  Main loop #', main_counter)
            new_upper = self.v_u[(start_repr, self.subtask)]
            new_lower = self.v_l[(start_repr, self.subtask)]
            new_diff = new_upper - new_lower
            if new_diff > diff + 0.01:
                self.start.update_display()
                self.start.display()
                self.start.print_agents()
                print('old: upper {}, lower {}'.format(upper, lower))
                print('new: upper {}, lower {}'.format(new_upper, new_lower))
            diff = new_diff
            upper = new_upper
            lower = new_lower
            main_counter +=1
            print('diff = {}, self.alpha = {}'.format(diff, self.alpha))
            self.runSampleTrial()







    def _configure_planner_level(self, env, subtask_agent_names, other_agent_planners):
        """
         如果 other agent_planners 是一个空字典，则此规划器应为 Level 0 规划器，
        并在 env 的本地副本中移除所有不相关的智能体 (将它们视为静态障碍)。

        否则 (other agent_planners 非空)，它应该是一个 Level 1 规划器，
        保留所有智能体，并维护这些智能体的规划器 (这些规划器已经被配置为我们认为它们正在执行的子任务)。
        
        Configure the planner s.t. it best responds to other agents as needed.

        If other agent_planners is an emtpy dict, then this planner should
        be a level-0 planner and remove all irrelevant agents in env.

        Otherwise, it should keep all agents and maintain their planners
        which have already been configured to the subtasks we believe them to
        have."""
        
        # --- Level 1 规划器 ---
        # 如果提供了其他智能体的规划器字典 (非空) 
        if other_agent_planners:
            self.planner_level = PlannerLevel.LEVEL1 ################## 在这里设置了 level 1 
            self.other_agent_planners = other_agent_planners
        # Level 0 Planner.
        else:
            self.planner_level = PlannerLevel.LEVEL0
            self.other_agent_planners = {}
            # Replace other agents with counters (frozen agents during planning).
            # --- 在环境副本中移除不相关的智能体 ---
            # 目的是在 Level 0 规划时，将其他智能体视为固定的障碍物。
            rm_agents = []
            for agent in env.sim_agents:
                if agent.name not in subtask_agent_names:
                    rm_agents.append(agent)
            for agent in rm_agents:
                env.sim_agents.remove(agent)
                
                 # 如果被移除的智能体正持有物品
                if agent.holding is not None:
                    self.removed_object = agent.holding
                    env.world.remove(agent.holding)

                # Remove Floor and replace with Counter. This is needed when
                # checking whether object @ location is collidable.
                 # 为了在碰撞检测时将其他智能体视为障碍物，需要将其位置上的 Floor 替换为 Counter。
                # 在相同位置插入一个特殊的 AgentCounter 格子 (collidable=True)
                env.world.remove(Floor(agent.location))
                env.world.insert(AgentCounter(agent.location))







    def _configure_subtask_information(self, subtask, subtask_agent_names):
        """ 
        存储当前规划器正在处理的子任务分配的相关信息。

        Args:
            subtask (Action or None): 当前要规划的子任务对象 (例如 Chop('Tomato'))。
            subtask_agent_names (tuple): 分配执行此子任务的智能体名称元组。 
        Tracking information about subtask allocation."""
        # Subtask allocation
        self.subtask = subtask
        self.subtask_agent_names = subtask_agent_names

        # Relevant objects for subtask allocation.
        self.start_obj, self.goal_obj = nav_utils.get_subtask_obj(subtask)
        self.subtask_action_obj = nav_utils.get_subtask_action_obj(subtask)






    def _define_goal_state(self, env, subtask):
        """
        为当前正在规划的子任务(subtask)定义目标状态(goal state)的判断条件。
        这个函数主要设置两个 lambda 函数：
        1. self.is_goal_state(h): 用于 BRTDP 规划过程中判断一个状态表示 h 是否达到了目标。
                                  规划会在达到目标状态时停止探索或返回0值。
                                  它比较的是 *规划过程中* 遇到的状态与 *开始规划时* 的状态。
        2. self.is_subtask_complete(w): 用于在 Agent 层面判断当前子任务是否真的完成了。
                                     它比较的是 *当前真实世界状态* w 与 *开始执行该子任务时* 的状态。
                                     这个结果会影响 Agent 是否更新其未完成任务列表以及是否重置信念。

        Args:
            env (OvercookedEnvironment): 调用此方法时的环境对象，用于获取初始状态信息 (例如物品数量)。
            subtask (Action or None): 当前规划器正在处理的子任务对象 (e.g., Chop('Tomato'), Deliver('Plate-Tomato'))。
                                     或者是 None，表示当前没有分配任务。
      
      
        区别特征	elif isinstance(subtask, Deliver):	         else: (Get, Chop, Merge)
        目标性质	物品到达 特定位置 (递送点)	              目标物品的 存在/创建 (任意位置)
        计数对象	在递送点上的 未持有 目标物品	          世界上 所有 (持有+未持有) 目标物品
        完成判断	递送点上未持有物品数量 增加	              世界上总物品数量 增加

                             
        Defining a goal state (termination condition on state) for subtask 
        """   


        ''' # --- 情况 1: 子任务是 None ---
        # 如果当前没有分配子任务 (智能体空闲)，那么任何状态都可以被认为是“目标”状态，
        # 因为没有特定的任务需要完成。规划会立即停止。'''
        if subtask is None:
            self.is_goal_state = lambda h: True

        #  --- 情况 2: 子任务是 Deliver (递送) ---
        # 递送任务的完成条件是：目标物品 (例如 'Plate-Tomato') 出现在递送点 (*) 上。
        # Termination condition is when desired object is at a Deliver location.
        elif isinstance(subtask, Deliver):
            '''  # --- 计算初始状态 ---
            # self.subtask_action_obj: 在 set settings 中已设置，代表 Delivery 格子对象。
            # self.goal_obj: 在  set settings 中已设置，代表需要递送的物品对象 (e.g., Object('Plate-Tomato'))。

            # 计算在 *开始规划时*，有多少个 *未被持有* 的目标物品 (self.goal_obj) 位于递送点 (self.subtask_action_obj) 的位置上。
            # env.world.get_all_object_locs(self.subtask_action_obj): 获取所有递送点的位置列表。
            # env.world.get_object_locs(self.goal_obj, is_held=False): 获取所有未被持有的目标物品的位置列表。
            # filter(...): 筛选出那些位置同时是递送点位置的目标物品。
            # len(...): 计算满足条件的目标物品数量，存入 self.cur_obj_count。
            # **例子**: 开始规划 Deliver('Plate-Tomato') 时，递送点上没有盘子番茄，self.cur_obj_count = 0。'''
            self.cur_obj_count = len(
                    list(filter(lambda o: o in set(env.world.get_all_object_locs(
                            self.subtask_action_obj)),
                    env.world.get_object_locs(self.goal_obj, is_held=False))))
            
            '''    定义一个辅助 lambda 函数，用于检查输入的数量 x 是否大于初始计数值 self.cur_obj_count。''' 
            self.has_more_obj = lambda x: int(x) > self.cur_obj_count
            
            ''' # --- 定义 is_goal_state (用于规划) ---
            # 这个 lambda 函数接收一个状态表示 h (通常是 tuple 类型)。
            # 它首先通过 self.repr_to_env_dict[h] 从状态表示恢复到完整的环境对象。
            # 然后计算在这个恢复的环境状态中，有多少个未被持有的目标物品位于递送点。
            # 最后用 self.has_more_obj 检查这个数量是否 *大于* 初始计数值。
            # **例子**: 规划过程中，如果某个状态 h 对应的环境里，递送点上出现了 1 个 'Plate-Tomato'，
            #         而初始计数是 0，那么 len(...) > self.cur_obj_count (即 1 > 0) 为 True，
            #         这个状态 h 就被视为目标状态，规划到此结束或返回 0 值。 '''
            self.is_goal_state = lambda h: self.has_more_obj(
                    len(list(filter(lambda o: o in set(env.world.get_all_object_locs(self.subtask_action_obj)),
                    self.repr_to_env_dict[h].world.get_object_locs(self.goal_obj, is_held=False)))))

            ''' # --- 定义 is_subtask_complete (用于 Agent 判断任务完成) ---
            # 这个 lambda 函数接收一个真实世界状态 w (OvercookedEnvironment 对象)。
            # 它直接在真实世界 w 上计算有多少个未被持有的目标物品位于递送点。

            # 特殊情况处理: 检查在 Level 0 规划时是否临时移除了一个 *与目标物品相同* 的对象。
            # self.removed_object: 在 _configure_planner_level 中设置，如果 Level 0 规划移除了其他智能体持有的物品。
            # 如果移除了目标物品，那么在检查完成时，需要将这个被移除的物品也算进去 (+1)。
            # 这是因为 Level 0 规划时看不到这个物品，但在真实世界中它可能被放回并递送。
            # **例子**: 规划 Deliver('Plate-Tomato') 时，另一个智能体 agent-2 正好拿着 'Plate-Tomato' (被移除，self.removed_object == self.goal_obj)。
            #         当 agent-1 检查任务是否完成时，如果递送点上有 0 个 'Plate-Tomato'，但加上被移除的 1 个，总数是 1。
            #         如果初始计数是 0，那么 1 > 0，任务完成。'''
            if self.removed_object is not None and self.removed_object == self.goal_obj:
                self.is_subtask_complete = lambda w: self.has_more_obj(
                        len(list(filter(lambda o: o in set(env.world.get_all_object_locs(self.subtask_action_obj)),
                        w.get_object_locs(self.goal_obj, is_held=False)))) + 1)
            else:
                ''' # 正常情况：直接计算真实世界 w 中递送点上的目标物品数量，并用 has_more_obj 比较。
                # **例子**: 真实世界 w 中递送点上出现了 1 个 'Plate-Tomato'，初始计数是 0，
                #         len(...) > self.cur_obj_count (1 > 0) 为 True，任务完成'''
                self.is_subtask_complete = lambda w: self.has_more_obj(
                        len(list(filter(lambda o: o in set(env.world.get_all_object_locs(self.subtask_action_obj)),
                        w.get_object_locs(obj=self.goal_obj, is_held=False)))))
        else:
            ''' # --- 情况 3: 其他子任务 (Get, Chop, Merge) ---
                这些任务的完成条件通常是：期望的目标物品 (self.goal_obj) 在世界上的总数增加了。
                Get current count of desired objects.'''
                
                
                
            ''' --- 计算初始状态 ---
            # self.goal_obj: 在 set settings 中已设置，代表任务完成时期望得到的物品对象
            #               (e.g., 对于 Chop('Tomato')，是 Object('ChoppedTomato');
            #                对于 Merge('Tomato', 'Plate')，是 Object('Plate-Tomato'))。
            # 计算在 *开始规划时*，世界上目标物品的总数 (无论是否被持有)。
            # **例子**: 开始规划 Chop('Tomato') 时，世界上没有切好的番茄，self.cur_obj_count = 0。
            # **例子**: 开始规划 Merge('Tomato', 'Plate') 时，假设 agent-1 拿着 ChoppedTomato，agent-2 拿着 Plate，
            #         世界上没有 Plate-Tomato，self.cur_obj_count = 0。    ''' 
            self.cur_obj_count = len(env.world.get_all_object_locs(self.goal_obj))
            # Goal state is reached when the number of desired objects has increased.
            self.has_more_obj = lambda x: int(x) > self.cur_obj_count
           
            ''' --- 定义 is_goal_state (用于规划) ---
            # 这个 lambda 函数接收状态表示 h。 repr_to_env_dict[h]
            # 用 self.has_more_obj 检查数量是否 *大于* 初始计数值。
            # **例子**: 规划 Chop('Tomato') 时，某个状态 h 对应的环境里出现了 1 个 ChoppedTomato，
            #         初始计数是 0，len(...) > self.cur_obj_count (1 > 0) 为 True，达到目标状态。   '''
            self.is_goal_state = lambda h: self.has_more_obj(
                    len(self.repr_to_env_dict[h].world.get_all_object_locs(self.goal_obj)))
            
            
            '''  --- 定义 is_subtask_complete (用于 Agent 判断任务完成) ---
            # 这个 lambda 函数接收真实世界状态 w。
            # 特殊情况处理 (同 Deliver): 如果 Level 0 规划时移除了目标物品，检查完成时要加回来。
            # **例子**: 规划 Chop('Tomato') 时，agent-2 拿着 ChoppedTomato (被移除)。
            #         当 agent-1 检查任务是否完成时，如果世界上自己刚切好 1 个，加上被移除的 1 个，总数是 2。
            #         如果初始计数是 0，那么 2 > 0，任务完成。'''
            if self.removed_object is not None and self.removed_object == self.goal_obj:
                self.is_subtask_complete = lambda w: self.has_more_obj(
                        len(w.get_all_object_locs(self.goal_obj)) + 1)
            else:
                '''   正常情况：直接计算真实世界 w 中目标物品的总数，并用 has_more_obj 比较。
                # **例子**: 真实世界 w 中出现了 1 个 ChoppedTomato，初始计数是 0，
                #         len(...) > self.cur_obj_count (1 > 0) 为 True，任务完成。'''
                self.is_subtask_complete = lambda w: self.has_more_obj(
                        len(w.get_all_object_locs(self.goal_obj)))






    def _configure_planner_space(self, subtask_agent_names):
        """Configure planner to either plan in joint space or single-agent space."""
        assert len(subtask_agent_names) <= 2, "Cannot have more than 2 agents! Hm... {}".format(subtask_agents)

        self.is_joint = len(subtask_agent_names) == 2




    def set_settings(self, env, subtask, subtask_agent_names, other_agent_planners={}):
        """  
        配置导航规划器 (E2E_BRTDP 实例) 以准备为特定的子任务分配进行规划。
        这是在每次调用 get_ next_action 之前必须执行的核心设置函数。

        Args:
            env (OvercookedEnvironment): 当前的环境状态对象。
            subtask (Action or None): 要规划的子任务。
            subtask_agent_names (tuple): 分配执行此子任务的智能体名称元组。
            other agent_planners (dict, optional):
                一个字典，映射其他智能体名称到他们的规划器实例。
                如果提供 (非空)，则启用 Level 1 规划，考虑其他智能体的预测行为。
                如果为空 (默认)，则执行 Level 0 规划，将其他智能体视为静态。
                默认为 {}。 
        Configure planner."""
        
        
        
        ''' # 1. 配置规划器级别 (Level 0 或 Level 1)
        #    调用内部方法 _configure_ planner_level。
        #    - 如果 other agent_planners 非空，则设置为 Level 1，并存储其他智能体的规划器。
        #    - 如果 other agent_planners 为空，则设置为 Level 0，并在 env 的副本中移除其他智能体
        #      (将其位置替换为 AgentCounter 障碍物)，可能记录被移除的持有物到 self.removed_object。
        #      Configuring the planner level.'''
        self._configure_planner_level(
                env=env,
                subtask_agent_names=subtask_agent_names,
                other_agent_planners=other_agent_planners)


        ''' # 2. 配置子任务相关信息
        #    调用内部方法 _configure_ subtask_information。
        #    - 存储 self.subtask 和 self.subtask_agent_names。
        #    - 调用 nav_utils 中的帮助函数，获取并存储子任务的起始物品(self.start_obj)、
        #      目标物品(self.goal_obj) 和静态交互对象(self.subtask_action_obj)。
        #    Configuring subtask related information. '''
        self._configure_subtask_information(
                subtask=subtask,
                subtask_agent_names=subtask_agent_names)


        ''' # 3. 定义当前子任务的目标状态判断条件
        #    调用内部方法 _define_ goal_state。
        #    - 根据子任务类型 (Deliver 或其他)，设置 self.cur_obj_count (初始目标物品计数)。
        #    - 定义 lambda 函数 self.is_goal_state(h) 用于规划中的目标判断。
        #    - 定义 lambda 函数 self.is_subtask_complete(w) 用于 Agent 判断任务是否完成。
            Defining what the goal is for this planner.'''
        self._define_goal_state(
                env=env,
                subtask=subtask)

        ''' # 4. 定义规划空间 (单智能体或联合)
        #    调用内部方法 _configure _planner_space。
        #    - 根据 subtask_agent_names 的长度设置 self.is_joint 标志 (True 或 False)。
           Defining the space of the planner (joint or single).'''  
        self._configure_planner_space(subtask_agent_names=subtask_agent_names)

        # Make sure termination counter has been reset.
        self.counter = 0
        self.num_explorations = 0
        self.stop = False
        self.num_explorations = 0

        # Set start state.
        self.start = copy.copy(env)
        
        #   初始化起始状态的表示形式并存入 self.repr_to_env_dict。
        self.repr_init(env_state=env)
        
        #   初始化起始状态的值函数上下界 (V_L, V_U)。
        self.value_init(env_state=env)




    def get_subtask_agents(self, env_state):
        """Return subtask agent for this planner given state."""
        subtask_agents = list(filter(lambda a: a.name in self.subtask_agent_names, env_state.sim_agents))

        assert list(map(lambda a: a.name, subtask_agents)) == list(self.subtask_agent_names), "subtask agent names are not in order: {} != {}".format(list(map(lambda a: a.name, subtask_agents)), self.subtask_agent_names)

        return subtask_agents




    def repr_init(self, env_state):
        """Initialize repr for environment state."""
        es_repr = env_state.get_repr()
        if es_repr not in self.repr_to_env_dict:
            self.repr_to_env_dict[es_repr] = copy.copy(env_state)
        return es_repr




    def value_init(self, env_state):
        """Initialize value for environment state."""
        ''' # 如果两个字典中都已存在该键，说明这个状态-子任务对的值已经被初始化过了，直接返回，不做任何操作。
           Skip if already initialized. ''' 
        es_repr = env_state.get_repr()
        if ((es_repr, self.subtask) in self.v_l and
            (es_repr, self.subtask) in self.v_u):
            return

        '''# 如果是目标状态，根据 MDP 定义，其最优价值为 0。
            # 将该状态-子任务对的 V_L 和 V_U 都设置为 0.0。
            Goal state has value 0.'''
        if self.is_goal_state(es_repr):
            self.v_l[(es_repr, self.subtask)] = 0.0
            self.v_u[(es_repr, self.subtask)] = 0.0 
            return

        '''--- 计算启发式下界 (lower) ---
        # 调用环境状态对象的 get lower_bound_for_subtask_given_objs 方法。
        # 这个方法会根据当前子任务、执行智能体、起始/目标物品等信息，
        # 利用预计算的距离（或曼哈顿距离等启发式）估算完成该子任务所需的最短路径成本（距离下界）。[cite: 452]
        # 这个距离是完成任务的理论最小步数的一种估计。
          Determine lower bound on this environment state.'''
        lower = env_state.get_lower_bound_for_subtask_given_objs(
                subtask=self.subtask,
                subtask_agent_names=self.subtask_agent_names,
                start_obj=self.start_obj,
                goal_obj=self.goal_obj,
                subtask_action_obj=self.subtask_action_obj)

        subtask_agents = self.get_subtask_agents(env_state=env_state)
        
        '''# 将距离下界转换为成本下界。
        # 乘以 (self.time_cost + self.action_cost) 是为了将步数（距离）转换成累积成本。
        # self.time_cost: 每走一步的时间成本 (论文中的 γ) [cite: 441]。
        # self.action_cost: 执行移动动作的额外成本 (论文中的 ϵ) [cite: 441]。
        # 这个乘积代表了走一步（移动）的最小成本。
        # 假设完成任务最少需要 lower 步，那么成本至少是 lower * (最小单步成本)。  '''
        lower = lower * (self.time_cost + self.action_cost)

        # By BRTDP assumption, this should never be negative.
        assert lower > 0, "lower: {}, {}, {}".format(lower, env_state.display(), env_state.print_agents())

        '''  V_L (下界) 初始化：使用计算出的启发式成本下界 `lower_cost`，再减去一个小的正数 (1.09)。
        #   减去一个值是为了确保初始下界确实低于可能的最优值。这个具体的 1.09 可能是经验值或特定于实现的调整。
        #   论文 [cite: 455] 提到 V_L 可以用启发式 h(s) 初始化，只要 h(s) <= V*(s)。'''
        self.v_l[(es_repr, self.subtask)] = lower - 1.09  #    1.09 是什么鬼，经验值吧
                
        '''  V_U (上界) 初始化：使用启发式成本下界 `lower_cost` 乘以一个较大的因子 (例如 5) 和单步成本。
        #   这提供了一个相对宽松但必须保证大于等于 V*(s) 的初始上界。因子 5 也是经验性的。
        #   论文 [cite: 455] 提到 V_U 需要是可接受的 (admissible)，即 V_U(s) >= V*(s)。'''
        self.v_u[(es_repr, self.subtask)] = lower * 5 * (self.time_cost + self.action_cost)

            
            
            
            
            
            
            
            

    def Q(self, state, action, value_f):  #self.v_l 或 self.v_u)
        """
        计算给定状态(state)下执行动作(action)的 Q 值。
        这个 Q 值是基于特定的值函数估计 (value_f，可以是下界 v_l 或上界 v_u) 来计算的。
        它遵循标准的贝尔曼方程形式：Q(s, a) = Cost(s, a) + Discount * Expected_Value(s')
        由于环境是确定性的，预期值就是下一个状态的值。折扣因子在这里隐含为1.0（或者说成本包含了时间衰减）。
        参考论文 [cite: 440]。

        Args:
            state (OvercookedEnvironment): 当前的环境状态对象。
            action (tuple or tuple of tuples): 要评估的动作。
                                                单智能体: (dx, dy)
                                                联合: ((dx1, dy1), (dx2, dy2))
            value_f (dict): 要使用的值函数字典 (self.v_l 或 self.v_u)。
                            它将 (state_repr, subtask) 映射到相应的 V_L 或 V_U 值。

        Returns:
            float: 计算得到的 Q(state, action) 值。
            
            Get Q value using value_f of (state, action).
        """   
        # Q(s,a) = c(x,a) + \sum_{y \in S} P(x, a, y) * v(y)
        cost = self.cost(state, action)

        # Initialize state if it's new.
        s_repr = self.repr_init(env_state=state)
        self.value_init(env_state=state)

        '''获取下一个状态 (Next State)
        #    调用 self.T 方法模拟执行动作 action 后的下一个状态 next_state。
        #    self.T 内部处理了单智能体和联合动作的转移逻辑。
        #    Get next state.'''
        next_state = self.T(state_repr=s_repr, action=action)

        # Initialize new state if it's new.
        ns_repr = self.repr_init(env_state=next_state)
        self.value_init(env_state=next_state)

        ''''expected_value = value_f[(ns_repr, ...)]：查找 value_f 字典（比如 self.v_l）中，
        键为 (ns_repr, self.subtask) 对应的值。因为 ns_repr 代表的是“原地踏步”的状态，这个状态离最终目标还很远，
        所以它在字典中存储的预估未来成本（cost-to-go）值本身就比较高。
        这个值是之前通过 run SampleTrial 的回溯更新或者 value_init 的启发式得到的，不是因为这次模拟碰撞而实时变高的。'''
        expected_value = 1.0 * value_f[(ns_repr, self.subtask)]
        return float(cost + expected_value)




    def V(self, state, _type):
        """
         计算给定状态 (state) 的值函数 V(s)。
        根据贝尔曼最优性原理，V*(s) = min_a Q*(s, a)。
        在 BRTDP 中，我们分别计算值函数的下界 V_L(s) 和上界 V_U(s)。
        
        Get V*(x) = min_{a \in A} Q_{v*}(x, a)."""

        # Initialize state if it's new.
        s_repr = self.repr_init(env_state=state)

        # Check if this is the desired goal state.
        if self.is_goal_state(s_repr):
            return 0

        ''' 计算 V_L (下界)    V_L(s) = min_a Q_L(s, a) Use lower bound on value function. '''
        if _type == "lower":
            return min([
                self.Q(state=state, action=action, value_f=self.v_l)
                for action in self.get_actions(state_repr=s_repr)])
        
        elif _type == "upper":
            '''计算 V_U (上界)  V_U(s) = min_a Q_U(s, a)   Use upper bound on value function.'''
            return min([
                self.Q(state=state, action=action, value_f=self.v_u)
                for action in self.get_actions(state_repr=s_repr)])
        else:
            raise ValueError("Don't recognize the value state function type: {}".format(_type))




    def cost(self, state, action):
        """Return Cost of taking action in this state."""
        cost = self.time_cost
        if isinstance(action[0], int):
            action = tuple([action])
            
       #    遍历动作元组中的每个单独动作 a (对于联合动作，会有多个；对于单智能体，只有一个)。
        for a in action:
            if a != (0, 0):
                cost += self.action_cost
        return cost




    def get_expected_diff(self, start_state, action):
        """
        计算在状态 start_state 执行动作 action 后，到达的下一个状态 s_ 的值函数上界和下界之差 (V_U(s_) - V_L(s_))。
        在 BRTDP 算法中，这个差值 B 用于判断是否需要继续探索 (run SampleTrial)。 Get next state."""
        
        '''获取下一个状态 #    使用 self.T 方法计算执行动作 action 后的下一个状态 s_。'''
        s_ = self.T(state_repr=start_state.get_repr(), action=action)

        '''初始化下一个状态 (如果需要) #    确保下一个状态 s_ 的表示和值函数边界已初始化。
        # Initialize state if it's new.'''
        s_repr = self.repr_init(env_state=s_)
        self.value_init(env_state=s_)

        ''' 计算 V_U 和 V_L 的差值 # Get expected diff.'''
        b = {s_repr: 1.0 * (self.v_u[(s_repr, self.subtask)] - self.v_l[(s_repr, self.subtask)])}
        return b




    def _get_modified_state_with_other_agent_actions(self, state):
        """
        根据当前规划器的级别 (Level 0 或 Level 1)，可能地修改输入状态 state。
        这个方法实现了论文 [cite: 203, 204, 205] 中描述的 Level-k 规划思想（这里具体是 Level-1）。

        - Level 0 规划: 不考虑其他智能体，直接返回原始状态 state 的副本。
        - Level 1 规划: 预测其他智能体 (根据 self.other agent_planners) 最可能执行的动作，
                       并将这些动作应用到 state 的一个副本上，生成一个 "修改后" 的状态 modified_state。
                       这个 modified_state 反映了对其他智能体一步行为的预期。

        Args:
            state (OvercookedEnvironment): 当前需要处理的环境状态对象。

        Returns:
            tuple: (modified_state, other_agent_actions)
                   - modified_state (OvercookedEnvironment): 修改后的（或原始的）环境状态副本。
                   - other_agent_actions (dict): 一个字典，映射其他智能体名称到他们被预测采取的动作。
                                                 对于 Level 0，此字典为空。
        
        Do nothing if the planner level is level 0.
        Otherwise, using self.other agent_planners, anticipate what other agents will do and modify the state appropriately.
        Returns the modified state and the actions of other agents that triggered the change.     
       
    在这个三智能体例子中，agent-1 在规划自己去拿番茄的路径时，会使用那个预测了 agent-2 正在处理生菜、agent-3 
    正在处理盘子的 modified_state。这意味着 agent-1 的规划（Q值、V值计算）会基于一个假设：接下来的瞬间，agent-2 和 agent-3 
    都会执行他们各自任务的最优动作。这有助于 agent-1 避免与他们发生冲突，或者利用他们可能的移动来优化自己的路径。例如，如果 agent-3 拿盘子的动作是向某个方向移动，agent-1 就能预见到这一点。 
        """
        modified_state = copy.copy(state)
        other_agent_actions = {}

        ''' # 如果是 Level 0，不修改状态，直接返回原始状态的副本和空字典   Do nothing if the planner level is 0.  '''
        if self.planner_level == PlannerLevel.LEVEL0:
            return modified_state, other_agent_actions

        
        '''# 如果是 Level 1，需要预测并应用其他智能体的动作。 
        # Otherwise, modify the state because Level 1 planners consider the actions of other agents.'''
        for other_agent_name, other_planner in self.other_agent_planners.items():
            
            
            '''# --- 为其他智能体配置临时规划器 ---
            # 目标是预测 other_agent 在 *当前状态 state* 下最可能做什么。 
            # 重要的是，在预测其他智能体的行为时，我们假设他们是 Level 0 的规划者 
              Keep their recipe subtask & subtask agent fixed, but change  their planner state to `state`. 
              These new planners should be level 0 planners.'''
            other_planner.set_settings(env=copy.copy(state),
                                       subtask=other_planner.subtask,
                                       subtask_agent_names=other_planner.subtask_agent_names)
            ''' 对于我们自己，我们自己的agent 是 level 1 ， 考虑别的，把别人当成贪心的 level 0'''
            assert other_planner.planner_level == PlannerLevel.LEVEL0

            '''# --- 预测其他智能体的贪婪动作 ---'''
            possible_actions = other_planner.get_actions(state_repr=other_planner.start.get_repr())
             
            greedy_action = possible_actions[
                    argmin([other_planner.Q(state=other_planner.start,
                                            action=action,
                                            value_f=other_planner.v_l)
                    for action in possible_actions])]


            ''' # --- 处理联合动作情况 ---
            # 如果 other_planner 本身是在进行联合规划 (is_joint is True)，
            # 那么 greedy_action 会是一个包含两个智能体动作的元组。
            # 我们需要从中提取出属于 other_agent_name 的那个动作。'''
            if other_planner.is_joint:
                greedy_action = greedy_action[other_planner.subtask_agent_names.index(other_agent_name)]

            '''  # --- 记录并应用预测的动作 ---
            # 将预测出的 other_agent_name 的动作 greedy_action 存入字典   # Keep track of their actions.'''
            other_agent_actions[other_agent_name] = greedy_action
            
            
            ######################
            ######################
            '''在 modified_state 这个环境副本的智能体列表 (modified_state.sim_agents) 中，找到代表 other_agent_name 的那个 SimAgent 对象实例  我们将其称为 other_agent
            设置动作属性: other_agent.action = greedy_action 这行代码直接修改了上一步找到的 other_agent 对象的 action 属性，将其设置为刚刚预测出来的 greedy_action
            modified_state 的修改并不是改变了环境的布局 ，而是改变了包含在 modified_state 中的其他智能体（SimAgent 对象）的内部状态——具体来说，是设置了它们预期在下一步要执行的动作 (.action 属性)
            这样一来，Level-1 智能体在规划自己的路径时，就能“预见到”其他智能体可能会如何移动（根据 Level-0 的贪心预测），从而可以规划出更有效的、能够避开潜在冲突或利用他人移动的路径
            '''
            other_agent = list(filter(lambda a: a.name == other_agent_name,
                                      modified_state.sim_agents))[0]
            other_agent.action = greedy_action

        # Initialize state if it's new.
        self.repr_init(env_state=modified_state)
        self.value_init(env_state=modified_state)
        return modified_state, other_agent_actions




    def get_next_action(self, env, subtask, subtask_agent_names, other_agent_planners):
        """Return next action."""
        print("-------------[e2e]-----------")
        self.removed_object = None
        start_time = time.time()

        '''# === 步骤 1: 配置规划器 ===
        # 调用  set settings 方法，使用传入的参数全面配置规划器的内部状态。
        # 这包括：设置规划级别(Level 0/1)、存储子任务信息(目标物品、交互对象等)、
        # 定义子任务完成条件(is_goal_state, is_subtask_complete)、
        # 配置规划空间(单智能体/联合)、重置内部计数器、设置起始状态(self.start)，
        # 以及初始化起始状态的值函数边界(V_L, V_U)。
        # Configure planner settings.'''
        self.set_settings(
                env=env, subtask=subtask,
                subtask_agent_names=subtask_agent_names,
                other_agent_planners=other_agent_planners)

        '''# === 步骤 2: 获取规划起点状态 ===
        # 调用内部方法获取实际用于规划的状态 `cur_state`。
        # - Level 0: `cur_state` 是 `env` 的一个副本（可能移除了其他 agent）。
        # - Level 1: `cur_state` 是 `env` 的一个副本，但其中其他 agent 的 `action` 属性
        #            已被 `_get _modified_state_with_other_agent_actions` 预测并设置好。   #### 《=重点！！！！！！！！
        # `other_agent_actions` 存储了预测出的其他 agent 的动作（Level 1 时）或为空（Level 0 时）。
         Modify the state with other_agent_planners (Level 1 Planning).'''  
        cur_state, other_agent_actions = self._get_modified_state_with_other_agent_actions(state=self.start)

        ''' # === 步骤 3: BRTDP 核心逻辑 - 判断是否需要探索 ===
        # 这部分实现了 BRTDP 算法的一个关键决策点：是否需要运行一轮 SampleTrial 来更新值函数。
        获取当前规划状态 `cur_state` 下所有可能的动作 `actions`。
        # BRTDP Main loop.'''
        actions = self.get_actions(state_repr=cur_state.get_repr())
        action_index = argmin([
            self.Q(state=cur_state, action=a, value_f=self.v_l)
            for a in actions])
        a = actions[action_index]
        
        '''计算执行这个贪婪动作 `a` 后，到达的下一个状态的不确定性 `B`。 
        B = V_U(T(cur_state, a)) - V_L(T(cur_state, a)) 
        它衡量了按照当前最优策略走一步后，所到达状态的值函数区间的宽度。'''
        B = sum(self.get_expected_diff(cur_state, a).values())
        
        
        '''计算当前状态 `cur_state` 的归一化不确定性 `diff`。
        #    diff = (V_U(cur_state) - V_L(cur_state)) / tau
        #    `tau` 是 BRTDP 的一个超参数，用于调整对当前状态不确定性的敏感度 [cite: 456]。
        #    `diff` 衡量了当前状态值函数区间的宽度（经过 tau 调整）。'''
        diff = (self.v_u[(cur_state.get_repr(), self.subtask)] - self.v_l[(cur_state.get_repr(), self.subtask)])/self.tau
        self.cur_state = cur_state
        
        
        '''决策：是否调用 BRTDP 主循环 (`self.Main`) 进行探索和值更新？
        #    条件 `B > diff` 的含义是：如果按照当前最优策略（基于 V_L）走一步后，
        #    到达的下一个状态的不确定性 (`B`) 比当前状态的归一化不确定性 (`diff`) 还要大，
        #    说明当前策略可能会引导到一个值函数边界很宽、估计很不准确的区域。
        #    这种情况下，BRTDP 认为有必要进行一次或多次 SampleTrial（通过调用 `self.Main`）
        #    来探索这个区域，并通过回溯更新来收紧相关状态的值函数边界 V_L 和 V_U '''
        if (B > diff):
            print('exploring, B: {}, diff: {}'.format(B, diff))
            self.main()



        ''' === 步骤 4: 确定最终返回的动作 ===
        # 无论是否执行了 self.Main()，现在都需要根据（可能已更新的）V_L 来选择最终要执行的动作。'''
        
        '''检查当前规划状态 `cur_state` 是否已经是目标状态。 
        Determine best action after BRTDP.'''
        if self.is_goal_state(cur_state.get_repr()):
             # 如果已经是目标状态，说明对于这个子任务，当前无需再执行任何动作。
            print('already at goal state, self.cur_obj_count:', self.cur_obj_count)
            return None
        else:
            '''# 如果还未到达目标状态：
            # b. 再次获取当前规划状态 `cur_state` 下的所有可用动作。
            #    （如果在 self.Main() 中状态发生了模拟演进，这里获取的是演进后状态的动作，
            #     但从代码看 self.Main() 结束后似乎没有更新 cur_state，所以这里获取的仍是原始 cur_state 的动作）。'''
            actions = self.get_actions(state_repr=cur_state.get_repr())
            
            ''' 对每个可能动作 a，计算其 Q 值。
               关键：Q 值计算使用 cur_state，因此模拟 T(cur_state, a) 时会考虑预测碰撞。
              导致预测碰撞的动作 a 会得到很高的 Q 值 (成本高)。'''
            qvals = [self.Q(state=cur_state, action=a, value_f=self.v_l)
                    for a in actions]
            print([x for x in zip(actions, qvals)])
            print('upper is', self.v_u[(cur_state.get_repr(), self.subtask)])
            print('lower is', self.v_l[(cur_state.get_repr(), self.subtask)])

            action_index = argmin(np.array(qvals))
            a = actions[action_index]

            print('chose action:', a)
            print('cost:', self.cost(cur_state, a))
            return a



