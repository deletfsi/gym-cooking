import recipe_planner.utils as recipe
from delegation_planner.delegator import Delegator
from delegation_planner.utils import SubtaskAllocDistribution
from navigation_planner.utils import get_subtask_obj, get_subtask_action_obj, get_single_actions
from utils.interact import interact
from utils.utils import agent_settings

from collections import defaultdict, namedtuple
from itertools import permutations, product, combinations
import scipy as sp
import numpy as np
import copy

SubtaskAllocation = namedtuple("SubtaskAllocation", "subtask subtask_agent_names")


class BayesianDelegator(Delegator):

    def __init__(self, agent_name, all_agent_names,
            model_type, planner, none_action_prob):
        """Initializing Bayesian Delegator for agent_name.

        Args:
        "bd": 贝叶斯委托 (论文中的完整模型)。
        "up": 均匀先验 (以均匀信念开始，使用贝叶斯规则更新)。
        "fb": 固定信念 (使用空间先验，但从不根据动作更新)。
        "dc": 分而治之 (只考虑智能体处理不同任务的分配，从不协作完成同一任务)。
        "greedy": 贪婪 (智能体只考虑自己能完成的任务，忽略其他智能体)。
planner: 属于该智能体的导航规划器实例（如 E2E_BRTDP）。需要用它来计算动作似然（逆向规划）和空间先验。
none_action_prob: 用于信念更新的参数，表示当分配的子任务为 None 时，智能体采取“原地不动”动作 (0, 0) 的假定概率。

            agent_name: Str of agent's name.
            all_agent_names: List of str agent names.
            model_type: Str of model type. Must be either "bd"=Bayesian Delegation,
                "fb"=Fixed Beliefs, "up"=Uniform Priors, "dc"=Divide & Conquer,
                "greedy"=Greedy.
            planner:    Navigation Planner object, belonging to agent.
            none_action_prob:   Float of probability for taking (0, 0) in a None subtask.
        """
        self.name = 'Bayesian Delegator'
        self.agent_name = agent_name
        self.all_agent_names = all_agent_names
        self.probs = None
        self.model_type = model_type
        self.priors = 'uniform' if model_type == 'up' else 'spatial'
        self.planner = planner
        self.none_action_prob = none_action_prob






    def should_reset_priors(self, obs, incomplete_subtasks):
        """Returns whether priors should be reset.

        Priors should be reset when 1) They haven't yet been set or
        2) If the possible subtask allocations to infer over have changed.

        Args:
            obs: Copy of the environment object. Current observation
                of environment.
            incomplete_subtasks: List of subtasks. Subtasks have not
                yet been completed according to agent.py.

        Return:
            Boolean of whether or not the subtask allocations have changed.
        """
        
        # 情况 1: 先验概率从未被设置过 , 需要设置/重置先验
        if self.probs is None:
            return True
        # Get currently available subtasks.
        self.incomplete_subtasks = incomplete_subtasks
        probs = self.get_subtask_alloc_probs()
        probs = self.prune_subtask_allocs(
                observation=obs, subtask_alloc_probs=probs)
        # Compare previously available subtasks with currently available subtasks.
        return not(len(self.probs.enumerate_subtask_allocs()) == len(probs.enumerate_subtask_allocs()))

    def get_subtask_alloc_probs(self):
        """Return the appropriate belief distribution (determined by model type) over
        subtask allocations (combinations of all_agent_names and incomplete_subtasks)."""
        if self.model_type == "greedy":
            probs = self.add_greedy_subtasks()
        elif self.model_type == "dc":
            probs = self.add_dc_subtasks()
        else:
            probs = self.add_subtasks()
        return probs







    def subtask_alloc_is_doable(self, env, subtask, subtask_agent_names):
        """
        判断一个特定的“子任务分配”（即让指定的 subtask_agent_names 去执行 subtask）在当前的环境状态 env下是否 物理上可行。
        这通常用来在生成假设空间后进行剪枝，去掉那些明显不可能完成的分配。
        
        
        智能体 agent-1 位于 (2, 1)。
        一个新鲜番茄 Tomato 位于柜台上 (6, 1)。
        一个砧板 Cutboard 位于 (2, 5)。
        假设从 (2, 1) 到 (6, 1) 再到 (2, 5) 的最短路径距离是 10 (先走 4 步拿到番茄，再走 6 步到砧板)。
        假设世界周长 env.world.perimeter 是 30。
        调用: subtask_alloc_is_doable(env, subtask=Chop(Tomato), subtask_agent_names=('agent-1',))

        执行流程:

        subtask 不是 None。
        agent_locs = [(2, 1)]。
        get_subtask_obj 返回 start_obj = FreshTomato, goal_obj = ChoppedTomato。
        get_subtask_action_obj 返回 Cutboard 对象。
        env.get_AB_locs_given_objs (针对 Chop) 返回 A_locs = [(6, 1)] (新鲜番茄的位置), B_locs = [(2, 5)] (砧板的位置)。
        env.world.get_lower_bound_between 被调用，参数包括 agent_locs=((2, 1),), A_locs=((6, 1),), B_locs=((2, 5),)。
        get_lower_bound_between_helper (对于单智能体 Chop) 计算从 (2, 1) 到 (6, 1) 再到 (2, 5) 的距离，假设结果是 distance = 10。
        最后比较 10 < 30，结果为 True。
        结论: 这个子任务分配是可行 (doable) 的

        反例: 如果番茄被完全围起来，get_lower_bound_between_helper 找不到路径 （比如 被完全切分的厨房）
        
        Return whether subtask allocation (subtask x subtask_agent_names) is doable
        in the current environment state."""
        # Doing nothing is always possible.
        if subtask is None:
            return True
        
        '''找到被分配执行该子任务的智能体 (subtask_agent_names) 在当前环境 env 中的位置 agent_locs'''
        agent_locs = [agent.location for agent in list(filter(lambda a: a.name in subtask_agent_names, env.sim_agents))]
        
        ''' get_subtask_obj: 根据子任务类型返回起始和目标状态的物品对象。例如，对于 Chop(Tomato)，start_obj 是新鲜番茄对象，goal_obj 是切好的番茄对象。
         对于 Merge(ChoppedTomato, Plate)，start_obj 是包含切好番茄和盘子对象的列表，goal_obj 是包含合并后物体的对象 '''
        start_obj, goal_obj = get_subtask_obj(subtask=subtask)
        
        '''get_subtask_action_obj: 返回子任务需要交互的固定位置对象。
        例如，Chop 需要 Cutboard (砧板)，Deliver 需要 Delivery (交付点)。Merge 通常不需要固定的交互对象，返回 None'''
        subtask_action_obj = get_subtask_action_obj(subtask=subtask)
        A_locs, B_locs = env.get_AB_locs_given_objs(
                subtask=subtask,
                subtask_agent_names=subtask_agent_names,
                start_obj=start_obj,
                goal_obj=goal_obj,
                subtask_action_obj=subtask_action_obj)
        
        '''智能体的当前位置 agent_locs 出发，完成与 A_locs 和 B_locs 相关的交互所需的最短路径距离（或成本）的下界 distance。
        遍历所有可能的 A 坐标和 B 坐标对 (product(A_locs, B_locs))，并对每一对计算距离下界，最终返回所有对中的最小值。'''
        distance = env.world.get_lower_bound_between(
                subtask=subtask,
                agent_locs=tuple(agent_locs),
                A_locs=tuple(A_locs),
                B_locs=tuple(B_locs))
        # Subtask allocation is doable if it's reachable between agents and subtask objects.
        return distance < env.world.perimeter





    def get_lower_bound_for_subtask_alloc(self, obs, subtask, subtask_agent_names):
        """ level 0 , 只考虑自己当前完成当前任务的 成本， 值越小越好
        这个函数的主要目的是估算对于一个特定的子任务分配（即让 subtask_agent_names 执行 subtask），
        从当前观察到的状态 obs 开始，完成该子任务所需的最小预期成本（或负的奖励，即价值的下界）。
        这个估算值主要被用在 get_spatial_priors 函数中，用来计算空间先验概率：
        成本越低（价值下界越接近0或更高），完成任务被认为越“容易”或越“近”，因此该分配获得更高的先验概率
        
        Return the value lower bound for a subtask allocation
        (subtask x subtask_agent_names)."""
        
        
        '''如果分配的任务是 None，意味着智能体处于空闲状态，完成这个“任务”的成本自然是 0。'''
        if subtask is None:
            return 0
        
        '''     #   调用导航规划器（主要目的是触发其内部状态设置和价值计算）
                #    注意：这里的 _ 表明 get_next_action 的返回值（即动作）被忽略了。
                #    关键在于调用这个函数会配置 self.planner 处理当前的 obs, subtask, 和 subtask_agent_names。
                #    传入空的 other_agent_planners={} 表示这里是为了获取基础成本估算，进行 Level-0 规划。
                # 在估算这个子任务的成本时，它不考虑其他智能体的行为，只是单纯地看当前分配的 subtask_agent_names 完成 subtask 的基础成本是多少。'''
        _ = self.planner.get_next_action(
                env=obs,
                subtask=subtask,
                subtask_agent_names=subtask_agent_names,
                other_agent_planners={})
        
        ''' #    获取价值下界 (self.planner.v_l[...]): 
            #    在调用 get_next_action 后，self.planner 内部的状态 (cur_state) 和价值函数 (v_l)
            #    已经被更新（或至少被计算过）以反映传入的 obs 和 subtask。
            #    这里直接从规划器的 v_l 字典中读取对应状态和子任务的价值下界。
            #    代码直接读取这个值。这个值代表了 BRTDP 算法当前对完成该子任务所需总成本的最低估计'''
        value = self.planner.v_l[(self.planner.cur_state.get_repr(), subtask)]
        return value





    def prune_subtask_allocs(self, observation, subtask_alloc_probs):
        """
        这个函数的作用是对初步生成的子任务分配假设空间 (subtask_alloc_probs) 进行清理和过滤（剪枝）。
        它移除了那些基于当前环境状态 observation 判断为不合理或不可能的分配方案，确保后续的贝叶斯推理只在有效的假设上进行。
        
        例子:

        假设 subtask_alloc_probs 初始包含以下几种分配方案（为了简化，只展示部分）：

        alloc1 = [(Chop(T), (A1,)), (Chop(L), (A2,))]
        alloc2 = [(Chop(T), (A1,)), (Get(Plate), (A2,))] # 假设 Get(Plate) 在当前状态不可行 (比如没有盘子)
        alloc3 = [(Chop(T), (A1,)), (None, (A2, A3))] # 假设有 A1, A2, A3 三个智能体
        alloc4 = [(None, (A1,)), (None, (A2,))]
        执行 prune_subtask_allocs(observation, subtask_alloc_probs):

        处理 alloc1:     结果: alloc1 保留
        检查 t1 = (Chop(T), (A1,))：假设 subtask_alloc_is_doable 返回 True。
        检查 t2 = (Chop(L), (A2,))：假设 subtask_alloc_is_doable 返回 True。
         
        
        处理 alloc2: 结果: alloc2 被删除。
        检查 t1 = (Chop(T), (A1,))：假设 subtask_alloc_is_doable 返回 True。
        检查 t2 = (Get(Plate), (A2,))：假设 subtask_alloc_is_doable 返回 False (因为盘子不可获取)。 =》t2 不可行 break
         
        处理 alloc3:    =》t3 不可行 break
        检查 t1 = (Chop(T), (A1,))：假设 subtask_alloc_is_doable 返回 True。
        检查 t2 = (None, (A2, A3))：   =>   不可行 break

        处理 alloc4:  =>   None：all([t.subtask is None for t in alloc4])  , t4 不可行 break
        检查 t1 = (None, (A1,))：通过可行性和联合 None 检查。
        检查 t2 = (None, (A2,))：通过可行性和联合 None 检查。
        
        最终，函数返回的 subtask_alloc_probs 对象将只包含 alloc1 (以及其他所有通过检查的分配方案)。
        这个剪枝过程确保了后续的推理和决策是基于一个更合理、更符合当前世界状况的假设集合
        

        Removing subtask allocs from subtask_alloc_probs that are
        infeasible or where multiple agents are doing None together."""
        for subtask_alloc in subtask_alloc_probs.enumerate_subtask_allocs():
            for t in subtask_alloc:
                # Remove unreachable/undoable subtask subtask_allocations.
                if not self.subtask_alloc_is_doable(
                        env=observation,
                        subtask=t.subtask,
                        subtask_agent_names=t.subtask_agent_names):
                    subtask_alloc_probs.delete(subtask_alloc)
                    break
                # Remove joint Nones (cannot be collaborating on doing nothing).
                if t.subtask is None and len(t.subtask_agent_names) > 1:
                    subtask_alloc_probs.delete(subtask_alloc)
                    break

            # Remove all Nones (at least 1 agent must be doing something).
            if all([t.subtask is None for t in subtask_alloc]) and len(subtask_alloc) > 1:
                subtask_alloc_probs.delete(subtask_alloc)

        return subtask_alloc_probs





    def set_priors(self, obs, incomplete_subtasks, priors_type):
        """Setting the prior probabilities for subtask allocations."""
        print('{} setting priors'.format(self.agent_name))
        self.incomplete_subtasks = incomplete_subtasks

        ''' 生成初始假设空间：根据 model_type 获取所有理论上的分配可能'''
        probs = self.get_subtask_alloc_probs()
        
        '''   剪枝假设空间：移除基于当前 observation 不可行或无效的分配'''
        probs = self.prune_subtask_allocs(
                observation=obs, subtask_alloc_probs=probs)
        
        '''最终，函数返回的 subtask_alloc_probs 对象将只包含 alloc1初步归一化：给所有通过剪枝的、有效的分配方案赋予均匀概率 '''
        probs.normalize()

        if priors_type == 'spatial':
            ''' 如果是空间先验 (BD, FB, DC 模型)  调用 get_spatial_priors 重新加权 probs 中的概率 
               基于任务的估计成本/距离（成本低的先验概率高）, 将加权后的（非均匀）分布赋值给 self.probs'''
            self.probs = self.get_spatial_priors(obs, probs)
        elif priors_type == 'uniform':
            '''# 如果是均匀先验 (UP 模型)  直接将剪枝后得到的均匀分布 probs 赋值给 self.probs
             Do nothing because probs already initialized to be uniform.'''
            self.probs = probs

        '''安全检查：确保至少有一个分配方案（特别是对 greedy 或 dc）    如果剪枝后或由于模型限制导致 probs 为空，添加一个 agent 自己做 None 的任务'''
        self.ensure_at_least_one_subtask()
        
        '''最终归一化：确保 self.probs 存储的最终先验分布概率总和为 1'''
        self.probs.normalize()





    def get_spatial_priors(self, obs, some_probs):
        """
        根据空间度量（例如，估计的成本/距离）设置先验概率。

        Args:
            obs: 当前的环境观察对象。
            some_probs: 一个 SubtaskAllocDistribution 对象，通常在调用此函数前
                        已经过剪枝并被赋予了初始（可能是均匀的）概率。

        Returns:
            SubtaskAllocDistribution: 修改后的概率分布对象，其中每个分配的概率
                                     已根据空间成本加权（但尚未归一化）。
                                     
          # 根据成本（距离）的倒数进行加权。成本越低，权重越高。
        # 遍历 some_probs 中所有可能的子任务分配方案 (subtask_alloc)
        
        假设：

        有 2 个智能体：A1, A2。
        有 2 个未完成的任务：T1 = Chop(Tomato), T2 = Chop(Lettuce)。
        some_probs 经过初始生成和剪枝后，包含以下两种可能的分配方案，初始概率（均匀）均为 0.5：
        alloc1 = [(T1, (A1,)), (T2, (A2,))] (A1 切番茄, A2 切生菜)
        alloc2 = [(T1, (A2,)), (T2, (A1,))] (A2 切番茄, A1 切生菜)
        假设 get_lower_bound_for_subtask_alloc 返回的成本如下 ：
        Cost(A1 执行 T1) = 10
        Cost(A2 执行 T2) = 20
        Cost(A2 执行 T1) = 15
        Cost(A1 执行 T2) = 12
          
        计算过程:

        处理 alloc1 = [(T1, (A1,)), (T2, (A2,))]:

        t = (T1, (A1,)): Cost = 10, Inverse Cost = 1/10 = 0.1
        t = (T2, (A2,)): Cost = 20, Inverse Cost = 1/20 = 0.05
        total_weight = 0.1 + 0.05 = 0.15
        内层循环结束时，最后一个 t 是 (T2, (A2,))。len(t) = 2。
        factor = 2 
        2   ×0.15=4×0.15=0.6    
        some_probs.update(alloc1, factor=0.6)。 alloc1 的权重变为 0.5×0.6=0.3 。   
        处理 alloc2 = [(T1, (A2,)), (T2, (A1,))]:

        t = (T1, (A2,)): Cost = 15, Inverse Cost = 1/15 ≈ 0.0667
        t = (T2, (A1,)): Cost = 12, Inverse Cost = 1/12 ≈ 0.0833
        total_weight ≈ 0.0667 + 0.0833 = 0.15
        内层循环结束时，最后一个 t 是 (T2, (A1,))。len(t) = 2。
        factor = 2 
        2   ×0.15=4×0.15=0.6    
        some_probs.update(alloc2, factor=0.6)。 alloc2 的权重变为 0.5×0.6=0.3 。
                
        """
        for subtask_alloc in some_probs.enumerate_subtask_allocs():
            total_weight = 0 
            for t in subtask_alloc:
                if t.subtask is not None:
                    '''   --- 注意：下面的权重因子计算可能与注释意图不符 ---
                    # 原始注释是 "Weight by number of nonzero subtasks." (根据非空子任务的数量加权)
                    # 但实际代码使用了 `len(t)**2`。这里的 `t` 是内层循环最后处理的任务指派。
                    # `len(t)` 对于 SubtaskAllocation 命名元组总是返回 2 （因为它有两个字段：subtask 和 subtask_agent_names）。
                    # 因此，实际的因子是 4 * total_weight。
                    #
                    # 一个更符合注释意图的实现可能是计算 subtask_alloc 中非 None 任务的数量 num_active_tasks，
                    # 然后使用类似 num_active_tasks**2 * total_weight 作为因子。
                    '''
                    total_weight += 1.0 / float(self.get_lower_bound_for_subtask_alloc(
                        obs=copy.copy(obs),
                        subtask=t.subtask,
                        subtask_agent_names=t.subtask_agent_names))
            # Weight by number of nonzero subtasks.
            some_probs.update(
                    subtask_alloc=subtask_alloc,
                    factor=len(t)**2. * total_weight)
             
        return some_probs





    def get_other_agent_planners(self, obs, backup_subtask):
        """
        根据自身信念推断其他智能体会做什么，并为他们创建和配置规划器。
        这是 Level 1 规划的关键，用于模拟其他智能体的行为。

        Args:
            obs: 当前的环境观察对象。
            backup_subtask: 一个备用的子任务。当推断某个其他智能体
                           最可能的任务是 None 时，会使用这个备用子任务
                           （通常假设是与当前智能体合作执行此任务）。

        Returns:
            dict: 一个字典，将其他智能体的名称映射到为其配置好的
                  导航规划器 (Planner) 实例的副本。
   
          # A dictionary mapping agent name to a planner.
        # The planner is based on THIS agent's planner because agents are decentralized. 
                
                
        Use own beliefs to infer what other agents will do."""
 
        planners = {}
        for other_agent_name in self.all_agent_names:
            # Skip over myself.
            if other_agent_name != self.agent_name:
                # Get most likely subtask and subtask agents for other agent
                # based on my beliefs.
                '''      
                    根据当前智能体的信念 (self.probs)，
                    获取 other_agent_name 最可能执行的子任务 (subtask)
                    以及执行该子任务的智能体组合 (subtask_agent_names)。
                    self.select_subtask 会查找概率最高的那个完整分配方案，
                    然后返回其中分配给 other_agent_name 的具体任务信息。'''
                subtask, subtask_agent_names = self.select_subtask(
                        agent_name=other_agent_name)

                if subtask is None:
                    # Using cooperative backup_subtask for this agent's None subtask.
                    subtask = backup_subtask
                    subtask_agent_names = tuple(sorted([other_agent_name, self.agent_name]))

                # Assume your planner for other agents with the right settings.
                planner = copy.copy(self.planner)
                
                ''' 使用推断出的子任务和执行者来配置这个规划器副本。
                 set_settings 会设置规划器的起始状态、目标、是否联合规划等。
                 注意：这里传递的 other_agent_planners 参数默认为 {}，
                 因此配置出的 planner 实际上是一个 Level 0 的规划器，
                 它只针对给定的 subtask 和 subtask_agent_names 进行规划。'''
                planner.set_settings(env=copy.copy(obs),
                                     subtask=subtask,
                                     subtask_agent_names=subtask_agent_names
                                     )
                planners[other_agent_name] = planner
        return planners





    def get_appropriate_state_and_other_agent_planners(self,
            obs_tm1, backup_subtask, no_level_1):
        """Return Level 1 planner if no_level_1 is False, otherwise
        return a Level 0 Planner."""
        # Get appropriate observation.
        ''' Level 0 规划 : 直接使用上一步的观察 obs_tm1 作为状态 ,   # 假设其他智能体是固定的（不进行预测）'''
        if no_level_1:
            # Level 0 planning: Just use obs_tm1.
            state = obs_tm1
            # Assume other agents are fixed.
            other_planners = {}
        else:
            '''Level 1 规划: 根据当前智能体的信念来修改状态 # 调用导航规划器的内部方法，预测其他智能体的动作并修改状态'''
            # Level 1 planning: Modify the state according to my beliefs.
            state, _ = self.planner._get_modified_state_with_other_agent_actions(state=obs_tm1)
            # Get other agent planners under my current beliefs.
            other_planners = self.get_other_agent_planners(
                    obs=obs_tm1, backup_subtask=backup_subtask)
        return state, other_planners







    def prob_nav_actions(self, obs_tm1, actions_tm1, subtask,
            subtask_agent_names, beta, no_level_1):
        """Return probabability that subtask_agents performed subtask, given
        previous observations (obs_tm1) and actions (actions_tm1).

        Args:
            obs_tm1: Copy of environment object. Represents environment at t-1.
            actions_tm1: Dictionary of agent actions. Maps agent str names to tuple actions.
            subtask: Subtask object to perform inference for.
            subtask_agent_names: Tuple of agent str names, of agents who perform subtask.
                subtask and subtask_agent_names make up subtask allocation.
            beta: Beta float value for softmax function.
            no_level_1: Bool, whether to turn off level-k planning.
        Returns:
            A float probability update of whether agents in subtask_agent_names are
            performing subtask.
        """
        """
        计算在给定上一个观察(obs_tm1)和上一个动作(actions_tm1)的情况下，
        假设智能体组 (subtask_agent_names) 正在执行子任务 (subtask) 时，
        他们采取这些实际动作的概率（似然）。

        Args:
            obs_tm1: 环境对象副本，代表 t-1 时刻的环境状态。
            actions_tm1: 智能体动作字典，映射智能体名称到其在 t-1 时刻采取的动作元组。
            subtask: 要进行推断的目标子任务对象。
            subtask_agent_names: 执行该子任务的智能体名称元组。
                                subtask 和 subtask_agent_names 共同构成一个任务分配假设。
            beta: Softmax 函数中的 beta 值（理性参数）。beta 越大，越倾向于选择最优动作。
            no_level_1: 布尔值，是否关闭 Level-1 规划（即执行 Level-0 规划）。

        Returns:
            float: 一个概率值，表示在给定子任务假设下观测到实际动作的可能性。
                   这个值用于贝叶斯信念更新。
                   
                   
                   
         if subtask is None: 的 代码的例子       
            # --- 示例解释 (None 任务部分) ---
            # 假设:
            # obs_tm1: agent-1 在 (2,1)。周围有4个空格子可以移动。
            # actions_tm1: {'agent-1': (0, 0)} (agent-1 实际原地不动)
            # self.none_action_prob = 0.6
            # beta = 1.0

            # 目标: 计算假设 "agent-1 正在执行 None 任务" 的似然
            # 调用: prob_nav_actions(obs_tm1, actions_tm1, None, ('agent-1',), 1.0, True)

            # 执行流程 (None 部分):
            # 1. subtask is None。
            # 2. agent_name_doing_none = 'agent-1'。
            # 3. get_single_actions 假设返回 5 个动作 [(0,0), (1,0), (-1,0), (0,1), (0,-1)]。
            # 4. num_move_actions = 5 - 1 = 4。
            # 5. move_action_prob = (1.0 - 0.6) / 4 = 0.4 / 4 = 0.1。
            # 6. diffs = [0.6] + [0.1] * 4 = [0.6, 0.1, 0.1, 0.1, 0.1]。
            # 7. softmax_diffs = softmax(1.0 * [0.6, 0.1, 0.1, 0.1, 0.1])。
            #    计算 exp: [e^0.6, e^0.1, e^0.1, e^0.1, e^0.1] = [1.822, 1.105, 1.105, 1.105, 1.105]
            #    求和: sum = 1.822 + 4 * 1.105 = 6.242
            #    归一化: softmax_diffs = [0.292, 0.177, 0.177, 0.177, 0.177] (近似值)
            # 8. 检查实际动作: actions_tm1['agent-1'] == (0, 0)，条件为 True。
            # 9. 返回 softmax_diffs[0] = 0.292。

            # 结论 (None 部分): 在假设 agent-1 无任务 (None) 的情况下，观察到它原地不动的概率大约是 0.292。

            # 如果 actions_tm1['agent-1'] 是 (1,0) (向右移动)：
            # 8. 检查实际动作: actions_tm1['agent-1'] == (0, 0)，条件为 False。
            # 9. 进入 else 分支。
            # 10. 返回 softmax_diffs[1] = 0.177。
            # 结论 (None 部分): 在假设 agent-1 无任务 (None) 的情况下，观察到它向右移动的概率大约是 0.177。   
                   
            
            
            感觉这个有问题啊 ， 需要更加细致的打印了。。。
            场景设置:

智能体: agent-1 (我们自己), agent-2, agent-3。
环境 (obs_tm1):
agent-1 @ (2,1)
agent-2 @ (5,1)
agent-3 @ (5,5)
番茄 (T) @ (6,1)
生菜 (L) @ (0,1)
砧板1 (C1) @ (2,5)
砧板2 (C2) @ (4,5)
盘子 (P) @ (0,5)
实际动作 (actions_tm1):
agent-1: (1, 0) (向右移动，靠近番茄)
agent-2: (-1, 0) (向左移动，靠近生菜)
agent-3: (0, -1) (向上移动)
参数: beta = 1.0, no_level_1 = True (Level-0)
目标: 计算假设 agent-2 正在 Chop(Lettuce) 的似然。

调用: prob_nav_actions(obs_tm1, actions_tm1, Chop(Lettuce), ('agent-2',), 1.0, True)

执行流程 (非 None 部分):

提取实际动作:

subtask_agent_names = ('agent-2',)
action = actions_tm1['agent-2'] = (-1, 0) (A2 向左移动)。
获取状态和规划器:

state = obs_tm1, other_planners = {}。
配置规划器:

self.planner.set_settings(env=copy.copy(obs_tm1), subtask=Chop(Lettuce), subtask_agent_names=('agent-2',), other_planners={})。
规划器配置为评估 agent-2 从 (5,1) 出发去拿生菜 L @ (0,1) 并送到砧板 C1 @ (2,5) 或 C2 @ (4,5) 的情况。
计算实际动作的 Q 值 (old_q):

old_q = self.planner.Q(state=obs_tm1, action=(-1, 0), value_f=self.planner.v_l)。
动作 (-1, 0) 使 agent-2 从 (5,1) 移动到 (4,1)，更靠近生菜 L @ (0,1)。这是一个好动作。
假设规划器计算出的 Q 值为 old_q = -12.0 (预期成本较低)。
获取所有有效动作 (valid_nav_actions):

agent-2 在 (5,1)。假设它可以向左、向上、向下、不动。向右是边界。
valid_nav_actions = self.planner.get_actions(state_repr=obs_tm1.get_repr())
假设返回 [(0,0), (-1,0), (0,1), (0,-1)] (顺序可能不同)。
检查有效性:

assert (-1, 0) in valid_nav_actions。通过。
联合任务过滤:

len(subtask_agent_names) 为 1，跳过。
计算 Q 值差异 (qdiffs):

我们需要计算 agent-2 在状态 (5,1) 下，以 Chop(Lettuce) 为目标时，其他动作的 Q 值。
Q((-1,0)) = -12.0 (实际动作，最佳)
假设其他 Q 值：
Q((0,0)) = -14.0 (不动，原地等待成本增加)
Q((0,1)) = -15.0 (向上，远离 L 和 C)
Q((0,-1)) = -15.0 (向下，远离 L 和 C)
计算 qdiffs = [old_q - Q(alt)]。假设 valid_nav_actions 的顺序是 [(0,0), (-1,0), (0,1), (0,-1)]：
qdiffs[0] = -12.0 - (-14.0) = 2.0 (对应 (0,0))
qdiffs[1] = -12.0 - (-12.0) = 0.0 (对应 (-1,0)，实际动作)
qdiffs[2] = -12.0 - (-15.0) = 3.0 (对应 (0,1))
qdiffs[3] = -12.0 - (-15.0) = 3.0 (对应 (0,-1))
所以 qdiffs = [2.0, 0.0, 3.0, 3.0]。
计算 softmax_diffs:

softmax_diffs = softmax(beta * qdiffs) = softmax(1.0 * [2.0, 0.0, 3.0, 3.0])
exp values: [e^2, e^0, e^3, e^3] = [7.389, 1.0, 20.086, 20.086]
sum = 7.389 + 1.0 + 2 * 20.086 = 48.561
softmax_diffs = [0.152, 0.021, 0.414, 0.414] (近似值)
找到实际动作索引:

实际动作是 (-1, 0)。在假设的 valid_nav_actions 顺序 [(0,0), (-1,0), (0,1), (0,-1)] 中，它的索引是 1。
返回结果:

return softmax_diffs[1] = 0.021。
                   
        """
        
        print("[BayesianDelgation.prob_nav_actions] Calculating probs for subtask {} by {}".format(str(subtask), ' & '.join(subtask_agent_names)))
        assert len(subtask_agent_names) == 1 or len(subtask_agent_names) == 2   # 断言：确保任务分配给 1 个或 2 个智能体

       
        # --- 情况 1: 推断子任务为 None (假设智能体空闲或随机移动) ---
        if subtask is None:
            # 断言：不能有两个智能体 *一起* 被分配到 None 任务 (无协作意义)
            assert len(subtask_agent_names) != 2, "Two agents are doing None."
            sim_agent = list(filter(lambda a: a.name == self.agent_name, obs_tm1.sim_agents))[0]
            # Get the number of possible actions at obs_tm1 available to agent.
            # 获取该智能体在 obs_tm1 状态下所有可能的动作数量（排除原地不动 (0,0)）
            num_actions = len(get_single_actions(env=obs_tm1, agent=sim_agent)) -1
            action_prob = (1.0 - self.none_action_prob)/(num_actions)    # exclude (0, 0)
            diffs = [self.none_action_prob] + [action_prob] * num_actions
            softmax_diffs = sp.special.softmax(beta * np.asarray(diffs))
           
            # 检查智能体实际采取的动作 actions_tm1[subtask_agent_names[0]]
            # 如果实际动作是原地不动 (0, 0)
            # 返回 Softmax 概率列表中对应 (0, 0) 的概率
            # 根据 diffs 的构建方式，这应该是列表的第一个元素 (索引 0)
            if actions_tm1[subtask_agent_names[0]] == (0, 0):
                return softmax_diffs[0]
             # 如果实际动作是移动 (任何非 (0, 0) 的动作)
            # 返回 Softmax 概率列表中对应 *任何一个* 移动动作的概率
            # 因为所有移动动作的基础概率相同，Softmax 后的概率也应该相同
            # 根据 diffs 的构建方式，这应该是列表的第二个元素 (索引 1)
            # (代表了所有移动动作共享的那个概率值)
            else:
                return softmax_diffs[1]

        
        
        
        # Perform inference over all non-None subtasks.
        # Calculate Q_{subtask}(obs_tm1, action) for all actions.
        # --- 情况 2: 推断子任务不是 None ---
        # (注释同前一个回答)
        # 获取每个 agent 的实际执行的动作
        action = tuple([actions_tm1[a_name] for a_name in subtask_agent_names])
        if len(subtask_agent_names) == 1:
            action = action[0]
            
         # 获取状态和规划器
        state, other_planners = self.get_appropriate_state_and_other_agent_planners(
                obs_tm1=obs_tm1, backup_subtask=subtask, no_level_1=no_level_1)
        
        # 配置规划器
        self.planner.set_settings(env=obs_tm1, subtask=subtask,
                subtask_agent_names=subtask_agent_names,
                other_agent_planners=other_planners)
        
        # 计算实际动作的 Q 值
        old_q = self.planner.Q(state=state, action=action,
                value_f=self.planner.v_l)

        # 获取所有有效动作    Collect actions the agents could have taken in obs_tm1.
        valid_nav_actions = self.planner.get_actions(state_repr=obs_tm1.get_repr())

        # 检查有效性check action taken is in the list of actions available to agents in obs_tm1.
        assert action in valid_nav_actions, "valid_nav_actions: {}\nlocs: {}\naction: {}".format(
                valid_nav_actions, list(filter(lambda a: a.location, state.sim_agents)), action)

        #  过滤联合任务的有效动作 If subtask allocation is joint, then find joint actions that match what the other
        # agent's action_tm1.
        if len(subtask_agent_names) == 2 and self.agent_name in subtask_agent_names:
            other_index = 1 - subtask_agent_names.index(self.agent_name)
            valid_nav_actions = list(filter(lambda x: x[other_index] == action[other_index], valid_nav_actions))

        # 计算 Q 值差异 Calculating the softmax Q_{subtask} for each action.
        qdiffs = [old_q - self.planner.Q(state=state, action=nav_action, value_f=self.planner.v_l)
                for nav_action in valid_nav_actions]
        
        #计算 Softmax 概率
        softmax_diffs = sp.special.softmax(beta * np.asarray(qdiffs))
       
        # 找到实际动作的索引，返回实际动作对应的 Softmax 概率  # Taking the softmax of the action actually taken.
        return softmax_diffs[valid_nav_actions.index(action)]







    def get_other_subtask_allocations(self, remaining_agents, remaining_subtasks, base_subtask_alloc):
        """
         为剩余的智能体分配剩余的子任务，以扩展一个基础的部分分配方案。

        Args:
            remaining_agents (list): 尚未被分配任务的智能体名称列表。
            remaining_subtasks (list): 尚未被分配的子任务对象列表 (可能包含 None)。
            base_subtask_alloc (list): 一个列表，包含已经确定的 SubtaskAllocation 对象，
                                       构成了当前正在构建的完整分配方案的基础部分。

        Returns:
            list: 一个列表，其中每个元素都是一个 *完整* 的子任务分配方案 (也是一个 SubtaskAllocation 对象的列表)，
                  这些方案都是在 base_subtask_alloc 基础上扩展得到的。

        例子 (来自文档字符串):
            如果 base_subtask_alloc = [SubtaskAllocation(subtask=Chop(T), subtask_agent_names=('agent-1', 'agent-2'))]
            并且 remaining_agents = ['agent-3'], remaining_subtasks = [Chop(L), None]
            那么此函数应该返回类似这样的列表：
            [
                [SubtaskAllocation(Chop(T), ('agent-1', 'agent-2')), SubtaskAllocation(Chop(L), ('agent-3',))],
                [SubtaskAllocation(Chop(T), ('agent-1', 'agent-2')), SubtaskAllocation(None,    ('agent-3',))]
            ]
            
            
            Return a list of subtask allocations to be added onto `subtask_allocs`.

        Each combination should be built off of the `base_subtask_alloc`.
        Add subtasks for all other agents and all other recipe subtasks NOT in
        the ignore set.

        e.g. base_subtask_combo=[
            SubtaskAllocation(subtask=(Chop(T)),
            subtask_agent_names(agent-1, agent-2))]
        To be added on: [
            SubtaskAllocation(subtask=(Chop(L)),
            subtask_agent_names(agent-3,))]
        Note the different subtask and the different agent.
        """
        other_subtask_allocs = []
        
        # --- 基本情况 1: 没有剩余的智能体需要分配 ---
        # 如果 remaining_agents 列表为空，说明 base_subtask_alloc 已经是一个完整的分配方案。
        if not remaining_agents:
            return [base_subtask_alloc]



        # This case is hit if we have more agents than subtasks.
        # --- 基本情况 2: 没有剩余的子任务可以分配 (但还有智能体) ---
        # 这通常发生在智能体数量多于可用（非None）子任务数量时。
        # 注意：remaining_subtasks 在调用此函数前可能已被加入 None。
        if not remaining_subtasks:
            for agent in remaining_agents:
                new_subtask_alloc = base_subtask_alloc + [SubtaskAllocation(subtask=None, subtask_agent_names=tuple(agent))]
                other_subtask_allocs.append(new_subtask_alloc)
            return other_subtask_allocs

        # Otherwise assign remaining agents to remaining subtasks.
        # If only 1 agent left, assign to all remaining subtasks.
          # 情况 3: 只剩下一个智能体需要分配
        if len(remaining_agents) == 1:
             # 将这个智能体分配给 *每一个* 剩余的子任务 (包括 None)
            for t in remaining_subtasks:
                new_subtask_alloc = base_subtask_alloc + [SubtaskAllocation(subtask=t, subtask_agent_names=tuple(remaining_agents))]
                other_subtask_allocs.append(new_subtask_alloc)
            return other_subtask_allocs
        # If >1 agent remaining, create cooperative and divide & conquer
        # subtask allocations.
         # 情况 4: 剩下多于一个智能体需要分配
        else:
             # --- 可能性 A: 剩下的智能体 *合作* 完成同一个任务 ---
            # 遍历所有剩余的子任务 t
             # 创建一个新的分配：让 *所有* 剩余的智能体一起执行任务 t
            # Cooperative subtasks (same subtask assigned to remaining agents).
            for t in remaining_subtasks:
                new_subtask_alloc = base_subtask_alloc + [SubtaskAllocation(subtask=t, subtask_agent_names=tuple(remaining_agents))]
                other_subtask_allocs.append(new_subtask_alloc)
           
           
           # --- 可能性 B: 剩下的智能体 *分工* 完成不同的任务 (只考虑前两个剩余智能体) ---
            # 检查是否有足够多的剩余任务来进行分工（至少需要2个不同的任务）
            # Divide and Conquer subtasks (different subtask assigned to remaining agents).
            if len(remaining_subtasks) > 1:
                for ts in permutations(remaining_subtasks, 2):
                    new_subtask_alloc = base_subtask_alloc + [SubtaskAllocation(subtask=ts[0], subtask_agent_names=(remaining_agents[0], )),
                                                   SubtaskAllocation(subtask=ts[1], subtask_agent_names=(remaining_agents[1], )),]
                    other_subtask_allocs.append(new_subtask_alloc)
            return other_subtask_allocs







    def add_subtasks(self):
        """
        这个函数是为 "bd" (Bayesian Delegation)、"up" (Uniform Priors) 和 "fb" (Fixed Beliefs) 模型生成子任务分配假设空间的核心方法，它会考虑所有理论上可能的分工与合作方式
        包括单个智能体独立完成任务、多个智能体合作完成同一个任务、以及智能体处于空闲（None）状态的所有组合
        
        
        
        假设:
        3个智能体: self.all_agent_names = ['agent-1', 'agent-2', 'agent-3']
        未完成任务: self.incomplete_subtasks = [Chop(T), Chop(L)]
        
        函数执行流程:
        subtasks 列表是 [Chop(T), Chop(L)]。
        进入 else 分支 (多于1个智能体)。
        外层循环: 迭代智能体对。我们只看 first_agents = ('agent-1', 'agent-2') 这个组合
        （其他组合如 ('agent-1', 'agent-3'), ('agent-2', 'agent-3') 会类似地处理）。
        subtasks_temp 变为 [Chop(T), Chop(L), None, None]。
        
        内层循环 - 合作式:
        
            当 t = Chop(T):
                基础分配 base = [SubtaskAllocation(subtask=Chop(T), subtask_agent_names=('agent-1', 'agent-2'))] (A1和A2合作切T)
                remaining_agents = ['agent-3'], remaining_subtasks = [Chop(L), None, None]
                调用 get_other_subtask_allocations(remaining_agents=['agent-3'], remaining_subtasks=[Chop(L), None, None], base_subtask_alloc=base):
                因为只剩一个 agent-3，它会被分配到 remaining_subtasks 中的每一个。
                生成:
                alloc1_1 = [SubtaskAllocation(subtask=Chop(T), agents=(A1,A2)), SubtaskAllocation(subtask=Chop(L), agents=(A3,))]
                alloc1_2 = [SubtaskAllocation(subtask=Chop(T), agents=(A1,A2)), SubtaskAllocation(subtask=None,     agents=(A3,))] (因为有两个None, 这里会生成两个一样的，后续可能会被处理或视为一个)
            
            当 t = Chop(L): 类似地，A1和A2合作切L，A3做剩下的 (Chop(T) 或 None)。
            
            当 t = None: A1和A2合作啥也不干，A3做剩下的 (Chop(T) 或 Chop(L))。
        
        内层循环 - 分工式:
            当 ts = (Chop(T), Chop(L)):
                基础分配 base = [SubtaskAllocation(subtask=Chop(T), agents=(A1,)), SubtaskAllocation(subtask=Chop(L), agents=(A2,))] (A1切T, A2切L)
                remaining_agents = ['agent-3'], remaining_subtasks = [None, None]
                调用 get_other_subtask_allocations(...):
                只剩一个 agent-3，只能分配 None。
                生成: alloc2_1 = [SubtaskAllocation(subtask=Chop(T), agents=(A1,)), SubtaskAllocation(subtask=Chop(L), agents=(A2,)), SubtaskAllocation(subtask=None, agents=(A3,))]
            当 ts = (Chop(L), Chop(T)): 类似，A1切L, A2切T, A3闲置。
            当 ts = (Chop(T), None): A1切T, A2闲置, A3做剩下的 (Chop(L) 或 None)。
            当 ts = (None, Chop(T)): A1闲置, A2切T, A3做剩下的 (Chop(L) 或 None)。
            其他 ts 的组合。
                
        
        
        
        Return the entire distribution of subtask allocations."""
        subtask_allocs = []

        subtasks = self.incomplete_subtasks
      
      
        '''处理单智能体情况 (Base Case): 如果只有一个智能体 (len(self.all_agent_names) == 1)，逻辑很简单：对于每个未完成的子任务 t，直接创建一个分配，将该任务分配给这个唯一的智能体 
          Just one agent: Assign itself to all subtasks.'''
        if len(self.all_agent_names) == 1:
            for t in subtasks:
                subtask_alloc = [SubtaskAllocation(subtask=t, subtask_agent_names=tuple(self.all_agent_names))]

                subtask_allocs.append(subtask_alloc)
        else:
            
            
            for first_agents in combinations(self.all_agent_names, 2):
                # Temporarily add Nones, to allow agents to be allocated no subtask.
                # Later, we filter out allocations where all agents are assigned to None.
                subtasks_temp = subtasks + [None for _ in range(len(self.all_agent_names) - 1)]
               
               
                # Cooperative subtasks (same subtask assigned to agents).
                for t in subtasks_temp:
                    subtask_alloc = [SubtaskAllocation(subtask=t, subtask_agent_names=tuple(first_agents))]
                    remaining_agents = sorted(list(set(self.all_agent_names) - set(first_agents)))
                    remaining_subtasks = list(set(subtasks_temp) - set([t]))
                    subtask_allocs += self.get_other_subtask_allocations(
                            remaining_agents=remaining_agents,
                            remaining_subtasks=remaining_subtasks,
                            base_subtask_alloc=subtask_alloc)
                # Divide and Conquer subtasks (different subtask assigned to remaining agents).
                '''for ts in permutations(subtasks_temp, 2):: 遍历从 subtasks_temp 中选出的两个不同任务 ts[0] 和 ts[1] 的所有排列。
                subtask_alloc = [...]: 创建一个基础分配，  让 first_agents 中的第一个智能体执行 ts[0]，第二个智能体执行 ts[1]。 
                remaining_agents = ..., remaining_subtasks = ...: 确定剩余的智能体和任务。'''
                if len(subtasks_temp) > 1:
                    for ts in permutations(subtasks_temp, 2):
                        subtask_alloc = [
                                SubtaskAllocation(subtask=ts[0], subtask_agent_names=(first_agents[0],)),
                                SubtaskAllocation(subtask=ts[1], subtask_agent_names=(first_agents[1],)),]
                        remaining_agents = sorted(list(set(self.all_agent_names) - set(first_agents)))
                        remaining_subtasks = list(set(subtasks_temp) - set(ts))
                        subtask_allocs += self.get_other_subtask_allocations(
                                remaining_agents=remaining_agents,
                                remaining_subtasks=remaining_subtasks,
                                base_subtask_alloc=subtask_alloc)
        return SubtaskAllocDistribution(subtask_allocs)





    def add_greedy_subtasks(self):
        """Greedy"（贪婪）模型生成子任务分配的假设空间。贪婪智能体只考虑自己（self.agent_name）能执行的任务，
        完全不考虑其他智能体可能在做什么或与他们合作。
        
        假设当前智能体是 agent-1，有两个智能体 (self.all_agent_names = ['agent-1', 'agent-2'])。
        未完成的子任务是 self.incomplete_subtasks = [Chop(T), Chop(L)] (切番茄，切生菜)。
        函数执行流程：
        subtasks 列表变为 [Chop(T), Chop(L), None] (因为 None 不在初始列表中，所以被添加进去)。
        遍历 subtasks:
        当 subtask 是 Chop(T) 时，创建 alloc1 = [SubtaskAllocation(subtask=Chop(T), subtask_agent_names=('agent-1',))]，将其加入 subtask_allocs。
        当 subtask 是 Chop(L) 时，创建 alloc2 = [SubtaskAllocation(subtask=Chop(L), subtask_agent_names=('agent-1',))]，将其加入 subtask_allocs。
        当 subtask 是 None 时，创建 alloc3 = [SubtaskAllocation(subtask=None, subtask_agent_names=('agent-1',))]，将其加入 subtask_allocs。
        返回 SubtaskAllocDistribution([alloc1, alloc2, alloc3])。这个分布表示 agent-1 的信念：我要么去切番茄，要么去切生菜，要么什么都不干，完全不考虑 agent-2

        Return the entire distribution of greedy subtask allocations.
        i.e. subtasks performed only by agent with self.agent_name."""
        subtask_allocs = []

        subtasks = self.incomplete_subtasks
        # At least 1 agent must be doing something.
        if None not in subtasks:
            subtasks += [None]

        # Assign this agent to all subtasks. No joint subtasks because this function
        # only considers greedy subtask allocations.
        for subtask in subtasks:
            
            #对于每一个子任务 subtask，创建一个只包含一个元素的列表 subtask_alloc。
            # subtask_agent_names 字段 仅仅包含当前智能体自己的名字 (self.agent_name,)
            subtask_alloc = [SubtaskAllocation(subtask=subtask, subtask_agent_names=(self.agent_name,))]
            subtask_allocs.append(subtask_alloc)
        return SubtaskAllocDistribution(subtask_allocs)




    def add_dc_subtasks(self):
        """"Divide & Conquer" (D&C，分而治之) 模型生成子任务分配的假设空间。
        D&C 模型假设智能体之间会进行分工，每个智能体负责不同的任务，但绝不会合作完成同一个任务。
        
        使用 itertools.permutations 生成 subtasks 列表（包含实际任务和添加的 None 任务）中长度等于智能体数量 (len(self.all_agent_names)) 的所有排列 p。每个排列 p 代表了一种将任务（或 None）分配给智能体的方式。
        对于每个排列 p：
        创建一个 subtask_alloc 列表。
        对于排列中的第 i 个任务 p[i]，创建一个 SubtaskAllocation，将其分配给第 i 个智能体 self.all_agent_names[i] 
        （注意这里假设了智能体名称列表 self.all_agent_names 的顺序是固定的）。将这个分配添加到 subtask_alloc 列表中。
        最终生成的 subtask_alloc 会包含对每个智能体的任务分配，例如 [SubtaskAllocation(subtask=p[0], agents=(A1,)), SubtaskAllocation(subtask=p[1], agents=(A2,))]



        例子: 3个智能体的 Divide & Conquer (D&C) 分配

        假设:

        有3个智能体：self.all_agent_names = ['agent-1', 'agent-2', 'agent-3']。
        未完成的子任务仍然是：self.incomplete_subtasks = [Chop(T), Chop(L)] (切番茄，切生菜)。
        add_dc_subtasks(self) 函数执行流程:

        准备任务列表:

        需要添加 len(self.all_agent_names) - 1 = 3 - 1 = 2 个 None 任务。
        因此，用于排列的基础任务列表 subtasks 变为：[Chop(T), Chop(L), None, None]。
        生成排列:

        函数需要从 subtasks 列表中取出 len(self.all_agent_names) = 3 个元素进行排列。
        所以，它会计算 permutations([Chop(T), Chop(L), None, None], 3)。
        排列的总数是 P(4, 3) = 4! / (4-3)! = 24 种。
        为每个排列创建分配:

        对于每一个长度为3的排列 p，创建一个包含3个 SubtaskAllocation 的列表 subtask_alloc。
        排列 p 中的第 i 个元素 (p[i]) 被分配给第 i 个智能体 (self.all_agent_names[i])。
        几个排列 p 的例子及其对应的 subtask_alloc:

        例 1: p = (Chop(T), Chop(L), None)

        subtask_alloc = [
        SubtaskAllocation(subtask=Chop(T), subtask_agent_names=('agent-1',)), # agent-1 切番茄
        SubtaskAllocation(subtask=Chop(L), subtask_agent_names=('agent-2',)), # agent-2 切生菜
        SubtaskAllocation(subtask=None,     subtask_agent_names=('agent-3',))   # agent-3 啥也不干
        ]
        例 2: p = (Chop(L), Chop(T), None)

        subtask_alloc = [
        SubtaskAllocation(subtask=Chop(L), subtask_agent_names=('agent-1',)), # agent-1 切生菜
        SubtaskAllocation(subtask=Chop(T), subtask_agent_names=('agent-2',)), # agent-2 切番茄
        SubtaskAllocation(subtask=None,     subtask_agent_names=('agent-3',))   # agent-3 啥也不干
        ]

        (总共会生成 24 种这样的 subtask_alloc 列表)

        返回结果:
        函数最终返回 SubtaskAllocDistribution(subtask_allocs)，其中 subtask_allocs 包含了上面生成的全部 24 种分配列表。
        这个分布代表了 D&C 智能体的信念：我们三个人总是做不同的事情，可能会有一到两个人什么都不做，但绝不会有两个人同时做同一个任务。



        Return the entire distribution of divide & conquer subtask allocations.
        i.e. no subtask is shared between two agents.

        If there are no subtasks, just make an empty distribution and return."""
        subtask_allocs = []

        '''为了能够给所有智能体都分配任务（即使任务数量少于智能体数量），需要添加足够多的 None 任务。具体来说，添加 len(self.all_agent_names) - 1 个 None。
        这样确保了即使只有一个实际任务，其他智能体也可以被分配到 None'''
        subtasks = self.incomplete_subtasks + [None for _ in range(len(self.all_agent_names) - 1)]
       
        for p in permutations(subtasks, len(self.all_agent_names)):
            subtask_alloc = [SubtaskAllocation(subtask=p[i], subtask_agent_names=(self.all_agent_names[i],)) for i in range(len(self.all_agent_names))]
            subtask_allocs.append(subtask_alloc)
        return SubtaskAllocDistribution(subtask_allocs)








    def select_subtask(self, agent_name):
        """Return subtask and subtask_agent_names for agent with agent_name
        with max. probability."""
        max_subtask_alloc = self.probs.get_max()
        if max_subtask_alloc is not None:
            for t in max_subtask_alloc:
                if agent_name in t.subtask_agent_names:
                    return t.subtask, t.subtask_agent_names
        return None, agent_name

    def ensure_at_least_one_subtask(self):
        # Make sure each agent has None task by itself.
        if (self.model_type == "greedy" or self.model_type == "dc"):
            if not self.probs.probs:
                subtask_allocs = [[SubtaskAllocation(subtask=None, subtask_agent_names=(self.agent_name,))]]
                self.probs = SubtaskAllocDistribution(subtask_allocs)

    def bayes_update(self, obs_tm1, actions_tm1, beta):
        """Apply Bayesian update based on previous observation (obs_tms1)
        and most recent actions taken (actions_tm1). Beta is used to determine
        how rational agents act."""
        # First, remove unreachable/undoable subtask agent subtask_allocs.
        for subtask_alloc in self.probs.enumerate_subtask_allocs():
            for t in subtask_alloc:
                if not self.subtask_alloc_is_doable(
                        env=obs_tm1,
                        subtask=t.subtask,
                        subtask_agent_names=t.subtask_agent_names):
                    self.probs.delete(subtask_alloc)
                    break

        self.ensure_at_least_one_subtask()

        if self.model_type  == "fb":
            return

        for subtask_alloc in self.probs.enumerate_subtask_allocs():
            update = 0.0
            for t in subtask_alloc:
                if self.model_type == "greedy":
                    # Only calculate updates for yourself.
                    if self.agent_name in t.subtask_agent_names:
                        update += self.prob_nav_actions(
                                obs_tm1=copy.copy(obs_tm1),
                                actions_tm1=actions_tm1,
                                subtask=t.subtask,
                                subtask_agent_names=t.subtask_agent_names,
                                beta=beta,
                                no_level_1=False)
                else:
                    p = self.prob_nav_actions(
                            obs_tm1=copy.copy(obs_tm1),
                            actions_tm1=actions_tm1,
                            subtask=t.subtask,
                            subtask_agent_names=t.subtask_agent_names,
                            beta=beta,
                            no_level_1=False)
                    update += len(t.subtask_agent_names) * p

            self.probs.update(
                    subtask_alloc=subtask_alloc,
                    factor=update)
            print("UPDATING: subtask_alloc {} by {}".format(subtask_alloc, update))
        self.probs.normalize()
