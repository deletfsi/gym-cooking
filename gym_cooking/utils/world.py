import numpy as np
from collections import defaultdict, OrderedDict
from itertools import product, combinations
import networkx as nx
import copy
import matplotlib.pyplot as plt
from functools import lru_cache

import recipe_planner.utils as recipe
from navigation_planner.utils import manhattan_dist
from utils.core import Object, GridSquare, Counter


class World:
    """World class that hold all of the non-agent objects in the environment."""
    NAV_ACTIONS = [(0, 1), (0, -1), (-1, 0), (1, 0)]

    def __init__(self, arglist):
        self.rep = [] # [row0, row1, ..., rown]
        self.arglist = arglist
        self.objects = defaultdict(lambda : [])

    def get_repr(self):
        return self.get_dynamic_objects()

    def __str__(self):
        # 示例输出:
        # - - t - -
        # /     l
        # * -
        _display = list(map(lambda x: ''.join(map(lambda y: y + ' ', x)), self.rep))
        return '\n'.join(_display)

    def __copy__(self):
        new = World(self.arglist)
        new.__dict__ = self.__dict__.copy()
        new.objects = copy.deepcopy(self.objects)
        new.reachability_graph = self.reachability_graph
        new.distances = self.distances
        return new

    def update_display(self):
        """更新用于打印的二维字符表示 (self.rep)。"""
        # Reset the current display (self.rep).
        self.rep = [[' ' for i in range(self.width)] for j in range(self.height)]
        objs = []
        for o in self.objects.values():
            objs += o
        for obj in objs:
            self.add_object(obj, obj.location)
        for obj in self.objects["Tomato"]:
            self.add_object(obj, obj.location)
        return self.rep

    def print_objects(self):
        """打印当前世界中所有对象的名称及其位置列表 (用于调试)。"""
        for k, v in self.objects.items():
            print(k, list(map(lambda o: o.location, v)))

    def make_loc_to_gridsquare(self):
        """Creates a mapping between object location and object."""
        self.loc_to_gridsquare = {}
        for obj in self.get_object_list():
            if isinstance(obj, GridSquare):
                self.loc_to_gridsquare[obj.location] = obj

    def make_reachability_graph(self):
        """
        创建并存储一个基于 NetworkX 的可达性图 (reachability_graph)。
        这个图用于导航规划 (如 BRTDP) 中计算两点间的最短路径。
        图的节点不是简单的位置 (x, y)，而是 (location, approach_direction) 对。
        - 对于非碰撞格子 (Floor): 节点是 ((x, y), (0, 0))，表示智能体可以处于该格子的中心。
        - 对于碰撞格子 (Counter, Cutboard, Delivery): 节点是 ((x, y), (dx, dy))，其中 (dx, dy) 是接近该格子的方向 (来自相邻的 Floor)。例如 ((3, 4), (0, -1)) 表示从下方接近 (3, 4) 处的柜台。
        图的边表示可以在相邻的节点状态之间移动一步。
        [cite: 440]
        """
        """Create a reachability graph between world objects."""
        self.reachability_graph = nx.Graph()
        for x in range(self.width):
            for y in range(self.height):
                location = (x, y)
                gs = self.loc_to_gridsquare[(x, y)]

                # If not collidable, add node with direction (0, 0).
                if not gs.collidable:
                    self.reachability_graph.add_node((location, (0, 0)))

                # Add nodes for collidable gs + all edges.
                for nav_action in World.NAV_ACTIONS:
                    new_location = self.inbounds(location=tuple(np.asarray(location) + np.asarray(nav_action)))
                    new_gs = self.loc_to_gridsquare[new_location]

                    # If collidable, add edges for adjacent noncollidables.
                    if gs.collidable and not new_gs.collidable:
                        self.reachability_graph.add_node((location, nav_action))
                        if (new_location, (0, 0)) in self.reachability_graph:
                            self.reachability_graph.add_edge((location, nav_action),
                                                             (new_location, (0, 0)))
                    # If not collidable and new_gs collidable, add edge.
                    elif not gs.collidable and new_gs.collidable:
                        if (new_location, tuple(-np.asarray(nav_action))) in self.reachability_graph:
                            self.reachability_graph.add_edge((location, (0, 0)),
                                                             (new_location, tuple(-np.asarray(nav_action))))
                    # If both not collidable, add direct edge.
                    elif not gs.collidable and not new_gs.collidable:
                        if (new_location, (0, 0)) in self.reachability_graph:
                            self.reachability_graph.add_edge((location, (0, 0)), (new_location, (0, 0)))
                    # If both collidable, add nothing.

        # If you want to visualize this graph, uncomment below.
        # plt.figure()
        # nx.draw(self.reachability_graph)
        # plt.show()

    def get_lower_bound_between(self, subtask, agent_locs, A_locs, B_locs):
        """
        计算完成子任务所需的距离下界。
        它会考虑智能体的当前位置 (agent_locs)、起始物品/位置集合 (A_locs) 和
        目标物品/位置集合 (B_locs)，找出所有 A-B 对之间的最短距离下界中的最小值。
        这个下界值主要用作导航规划中的启发式信息 [cite: 452]。

        Args:
            subtask (Action): 当前考虑的子任务 (如 Chop, Deliver, Merge)。
            agent_locs (tuple): 一个或两个智能体的当前位置元组。
            A_locs (tuple): 起始物品/位置的可能位置元组。
            B_locs (tuple): 目标物品/位置的可能位置元组。

        Returns:
            float: 完成该子任务交互的最短路径距离下界。
        """
        """Return distance lower bound between subtask-relevant locations."""
        # 初始化下界为一个较大值 (地图周长+1)
        lower_bound = self.perimeter + 1
        
        # 遍历所有可能的 A 位置和 B 位置的组合 (笛卡尔积)
        for A_loc, B_loc in product(A_locs, B_locs):
            bound = self.get_lower_bound_between_helper(
                    subtask=subtask,
                    agent_locs=agent_locs,
                    A_loc=A_loc,
                    B_loc=B_loc)
            if bound < lower_bound:
                lower_bound = bound
        return lower_bound



    @lru_cache(maxsize=40000)
    def get_lower_bound_between_helper(self, subtask, agent_locs, A_loc, B_loc):
        """
        计算从智能体当前位置出发，完成与特定 A_loc 和 B_loc 交互所需的距离下界。
        利用预先计算好的 reachability_graph。使用了 lru_cache 来加速重复查询。

        Args:
            subtask (Action): 子任务类型。
            agent_locs (tuple): 智能体位置元组。
            A_loc (tuple): 起始物品/位置。
            B_loc (tuple): 目标物品/位置。

        Returns:
            float: 距离下界。
        """
        lower_bound = self.perimeter + 1
        A = self.get_gridsquare_at(A_loc)
        B = self.get_gridsquare_at(B_loc)
        
        
        # 确定接近 A 和 B 的可能方式 (节点类型)
        # 如果 A 是 Floor，只能从中心 ((A_loc, (0,0))) 接近
        # 如果 A 是障碍物，可以从四个相邻方向接近 ((A_loc, nav_action)
        A_possible_na = [(0, 0)] if not A.collidable else World.NAV_ACTIONS
        B_possible_na = [(0, 0)] if not B.collidable else World.NAV_ACTIONS

        # 遍历所有可能的接近 A 和 B 的方式组合
        for A_na, B_na in product(A_possible_na, B_possible_na):
            
             # --- 单智能体情况 ---
            if len(agent_locs) == 1:
                try:
                    # 计算 Agent -> A 的最短路径长度
                    bound_1 = nx.shortest_path_length(
                            self.reachability_graph, (agent_locs[0], (0, 0)), (A_loc, A_na))
                    
                    # 计算 A -> B 的最短路径长度
                    bound_2 = nx.shortest_path_length(
                            self.reachability_graph, (A_loc, A_na), (B_loc, B_na))
                except:
                    continue
                bound = bound_1 + bound_2 - 1


          # --- 双智能体情况 ---
            elif len(agent_locs) == 2:
                # Try to calculate the distances between agents and Objects A and B.
                # Distance between Agent 1 <> Object A.
                 # 计算 Agent 1 -> A 和 Agent 2 -> A 的距离
                try:
                    bound_1_to_A = nx.shortest_path_length(
                            self.reachability_graph, (agent_locs[0], (0, 0)), (A_loc, A_na))
                except:
                    bound_1_to_A = self.perimeter
                # Distance between Agent 2 <> Object A.
                try:
                    bound_2_to_A = nx.shortest_path_length(
                            self.reachability_graph, (agent_locs[1], (0, 0)), (A_loc, A_na))
                except:
                    bound_2_to_A = self.perimeter

                # Take the agent that's the closest to Object A.
                 # 找到离 A 最近的智能体的距离
                min_bound_to_A = min(bound_1_to_A, bound_2_to_A)

                # Distance between the agents.
                 # 计算 A 和 B 之间的曼哈顿距离 (作为近似)
                bound_between_agents = manhattan_dist(A_loc, B_loc)





                # Distance between Agent 1 <> Object B.
                # 计算 Agent 1 -> B 和 Agent 2 -> B 的距离
                try:
                    bound_1_to_B = nx.shortest_path_length(self.reachability_graph, (agent_locs[0], (0, 0)), (B_loc, B_na))
                except:
                    bound_1_to_B = self.perimeter

                # Distance between Agent 2 <> Object B.
                  # 找到离 B 最近的智能体的距离
                try:
                    bound_2_to_B = nx.shortest_path_length(self.reachability_graph, (agent_locs[1], (0, 0)), (B_loc, B_na))
                except:
                    bound_2_to_B = self.perimeter

                # Take the agent that's the closest to Object B.
                min_bound_to_B = min(bound_1_to_B, bound_2_to_B)


                # For chop or deliver, must bring A to B.
                # --- 根据子任务类型计算总距离下界 ---
                # 对于 Chop 或 Deliver，需要一个智能体把 A 带到 B
                if isinstance(subtask, recipe.Chop) or isinstance(subtask, recipe.Deliver):
                    '''# 距离 = (最近智能体到 A 的距离) + (A 到 B 的距离)'''
                    bound = min_bound_to_A + bound_between_agents - 1
                # For merge, agents can separately go to A and B and then meet in the middle.
                # 检查是否是同一个智能体离 A 和 B 都最近，如果是，调整距离计算方式 (通过 check_bound)'''
                # 距离 = max(到A的调整距离, 到B的调整距离) + (A到B距离的一半)
                # 假设两个智能体同时出发，距离取决于走得慢的那个，并在 A B 连线中点附近汇合
                elif isinstance(subtask, recipe.Merge):
                    min_bound_to_A, min_bound_to_B = self.check_bound(
                            min_bound_to_A=min_bound_to_A,
                            min_bound_to_B=min_bound_to_B,
                            bound_1_to_A=bound_1_to_A,
                            bound_2_to_A=bound_2_to_A,
                            bound_1_to_B=bound_1_to_B,
                            bound_2_to_B=bound_2_to_B
                            )
                    bound = max(min_bound_to_A, min_bound_to_B) + (bound_between_agents - 1)/2

            if bound < lower_bound:
                lower_bound = bound

 # 返回最终找到的最小下界 (至少为 1)
        return max(1, lower_bound)

    def check_bound(self, min_bound_to_A, min_bound_to_B,
                            bound_1_to_A, bound_2_to_A,
                            bound_1_to_B, bound_2_to_B):
        """
        辅助函数，用于双智能体 Merge 任务的距离计算。
        检查是否是同一个智能体同时离 A 和 B 最近。
        如果是，将对应的最小距离乘以 2，模拟该智能体需要依次访问 A 和 B 的情况。
        """
        # Checking for whether it's the same agent that does the subtask.
        # 检查是否 Agent 1 同时是离 A 最近且离 B 最近
        # 或者 Agent 2 同时是离 A 最近且离 B 最近
        if ((bound_1_to_A == min_bound_to_A and bound_1_to_B == min_bound_to_B) or
            (bound_2_to_A == min_bound_to_A and bound_2_to_B == min_bound_to_B)):
            # 如果是同一个智能体，返回调整后的距离 (相当于该智能体需要跑两段路)
            return 2*min_bound_to_A, 2*min_bound_to_B
        
         # 如果是不同智能体分别离 A 和 B 最近，则直接返回原始最小距离
        return min_bound_to_A, min_bound_to_B

    def is_occupied(self, location):
        o = list(filter(lambda obj: obj.location == location and
         isinstance(obj, Object) and not(obj.is_held), self.get_object_list()))
        if o:
            return True
        return False

    def clear_object(self, position):
        """Clears object @ position in self.rep and replaces it with an empty space"""
        x, y = position
        self.rep[y][x] = ' '

    def clear_all(self):
        self.rep = []

    def add_object(self, object_, position):
        x, y = position
        self.rep[y][x] = str(object_)

    def insert(self, obj):
        self.objects.setdefault(obj.name, []).append(obj)

    def remove(self, obj):
        num_objs = len(self.objects[obj.name])
        index = None
        for i in range(num_objs):
            if self.objects[obj.name][i].location == obj.location:
                index = i
        assert index is not None, "Could not find {}!".format(obj.name)
        self.objects[obj.name].pop(index)
        assert len(self.objects[obj.name]) < num_objs, "Nothing from {} was removed from world.objects".format(obj.name)

    def get_object_list(self):
        all_obs = []
        for o in self.objects.values():
            all_obs += o
        return all_obs

    def get_dynamic_objects(self):
        """
        获取所有动态对象 (可移动物品，如 Food, Plate, 组合物品) 的表示。
        返回一个元组的元组，用于状态哈希或比较。
        格式: ((ObjectRepr1_typeA, ObjectRepr2_typeA, ...), (ObjectRepr1_typeB, ...), ...)
        """
        """Get objects that can be moved."""
        objs = list()

        for key in sorted(self.objects.keys()):
            if key != "Counter" and key != "Floor" and "Supply" not in key and key != "Delivery" and key != "Cutboard":
                objs.append(tuple(list(map(lambda o: o.get_repr(), self.objects[key]))))

        # Must return a tuple because this is going to get hashed.
        return tuple(objs)

    def get_collidable_objects(self):
        """返回世界中所有可碰撞对象 (collidable=True) 的列表。"""
        return list(filter(lambda o : o.collidable, self.get_object_list()))

    def get_collidable_object_locations(self):
        """返回世界中所有可碰撞对象的位置列表。"""
        return list(map(lambda o: o.location, self.get_collidable_objects()))

    def get_dynamic_object_locations(self):
        return list(map(lambda o: o.location, self.get_dynamic_objects()))

    def is_collidable(self, location):
        """检查指定位置是否包含一个可碰撞的对象。"""
        return location in list(map(lambda o: o.location, list(filter(lambda o: o.collidable, self.get_object_list()))))

    def get_object_locs(self, obj, is_held):
        """
        获取特定类型对象 (obj) 在特定持有状态 (is_held) 下的所有位置。

        Args:
            obj (Object or GridSquare): 要查找的对象模板 (基于名称和类型进行比较)。
            is_held (bool): 指定查找被持有 (True) 还是未被持有 (False) 的 Object。
                           对于 GridSquare，此参数无效。

        Returns:
            list: 包含所有匹配对象位置的列表。
        """
        if obj.name not in self.objects.keys():
            return []

        if isinstance(obj, Object):
            return list(
                    map(lambda o: o.location, list(filter(lambda o: obj == o and
                    o.is_held == is_held, self.objects[obj.name]))))
        else:
            return list(map(lambda o: o.location, list(filter(lambda o: obj == o,
                self.objects[obj.name]))))

    def get_all_object_locs(self, obj):
        """
        获取特定类型对象 (obj) 的所有位置，无论其是否被持有。

        Args:
            obj (Object or GridSquare): 要查找的对象模板。

        Returns:
            list: 包含所有匹配对象位置的列表 (去重)。
        """
        return list(set(self.get_object_locs(obj=obj, is_held=True) + self.get_object_locs(obj=obj, is_held=False)))

    def get_object_at(self, location, desired_obj, find_held_objects):
        """
        获取指定位置的特定类型的 Object 实例。

        Args:
            location (tuple): 要查询的位置 (x, y)。
            desired_obj (Object or None): 希望找到的对象类型模板。如果为 None，则返回该位置任意类型的 Object。
            find_held_objects (bool): 是否查找被持有的 Object (True) 或未被持有的 (False)。

        Returns:
            Object: 找到的 Object 实例。
        """
        # Map obj => location => filter by location => return that object.
        all_objs = self.get_object_list()
 # 如果不指定具体类型 (desired_obj is None)
        if desired_obj is None:
             # 筛选出位置匹配、是 Object 类型、且 is_held 状态匹配的对象
            objs = list(filter(
                lambda obj: obj.location == location and isinstance(obj, Object) and obj.is_held is find_held_objects,
                all_objs))
            # 如果指定了具体类型
        else:
             # 筛选出名称匹配、位置匹配、是 Object 类型、且 is_held 状态匹配的对象
            # 注意：这里比较的是 obj.name == desired_obj.name，而不是直接比较对象 obj == desired_obj
            objs = list(filter(lambda obj: obj.name == desired_obj.name and obj.location == location and
                isinstance(obj, Object) and obj.is_held is find_held_objects,
                all_objs))

        assert len(objs) == 1, "looking for {}, found {} at {}".format(desired_obj, ','.join(o.get_name() for o in objs), location)

        return objs[0]

    def get_gridsquare_at(self, location):
        """
        获取指定位置的 GridSquare 对象。
        (内部使用了 self.loc_to_gridsquare 映射，但这里是重新查找实现)。
        """
        gss = list(filter(lambda o: o.location == location and\
            isinstance(o, GridSquare), self.get_object_list()))

        assert len(gss) == 1, "{} gridsquares at {}: {}".format(len(gss), location, gss)
        return gss[0]

    def inbounds(self, location):
         # 使用 min 和 max 确保 x 在 [0, width-1] 之间，y 在 [0, height-1] 之间
        """将给定位置修正到世界边界内。 Correct locaiton to be in bounds of world object."""
        x, y = location
        return min(max(x, 0), self.width-1), min(max(y, 0), self.height-1)
