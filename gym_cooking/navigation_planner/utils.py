from utils.core import *
import recipe_planner.utils as recipe
import recipe_planner.recipe
import numpy as np
import random

from queue import PriorityQueue

StringToGridSquare = {
        "Tomato"   : Counter,
        "Lettuce"  : Counter,
        "Onion"    : Counter,
        "Plate"    : Counter,
        "Cutboard" : Cutboard,
        "Delivery" : Delivery,
        }

StringToObject = {
        "Tomato"  : Tomato,
        "Lettuce" : Lettuce,
        "Onion"   : Onion,
        }


class MinPriorityQueue(PriorityQueue):
    """Used for min priority queue in BRTDP algorithm."""

    def __init__(self):
        PriorityQueue.__init__(self)
        self.counter = 0
    def put(self, item, priority):
        PriorityQueue.put(self, (priority, self.counter, item))
        self.counter += 1
    def get(self, *args, **kwargs):
        _, _, item = PriorityQueue.get(self, args, kwargs)
        return item

class Stack:
    def __init__(self):
        self.items = []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        return self.items.pop()
    def empty(self):
        return len(self) == 0
    def __len__(self):
        return len(self.items)

def is_smaller(p_, p):
    if type(p) is not tuple:
        return p_[0] < p
    else: return  p_ < p

def get_single_actions(env, agent):
    '''  
    计算给定智能体在当前环境状态下所有有效的单步动作 (包括移动和原地不动)。
    这是智能体动作空间 $A_i$ 的具体实现 [cite: 139]。'''
    actions = []
    
     # 获取环境中所有智能体的当前位置列表
    agent_locs = list(map(lambda a: a.location, env.sim_agents))

    # Check valid movement actions    检查四个基本移动方向: (0, 1)下, (0, -1)上, (-1, 0)左, (1, 0)右
    for t in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
        new_loc = env.world.inbounds(tuple(np.asarray(agent.location) + np.asarray(t)))
        
        # Check to make sure not at boundary
        # 条件1: 移动后的位置不能是其他智能体当前所在的位置 (防止移动到同一格)
        # 注意：这里没有检查目标位置是否会被其他智能体 *同时* 移动到，碰撞检测在 env.check_collisions 中处理
        if new_loc not in agent_locs:
            gs = env.world.get_gridsquare_at(new_loc)
            # Can move into floors
            if not gs.collidable:
                actions.append(t)  #  目标格子是 Floor (不可碰撞)，则可以移动过去
            # Can interact with deliveries
            elif isinstance(gs, Delivery):
                actions.append(t)  #   目标格子是 Delivery 点，可以移动过去进行交互 (递送)
            # Can interact with others if at least one of me or gs is holding something, or mergeable
            elif gs.holding is None and agent.holding is not None:
                actions.append(t) #  目标格子为空，而智能体拿着东西 (准备放下)
            elif gs.holding is not None and isinstance(gs.holding, Object) and agent.holding is None:
                actions.append(t)  #   - 目标格有东西，而智能体没拿东西 (准备拿起)
            elif gs.holding is not None and isinstance(gs.holding, Object) and\
                agent.holding is not None and mergeable(agent.holding, gs.holding):
                actions.append(t)    #    目标格子有东西，智能体也拿着东西，并且两者可以合并 (准备合并)
    # doing nothing is always possible
    actions.append((0, 0))

    return actions

def euclidean_dist(A, B):
    return np.linalg.norm(B - A)

def manhattan_dist(A, B):
    A_x, A_y = A
    B_x, B_y = B
    return float(abs(A_x - B_x) + abs(A_y - B_y))

def get_closest(obj_locations, source_loc):
    assert obj_locations, "Obj locations is probably empty? {}".format(obj_locations)

    closest_dist = np.inf
    for loc in obj_locations:
        dist = manhattan_dist(loc, source_loc)
        if dist < closest_dist:
            closest_dist = dist
    return closest_dist

def get_min_dist_between(A_locations, B_locations):
    min_dist = np.inf

    for A in A_locations:
        A_min = get_closest(B_locations, A)
        if A_min < min_dist: min_dist = A_min
    return min_dist


def get_obj(obj_string, type_, state, location=(None, None)):
    # Core.Supply objects
    if type_ == "is_supply":
         # 检查字符串是否在映射字典 StringToGridSquare 中
        if obj_string not in StringToGridSquare:
            raise NotImplementedError("{} is not recognized.".format(obj_string))
        return StringToGridSquare[obj_string](location)

    # Core.Object(Food / Plate) objects
    elif type_ == "is_object":
        if "-" in obj_string: # 返回最终的复合 Object
            obj_strs = obj_string.split("-")
            # just getting objects
            objects = [get_obj(obj_string=s,
                type_="is_object", state=FoodState.FRESH) for s in obj_strs]
            # getting into right food env
            for i, s in enumerate(obj_strs):
                if s == "Plate":  
                    continue
                objects[i] = get_obj(obj_string=s,
                        type_="is_object",
                        state=objects[i].contents[0].state_seq[-1])
            o = Object(location, objects[0].contents[0])
            for obj in objects[1:]:
                o.merge(obj.contents[0])
            return o
        elif obj_string == "Plate":  #  创建只包含盘子的 Object
            return Object(location, Plate())
        elif obj_string in StringToObject: # 创建只包含单个 Food 的 Object
            obj = StringToObject[obj_string]()
            obj.set_state(state)
            return Object(location, obj)
        else:
            raise NotImplementedError("Type {} is not recognized".format(type_))

def get_subtask_action_obj(subtask):
    """
    根据给定的子任务 (Action 实例)，返回执行该子任务需要交互的 **静态** GridSquare 对象。
    例如，Chop 需要 Cutboard，Deliver 需要 Delivery。

    Args:
        subtask (Action or None): 子任务对象。

    Returns:
       Counter / Cutboard / Delivery 对应的静态交互对象，如果不需要则返回 None。
    """
    if isinstance(subtask, recipe.Get):
        obj = get_obj(obj_string=subtask.args[0], type_="is_supply", state=None)
    elif isinstance(subtask, recipe.Chop):
        obj = get_obj(obj_string="Cutboard", type_="is_supply", state=None)
    elif isinstance(subtask, recipe.Deliver):
        obj = get_obj(obj_string="Delivery", type_="is_supply", state=None)
    elif isinstance(subtask, recipe.Merge):
        obj = None
    elif subtask is None:
        obj = None
    else:
        raise ValueError("Did not recognize subtask {} so could not find the appropriate subtask location".format(subtask))
    return obj



def get_subtask_obj(subtask):
    """
    根据给定的子任务 (Action 实例)，返回该子任务涉及的 **起始状态** 和 **目标状态** 的 Object 实例。
    这对于定义任务完成条件和规划启发式非常重要 [cite: 146, 148]。

    Args:
        subtask (Action or None): 子任务对象。

    Returns:
        tuple: (start_obj, goal_obj)
               - start_obj: 子任务开始时需要的物品对象或对象列表 (对于 Merge)。
               - goal_obj: 子任务完成后期望得到的物品对象。
               如果 subtask 为 None，则返回 (None, None)。
    """
    
    ''' # --- Chop 任务 (e.g., Chop('Tomato')) ---'''
    if isinstance(subtask, recipe.Chop):
        # 起始: 新鲜的物品 (e.g., FreshTomato)
        start_obj = get_obj(obj_string=subtask.args[0],
                type_="is_object", state=FoodState.FRESH)
        # 目标: 切好的物品 (e.g., ChoppedTomato)
        goal_obj = get_obj(obj_string=subtask.args[0],
                type_="is_object", state=FoodState.CHOPPED)

  # --- Merge 任务 (e.g., Merge('Tomato', 'Plate')) ---
    elif isinstance(subtask, recipe.Merge):
        # only need in last state
        obj1 = get_obj(obj_string=subtask.args[0],
                type_="is_object", state=FoodState.FRESH)
        obj2 = get_obj(obj_string=subtask.args[1],
                type_="is_object", state=FoodState.FRESH)
        object_list = [obj1, obj2]
        start_obj = [] # initial state before merging
        # expected objects in their last food state

        for i, o in enumerate(object_list):
            if isinstance(o.contents[0], Plate):
                start_obj.append(copy.copy(o))
                continue

            object_list[i] = get_obj(obj_string=subtask.args[i],
                    type_="is_object", state=o.contents[0].state_seq[-1])
            # Must be in the last state before merging
            start_obj.append(get_obj(obj_string=subtask.args[i],
                type_="is_object", state=o.contents[0].state_seq[-1]))

        # Merging objects
        object_list[0].merge(object_list[1])
        goal_obj = object_list[0]

    elif isinstance(subtask, recipe.Deliver):
        start_obj = get_obj(obj_string=subtask.args[0],
                type_="is_object", state=FoodState.FRESH)
        # Correct the state.
        state = start_obj.contents[0].state_seq[-1] if not isinstance(start_obj.contents[0], Plate) else start_obj.contents[1].state_seq[-1]
        start_obj = get_obj(obj_string=subtask.args[0],
                type_="is_object", state=state)
        goal_obj = copy.copy(start_obj)

    elif subtask is None:
        return None, None

    else:
        raise NotImplementedError("{} was not recognized".format(subtask))

    return start_obj, goal_obj

