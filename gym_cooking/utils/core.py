# recipe planning
import recipe_planner.utils as recipe

# helpers
import numpy as np
import copy
import random
from termcolor import colored as color
from itertools import combinations
from collections import namedtuple


# -----------------------------------------------------------
# GRIDSQUARES
# -----------------------------------------------------------
GridSquareRepr = namedtuple("GridSquareRepr", "name location holding")

class Rep:
    FLOOR = ' '
    COUNTER = '-'
    CUTBOARD = '/'
    DELIVERY = '*'
    TOMATO = 't'
    LETTUCE = 'l'
    ONION = 'o'
    PLATE = 'p'

class GridSquare:
    def __init__(self, name, location):
        self.name = name
        self.location = location   # (x, y) tuple
        self.holding = None
        self.color = 'white'
        self.collidable = True     # cannot go through
        self.dynamic = False       # cannot move around

    def __str__(self):
        return color(self.rep, self.color)

    def __eq__(self, o):
        return isinstance(o, GridSquare) and self.name == o.name

    def __copy__(self):
        gs = type(self)(self.location)
        gs.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            gs.holding = copy.copy(self.holding)
        return gs

    def acquire(self, obj):
        obj.location = self.location
        self.holding = obj

    def release(self):
        temp = self.holding
        self.holding = None
        return temp

class Floor(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self,"Floor", location)
        self.color = None
        self.rep = Rep.FLOOR
        self.collidable = False
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

class Counter(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self,"Counter", location)
        self.rep = Rep.COUNTER
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

class AgentCounter(Counter):
    """
    一种特殊的 Counter，用于在 Level-0 规划时临时替换其他智能体的位置。
    它表示一个被“冻结”的智能体占据的位置，是可碰撞的障碍物。
    """
    def __init__(self, location):
        GridSquare.__init__(self,"Agent-Counter", location)
        self.rep = Rep.COUNTER
        self.collidable = True
    def __eq__(self, other):
        return Counter.__eq__(self, other)
    def __hash__(self):
        return Counter.__hash__(self)
    def get_repr(self):
        return GridSquareRepr(name=self.name, location=self.location, holding= None)

class Cutboard(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "Cutboard", location)
        self.rep = Rep.CUTBOARD
        self.collidable = True
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)

class Delivery(GridSquare):
    def __init__(self, location):
        GridSquare.__init__(self, "Delivery", location)
        self.rep = Rep.DELIVERY
        self.holding = []
    def acquire(self, obj):
        obj.location = self.location
        self.holding.append(obj)
    def release(self):
        if self.holding:
            return self.holding.pop()
        else: return None
    def __eq__(self, other):
        return GridSquare.__eq__(self, other)
    def __hash__(self):
        return GridSquare.__hash__(self)


# -----------------------------------------------------------
# OBJECTS
# -----------------------------------------------------------
# Objects are wrappers around foods items, plates, and any combination of them

ObjectRepr = namedtuple("ObjectRepr", "name location is_held")

class Object:
    def __init__(self, location, contents):
        self.location = location
        self.contents = contents if isinstance(contents, list) else [contents]
        self.is_held = False
        self.update_names()
        self.collidable = False
        self.dynamic = False

    def __str__(self):
        res = "-".join(list(map(lambda x : str(x), sorted(self.contents, key=lambda i: i.name))))
        return res

    def __eq__(self, other):
        # check that content is the same and in the same state(s)
        return isinstance(other, Object) and \
                self.name == other.name and \
                len(self.contents) == len(other.contents) and \
                self.full_name == other.full_name
                # all([i == j for i, j in zip(sorted(self.contents, key=lambda x: x.name),
                #                             sorted(other.contents, key=lambda x: x.name))])

    def __copy__(self):
        new = Object(self.location, self.contents[0])
        new.__dict__ = self.__dict__.copy()
        new.contents = [copy.copy(c) for c in self.contents]
        return new

    def get_repr(self):
        return ObjectRepr(name=self.full_name, location=self.location, is_held=self.is_held)

    def update_names(self):
        """
        根据当前 contents 更新 self.name 和 self.full_name。
        名称由内部物品按字母顺序排序后用 '-' 连接。
        self.name: 只包含基础物品名称 (e.g., "Lettuce-Plate-Tomato")
        self.full_name: 包含物品状态 (e.g., "ChoppedLettuce-Plate-FreshTomato")
        """
        # concatenate names of alphabetically sorted items, e.g.
        sorted_contents = sorted(self.contents, key=lambda c: c.name)
        self.name = "-".join([c.name for c in sorted_contents])
        self.full_name = "-".join([c.full_name for c in sorted_contents])

    def contains(self, c_name):
        return c_name in list(map(lambda c : c.name, self.contents))

    def needs_chopped(self):
        if len(self.contents) > 1: return False
        return self.contents[0].needs_chopped()

    def is_chopped(self):
        # 如果包含盘子，或者任何一个 Food 没到 Chopped 状态，则返回 False
        for c in self.contents:
            if isinstance(c, Plate) or c.get_state() != 'Chopped':
                return False
        return True

    def chop(self):
        assert len(self.contents) == 1
        assert self.needs_chopped()
        self.contents[0].update_state()
        assert not (self.needs_chopped())
        self.update_names()

    def merge(self, obj):
        if isinstance(obj, Object):
            # move obj's contents into this instance
            for i in obj.contents: self.contents.append(i)
        elif not (isinstance(obj, Food) or isinstance(obj, Plate)):
            raise ValueError("Incorrect merge object: {}".format(obj))
        else:
            self.contents.append(obj)
        self.update_names()

    def unmerge(self, full_name):
        # remove by full_name, assumming all unique contents
        matching = list(filter(lambda c: c.full_name == full_name, self.contents))
        self.contents.remove(matching[0])
        self.update_names()
        return matching[0]

    def is_merged(self):
        return len(self.contents) > 1

    def is_deliverable(self):
        # must be merged, and all contents must be Plates or Foods in done state
        for c in self.contents:
            if not (isinstance(c, Plate) or (isinstance(c, Food) and c.done())):
                return False
        return self.is_merged()


def mergeable(obj1, obj2):
    # 条件1: 最多只能有一个盘子
    # 条件2: 所有非 Plate 的内容 (即 Food) 必须处于其最终状态 (done)
    # query whether two objects are mergeable
    contents = obj1.contents + obj2.contents
    # check that there is at most one plate
    try:
        contents.remove(Plate())
    except:
        pass  # do nothing, 1 plate is ok
    finally:
        try:
            contents.remove(Plate())
        except:
            for c in contents:   # everything else must be in last state
                if not c.done():
                    return False
        else:
            return False  # more than 1 plate
    return True


# -----------------------------------------------------------

class FoodState:
    FRESH = globals()['recipe'].__dict__['Fresh']
    CHOPPED = globals()['recipe'].__dict__['Chopped']

class FoodSequence:
    FRESH = [FoodState.FRESH]
    FRESH_CHOPPED = [FoodState.FRESH, FoodState.CHOPPED]

class Food:
    def __init__(self):
        self.state = self.state_seq[self.state_index]
        self.movable = False
        self.color = self._set_color()
        self.update_names()

    def __str__(self):
        return color(self.rep, self.color)

    # def __hash__(self):
    #     return hash((self.state, self.name))

    def __eq__(self, other):
        return isinstance(other, Food) and self.get_state() == other.get_state()

    def __len__(self):
        return 1   # one food unit

    def set_state(self, state):
        assert state in self.state_seq, "Desired state {} does not exist for the food with sequence {}".format(state, self.state_seq)
        self.state_index = self.state_seq.index(state)
        self.state = state
        self.update_names()

    def get_state(self):
        return self.state.__name__

    def update_names(self):
        self.full_name = '{}{}'.format(self.get_state(), self.name)

    def needs_chopped(self):
        """检查食物是否需要被切 (即下一个状态是否是 Chopped)。"""
        # 使用取模运算 (%) 来处理状态序列循环 (虽然当前序列不循环)
        # 检查 state_seq 中当前索引的下一个状态是否是 FoodState.CHOPPED
        # 例如: 当前是 FRESH (index 0), state_seq[1] 是 CHOPPED -> True
        # 例如: 当前是 CHOPPED (index 1), state_seq[0] 是 FRESH -> False
        return self.state_seq[(self.state_index+1)%len(self.state_seq)] == FoodState.CHOPPED

    def done(self):
        """检查食物是否处于其最终状态。"""
        # 检查当前状态索引是否是 state_seq 的最后一个索引
        return (self.state_index % len(self.state_seq)) == len(self.state_seq) - 1
        # 例如: 对 FRESH_CHOPPED 序列:
        # FRESH (index 0): 0 == 1 -> False
        # CHOPPED (index 1): 1 == 1 -> True

    def update_state(self):
        self.state_index += 1
        assert 0 <= self.state_index and self.state_index < len(self.state_seq), "State index is out of bounds for its state sequence"
        self.state = self.state_seq[self.state_index]
        self.update_names()

    def _set_color(self):
        pass

class Tomato(Food):
    def __init__(self, state_index = 0):
        self.state_index = state_index   # index in food's state sequence
        self.state_seq = FoodSequence.FRESH_CHOPPED
        self.rep = 't'
        self.name = 'Tomato'
        Food.__init__(self)
    def __hash__(self):
        return Food.__hash__(self)
    def __eq__(self, other):
        return Food.__eq__(self, other)
    def __str__(self):
        return Food.__str__(self)

class Lettuce(Food):
    def __init__(self, state_index = 0):
        self.state_index = state_index   # index in food's state sequence
        self.state_seq = FoodSequence.FRESH_CHOPPED
        self.rep = 'l'
        self.name = 'Lettuce'
        Food.__init__(self)
    def __eq__(self, other):
        return Food.__eq__(self, other)
    def __hash__(self):
        return Food.__hash__(self)

class Onion(Food):
    def __init__(self, state_index = 0):
        self.state_index = state_index   # index in food's state sequence
        self.state_seq = FoodSequence.FRESH_CHOPPED
        self.rep = 'o'
        self.name = 'Onion'
        Food.__init__(self)
    def __eq__(self, other):
        return Food.__eq__(self, other)
    def __hash__(self):
        return Food.__hash__(self)


# -----------------------------------------------------------

class Plate:
    def __init__(self):
        self.rep = "p"
        self.name = 'Plate'
        self.full_name = 'Plate'
        self.color = 'white'
    def __hash__(self):
        return hash((self.name))
    def __str__(self):
        return color(self.rep, self.color)
    def __eq__(self, other):
        return isinstance(other, Plate)
    def __copy__(self):
        return Plate()
    def needs_chopped(self):
        return False


# -----------------------------------------------------------
# PARSING
# -----------------------------------------------------------
RepToClass = {
    Rep.FLOOR: globals()['Floor'],
    Rep.COUNTER: globals()['Counter'],
    Rep.CUTBOARD: globals()['Cutboard'],
    Rep.DELIVERY: globals()['Delivery'],
    Rep.TOMATO: globals()['Tomato'],
    Rep.LETTUCE: globals()['Lettuce'],
    Rep.ONION: globals()['Onion'],
    Rep.PLATE: globals()['Plate'],
}



