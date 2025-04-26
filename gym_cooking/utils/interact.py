from utils.core import *
import numpy as np

def interact(agent, world):
    """ 
    根据智能体(agent)的动作(agent.action)在世界(world)中执行交互。
    这个函数包含了所有低层动作的实现逻辑，如移动、拾取、放下、合并、切菜、递送等。

    Args:
        agent (SimAgent): 需要执行动作的模拟智能体对象。
        world (World): 当前环境的世界对象。
     
    Carries out interaction for this agent taking this action in this world.

    The action that needs to be executed is stored in `agent.action`.
    """

    # agent does nothing (i.e. no arrow key)
    if agent.action == (0, 0):
        return

    action_x, action_y = world.inbounds(tuple(np.asarray(agent.location) + np.asarray(agent.action)))
     # 获取目标位置对应的格子对象 (GridSquare 实例，如 Floor, Counter, Cutboard, Delivery)。
    gs = world.get_gridsquare_at((action_x, action_y))

    ''' 情况 1: 目标格子是地板 (Floor), 调用 agent.move_to 方法，将智能体的位置更新为目标格子的位置。
        move_to 内部也会更新智能体持有物品的位置（如果持有的话）。
         # if floor in front --> move to that square'''
    if isinstance(gs, Floor): #and gs.holding is None:
        agent.move_to(gs.location)

    #情况 2: 智能体正持有物品   
    elif agent.holding is not None:
        ''' 情况 2a: 目标格子是递送点 (Delivery)'''
        if isinstance(gs, Delivery):
            obj = agent.holding
            if obj.is_deliverable():  #  检查该物品是否满足递送条件
                gs.acquire(obj)
                agent.release()
                print('\nDelivered {}!'.format(obj.full_name))
        
        #情况 2b: 目标格子已被占用 # if occupied gridsquare in front --> try merging   
        elif world.is_occupied(gs.location):
            ''' # 获取目标格子上放置的物品对象'''
            obj = world.get_object_at(gs.location, None, find_held_objects = False)

            if mergeable(agent.holding, obj):
                world.remove(obj)
                o = gs.release() # agent is holding object
                world.remove(agent.holding)
                agent.acquire(obj)
                world.insert(agent.holding)
                # if playable version, merge onto counter first
                if world.arglist.play:
                    gs.acquire(agent.holding)
                    agent.release()


        #  --- 情况 2c: 目标格子是空的交互点 (例如 Counter, Cutboard) (尝试放下或切菜)    
        # if holding something, empty gridsquare in front --> chop or drop 如果目标格子未被占用 (不是 Floor，因为前面处理过了；不是 Delivery，前面处理过了；没有其他物品)'''
        elif not world.is_occupied(gs.location):
            obj = agent.holding
            if isinstance(gs, Cutboard) and obj.needs_chopped() and not world.arglist.play:
                # normally chop, but if in playable game mode then put down first
                obj.chop()
            else:
                gs.acquire(obj) # obj is put onto gridsquare
                agent.release()
                assert world.get_object_at(gs.location, obj, find_held_objects =\
                    False).is_held == False, "Verifying put down works"

    # 情况 3: 智能体未持有物品 (agent.holding is None )
    elif agent.holding is None:
        '''# --- 情况 3a: 目标格子已被占用 (尝试拾取) ---
        # 检查目标格子是否被占用，并且目标格子不是递送点 (不能从递送点拾取)
        not empty in front --> pick up'''
        if world.is_occupied(gs.location) and not isinstance(gs, Delivery):
            obj = world.get_object_at(gs.location, None, find_held_objects = False)
            # if in playable game mode, then chop raw items on cutting board
            if isinstance(gs, Cutboard) and obj.needs_chopped() and world.arglist.play:
                obj.chop()
            else:
                gs.release()
                agent.acquire(obj)

         # --- 情况 3b: 目标格子是空的交互点 (无操作) ---    
         # 如果智能体没拿东西，并且试图移动到一个空的 Counter、Cutboard 或 Delivery，则不发生任何状态改变。   if empty in front --> interact'''
        elif not world.is_occupied(gs.location):
            pass
