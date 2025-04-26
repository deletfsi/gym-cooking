import recipe_planner.utils as recipe

# core modules
from utils.core import Object

# helpers
import networkx as nx
import copy


class STRIPSWorld:
    def __init__(self, world, recipes):
        self.initial = recipe.STRIPSState()
        self.recipes = recipes

        # set initial state
        self.initial.add_predicate(recipe.NoPredicate())
        for obj in world.get_object_list():
            if isinstance(obj, Object):
                for obj_name in ['Plate', 'Tomato', 'Lettuce', 'Onion']:
                    if obj.contains(obj_name):
                        '''如果包含，则在 STRIPS 初始状态中添加一个 "Fresh" 谓词'''
                        self.initial.add_predicate(recipe.Fresh(obj_name))

    def generate_graph(self, recipe, max_path_length):
        '''  
        为单个食谱生成状态空间图。
        使用广度优先搜索 (BFS) 的变体来探索从初始状态开始，通过应用食谱定义的动作可以达到的状态。'''
        all_actions = recipe.actions   # set
        goal_state = None

        new_preds = set()
        graph = nx.DiGraph()
        graph.add_node(self.initial, obj=self.initial)
        frontier = set([self.initial])
        next_frontier = set()
        for i in range(max_path_length):
            # print('CHECKING FRONTIER #:', i)
            for state in frontier:
                # for each action, check whether from this state
                for a in all_actions:
                    if a.is_valid_in(state):
                        next_state = a.get_next_from(state)
                        for p in next_state.predicates:
                            new_preds.add(str(p))
                        graph.add_node(next_state, obj=next_state)
                        graph.add_edge(state, next_state, obj=a)

                        # as soon as goal is found, break and return                       
                        if self.check_goal(recipe, next_state) and goal_state is None:
                            goal_state = next_state
                            return graph, goal_state
                        
                        next_frontier.add(next_state)

            frontier = next_frontier.copy()
        
        if goal_state is None:
            print('goal state could not be found, try increasing --max-num-subtasks')
            import sys; sys.exit(0)
        
        return graph, goal_state





    def get_subtasks(self, max_path_length=10, draw_graph=False):
        '''对于沙拉，可能存在多条长度相同的最短路径，这取决于动作的并行或不同顺序。例如：
        路径 A (先合并食材): Get(T) -> Get(L) -> Chop(T) -> Chop(L) -> Merge(T, L) -> Get(P) -> Merge(TL, P) -> Deliver(TLP)
        路径 B (先拿盘子，T先上盘): Get(T) -> Chop(T) -> Get(P) -> Merge(T, P) -> Get(L) -> Chop(L) -> Merge(L, TP) -> Deliver(TLP)
        路径 C (先拿盘子，L先上盘): Get(L) -> Chop(L) -> Get(P) -> Merge(L, P) -> Get(T) -> Chop(T) -> Merge(T, LP) -> Deliver(TLP)

        action_paths.append(union_action_path) -> action_paths 变为 [ {Get(T), Get(L), Get(P), Chop(T), Chop(L), Merge(T,L), Merge(TL,P), Merge(T,P), Merge(L,TP), Merge(L,P), Merge(T,LP), Deliver(TLP)} ]
        '''
        action_paths = []

        for recipe in self.recipes:
            graph, goal_state = self.generate_graph(recipe, max_path_length)

            if draw_graph:   # not recommended for path length > 4
                nx.draw(graph, with_labels=True)
                plt.show()
            
            all_state_paths = nx.all_shortest_paths(graph, self.initial, goal_state)
            union_action_path = set()
            for state_path in all_state_paths:
                action_path = [graph[state_path[i]][state_path[i+1]]['obj'] for i in range(len(state_path)-1)]
                union_action_path = union_action_path | set(action_path)
            # print('all tasks for recipe {}: {}\n'.format(recipe, ', '.join([str(a) for a in union_action_path])))
            action_paths.append(union_action_path)

        return action_paths
        

    def check_goal(self, recipe, state):
        # check if this state satisfies completion of this recipe
        return state.contains(recipe.goal)



''' 
# --- 示例 ---
# 假设我们有一个 SimpleTomato 食谱，其目标是 Delivered('Plate-Tomato')
# 假设初始状态 self.initial = [NoPredicate(), Fresh('Tomato'), Fresh('Plate')]
# 食谱的动作集 recipe.actions 包含:
#   Get('Plate'), Get('Tomato'), Chop('Tomato'),
#   Merge('Tomato', 'Plate', [Chopped('Tomato'), Fresh('Plate')], None),
#   Deliver('Plate-Tomato', [Merged('Plate-Tomato')], None) # 假设 Merge 的结果是 Merged

# generate_graph 会进行类似 BFS 的搜索：
# 层 0: [NoPredicate(), Fresh('Tomato'), Fresh('Plate')]
# 层 1:
#   - 应用 Get('Tomato') (无效，已有 Fresh('Tomato'))
#   - 应用 Get('Plate') (无效，已有 Fresh('Plate'))
#   - 应用 Chop('Tomato') (有效) -> next_state1 = [NoPredicate(), Chopped('Tomato'), Fresh('Plate')] (添加边 Chop(T))
# 层 2:
#   - 从 next_state1 应用 Merge('Tomato', 'Plate', ...) (有效) -> next_state2 = [NoPredicate(), Merged('Plate-Tomato')] (添加边 Merge(T,P))
# 层 3:
#   - 从 next_state2 应用 Deliver('Plate-Tomato', ...) (有效) -> next_state3 = [NoPredicate(), Delivered('Plate-Tomato')] (添加边 Deliver(PT))
#       - 检查 check_goal(SimpleTomato, next_state3) -> True，因为 next_state3 包含 SimpleTomato.goal
#       - 找到目标！返回 graph 和 next_state3

# get_subtasks 会找到最短路径：
#   - all_shortest_paths 返回 [[initial, next_state1, next_state2, next_state3]]
#   - 提取动作：[Chop('Tomato'), Merge('Tomato', 'Plate', ...), Deliver('Plate-Tomato')]
#   - union_action_path 集合包含这三个动作。
#   - action_paths = [ {Chop('Tomato'), Merge(...), Deliver(...)} ]
# 最终在 OvercookedEnvironment.run_recipes() 中，这个集合会被扁平化返回给智能体。'''