from collections import namedtuple
import numpy as np
import scipy as sp
import random
from utils.utils import agent_settings


class SubtaskAllocDistribution():
    """Represents a distribution over subtask allocations."""

    def __init__(self, subtask_allocs):
        # subtask_allocs are a list of tuples of (subtask, subtask_agents).
        self.probs = {}
        if len(subtask_allocs) == 0:
            return
        prior = 1./(len(subtask_allocs))
        print('set prior', prior)

        for subtask_alloc in subtask_allocs:
            self.probs[tuple(subtask_alloc)] = prior

    def __str__(self):
        s = ''
        for subtask_alloc, p in self.probs.items():
            s += str(subtask_alloc) + ': ' + str(p) + '\n'
        return s

    def enumerate_subtask_allocs(self):
        return list(self.probs.keys())

    def get_list(self):
        return list(self.probs.items())

    def get(self, subtask_alloc):
        return self.probs[tuple(subtask_alloc)]

    def get_max(self):
        if len(self.probs) > 0:
            max_prob = max(self.probs.values())
            max_subtask_allocs = [subtask_alloc for subtask_alloc, p in self.probs.items() if p == max_prob]
            return random.choice(max_subtask_allocs)
        return None







    def get_max_bucketed(self):
        """
        计算与特定 agent_name 相关的所有单个任务分配的边际概率，
        然后返回包含具有最高边际概率的那个单个任务分配的 *完整* 子任务分配方案。
        (这个方法的逻辑比较复杂，且依赖外部 agent_name)

        Args:
            agent_name (str): 需要计算边际概率的目标智能体名称。

        Returns:
            tuple: 包含目标智能体最可能执行的任务的那个 *完整* 分配方案。
            
            我服辣这也太复杂了
            
            distribution.probs = {
                # 方案1: A1切T, A2切L, A3拿P     (A1T1, A2T2, A3T3): 0.4,
                # 方案2: A1切L, A2切T, A3拿P   (A1T2, A2T1, A3T3): 0.3,
                # 方案3: A1切T, A2拿P, A3切L     (A1T1, A2T3, A3T2): 0.2,
                # 方案4: A1空闲, A2切T, A3切L     (A1None, A2T1, A3T2): 0.1
            }
            # 注意：总概率 0.4 + 0.3 + 0.2 + 0.1 = 1.0
            执行 distribution.get_max_bucketed('A1') 的详细步骤

            初始化:

            subtasks = [] （存储 A1 可能执行的单一任务分配）
            probs = [] （存储对应的边际概率）
            遍历 distribution.probs 中的完整分配方案:

            处理方案1 (概率 0.4): (A1T1, A2T2, A3T3)
            t = A1T1: 'A1' 在 ('A1',) 中。A1T1 不在 subtasks 中。
            subtasks -> [A1T1]
            probs -> [0.4] 


            处理方案2 (概率 0.3): (A1T2, A2T1, A3T3)
            t = A1T2: 'A1' 在 ('A1',) 中。A1T2 不在 subtasks 中。
            subtasks -> [A1T1, A1T2]
            probs -> [0.4, 0.3]


            处理方案3 (概率 0.2): (A1T1, A2T3, A3T2)
            t = A1T1: 'A1' 在 ('A1',) 中。A1T1 在 subtasks 中，索引为 0。
            更新 probs[0]: probs[0] = 0.4 + 0.2 = 0.6。
            subtasks 仍为 [A1T1, A1T2]
            probs 变为 [0.6, 0.3]

            处理方案4 (概率 0.1): (A1None, A2T1, A3T2)
            t = A1None: 'A1' 在 ('A1',) 中。A1None 不在 subtasks 中。
            subtasks -> [A1T1, A1T2, A1None]
            probs -> [0.6, 0.3, 0.1]


            subtasks = [A1T1, A1T2, A1None] (A1切T, A1切L, A1空闲)
            probs = [0.6, 0.3, 0.1] (对应的边际概率) 

            找到 best_subtask (边际概率最高的单一任务分配):

            best_subtask_index = np.argmax(probs) = np.argmax([0.6, 0.3, 0.1]) = （下标） 0。
            best_subtask = subtasks[0] = A1T1 (即 SubtaskAllocation(T1, ('A1',)))。
            结论：综合来看，A1 最可能执行的任务是 Chop(Tomato)，其总概率为 0.6。
            调用 self.get_best_containing(best_subtask) (即 distribution.get_best_containing(A1T1)) :

            内部逻辑:
            初始化 valid_subtask_allocs = [], valid_p = []。
            检查方案1 (A1T1, A2T2, A3T3) (概率 0.4): 包含 A1T1。
            valid_subtask_allocs -> [ (A1T1, A2T2, A3T3) ]
            valid_p -> [ 0.4 ]

            检查方案2 (A1T2, A2T1, A3T3) (概率 0.3): 不包含 A1T1。跳过。

            检查方案3 (A1T1, A2T3, A3T2) (概率 0.2): 包含 A1T1。
            valid_subtask_allocs -> [ (A1T1, A2T2, A3T3), (A1T1, A2T3, A3T2) ]
            valid_p -> [ 0.4, 0.2 ]

            检查方案4 (A1None, A2T1, A3T2) (概率 0.1): 不包含 A1T1。跳过。


            best_index = np.argmax(valid_p) = np.argmax([0.4, 0.2]) = （下标）0。
            返回 valid_subtask_allocs[0]，也就是方案1的元组 (A1T1, A2T2, A3T3)。
            最终返回: get_max_bucketed('A1') 返回方案1的元组：
            (SubtaskAllocation(subtask=recipe.Chop('Tomato'), subtask_agent_names=('A1',)), SubtaskAllocation(subtask=recipe.Chop('Lettuce'), subtask_agent_names=('A2',)), SubtaskAllocation(subtask=recipe.Get('Plate'), subtask_agent_names=('A3',)))

            总结:

            在这个3智能体的例子中，get_max_bucketed('A1') 首先计算出 A1 做 Chop(Tomato) 的边际概率是 0.6 (来自方案1和方案3)，做 Chop(Lettuce) 的边际概率是 0.3 (来自方案2)，
            空闲的边际概率是 0.1 (来自方案4)。因为 0.6 是最高的，所以该方法确定 A1 最可能是在 Chop(Tomato)。
            然后，它在所有包含 A1 做 Chop(Tomato) 的完整方案（即方案1和方案3）中，找到了本身概率最高的那个方案，也就是方案1（概率0.4 > 方案3的概率0.2）。因此，最终返回了方案1。
        """
        
        subtasks = []
        probs = []
        for subtask_alloc, p in self.probs.items():
            for t in subtask_alloc:
                if agent_name in t.subtask_agent_names:
                    # If already accounted for, then add probability.
                    if t in subtasks:
                        probs[subtasks.index(t)] += p
                    # Otherwise, make a new element in the distribution.
                    else:
                        subtasks.append(t)
                        probs.append(p)
        best_subtask = subtasks[np.argmax(probs)]
        return self.probs.get_best_containing(best_subtask)






    def get_best_containing(self, subtask):
        """Return max likelihood subtask_alloc that contains the given subtask."""
        valid_subtask_allocs = []
        valid_p = []
        for subtask_alloc, p in self.probs.items():
            if subtask in subtask_alloc:
                valid_subtask_allocs.append(subtask)
                valid_p.append(p)
        return valid_subtask_allocs[np.argmax(valid_p)]





    def set(self, subtask_alloc, value):
        self.probs[tuple(subtask_alloc)] = value

    def update(self, subtask_alloc, factor):
        self.probs[tuple(subtask_alloc)] *= factor

    def delete(self, subtask_alloc):
        try:
            del self.probs[tuple(subtask_alloc)]
        except:
            print('subtask_alloc {} not found in probsdict'.format(subtask_alloc))

    def normalize(self):
        total = sum(self.probs.values())
        for subtask_alloc in self.probs.keys():
            if total == 0:
                self.probs[subtask_alloc] = 1./len(self.probs)
            else:
                self.probs[subtask_alloc] *= 1./total
        return self.probs
