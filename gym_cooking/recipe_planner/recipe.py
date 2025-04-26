from utils.core import *
import recipe_planner.utils as recipe


class Recipe:
    def __init__(self, name):
        self.name = name
        self.contents = []
        self.actions = set()
        self.actions.add(recipe.Get('Plate'))

    def __str__(self):
        return self.name

    def add_ingredient(self, item):
        self.contents.append(item)

        # always starts with FRESH
        self.actions.add(recipe.Get(item.name))

        if item.state_seq == FoodSequence.FRESH_CHOPPED:
            self.actions.add(recipe.Chop(item.name))
            self.actions.add(recipe.Merge(item.name, 'Plate',\
                [item.state_seq[-1](item.name), recipe.Fresh('Plate')], None))

    def add_goal(self):
        # 按名称字母顺序对食材列表排序
        self.contents = sorted(self.contents, key = lambda x: x.name)   # list of Food objects
        
        # 提取排序后食材的名称列表
        self.contents_names = [c.name for c in self.contents]   # list of strings
        
        # 生成食谱的完整名称 (不含盘子)，用 '-' 连接，按字母排序
        # 例如: ['Lettuce', 'Tomato'] -> 'Lettuce-Tomato'
        self.full_name = '-'.join(sorted(self.contents_names))   # string
        
         # 生成食谱的完整装盘名称 (包含盘子)，用 '-' 连接，按字母排序
        # 例如: ['Lettuce', 'Plate', 'Tomato'] -> 'Lettuce-Plate-Tomato'
        self.full_plate_name = '-'.join(sorted(self.contents_names + ['Plate']))   # string
        
        # 定义食谱的最终目标谓词：完整的装盘物品被递送 (Delivered)
        # 例如: Delivered('Lettuce-Plate-Tomato')
        self.goal = recipe.Delivered(self.full_plate_name)
        
        # 将递送最终成品的动作添加到 actions 集合中
        # 例如: Deliver('Lettuce-Plate-Tomato')
        self.actions.add(recipe.Deliver(self.full_plate_name))







    def add_merge_actions(self):
        ''' 
        为食谱添加所有可能的合并 (Merge) 动作。
        这个方法旨在处理各种沙拉或生的装盘蔬菜的组合方式。
        它会生成将不同数量的食材组合在一起，以及将食材与盘子组合的所有有效 Merge 动作。
        # 遍历可能的食材组合数量 i (从 2 到 食材总数)
        
        # should be general enough for any kind of salad / raw plated veggies

        # alphabetical, joined by dashes ex. Ingredient1-Ingredient2-Plate
        #self.full_name = '-'.join(sorted(self.contents + ['Plate']))

        # for any plural number of ingredients'''  
        
        '''  
 Salad  外层循环 for i in range(2, len(self.contents)+1):   2个食材 ， len(self.contents)+1) = 3 ， 实际上也就是2
  相当于只考虑 i=2 , combinations(['Lettuce', 'Tomato'], 2) 只会生成一个组合 combo: ('Lettuce', 'Tomato')
  
 添加动作 1 (组合与盘子合并):   can merge all with plate
self.actions.add(recipe.Merge('Lettuce-Tomato', 'Plate', [recipe.Merged('Lettuce-Tomato'), recipe.Fresh('Plate')], None))

内部循环 for item in combo:
第一次迭代: item = 'Lettuce' , rem = ['Tomato'] , rem_str = 'Tomato' , len(rem) == 1 条件为 True
 plate_str = 'Lettuce-Plate'
rem_plate_str = 'Plate-Tomato'


    self.actions.add(recipe.Merge('Lettuce', 'Tomato', [recipe.Chopped('Lettuce'), recipe.Chopped('Tomato')], None)
    self.actions.add(recipe.Merge('Tomato', 'Lettuce-Plate'))
    self.actions.add(recipe.Merge('Lettuce', 'Plate-Tomato'))


第二次迭代: item = 'Tomato',rem = ['Lettuce'],rem_str = 'Lettuce',len(rem) == 1 条件为 True。
plate_str = 'Plate-Tomato'
rem_plate_str = 'Lettuce-Plate'

self.actions.add(recipe.Merge('Tomato', 'Lettuce', [recipe.Chopped('Tomato'), recipe.Chopped('Lettuce')], None))
self.actions.add(recipe.Merge('Lettuce', 'Plate-Tomato'))
self.actions.add(recipe.Merge('Tomato', 'Lettuce-Plate'))








如果是 onionSala ， for i in range(2, len(self.contents)+1):食材数量为 3，所以 i 会取值为 2 和 3。

i = 2
combinations(['Lettuce', 'Onion', 'Tomato'], 2) 会生成以下组合 combo:
('Lettuce', 'Onion')
('Lettuce', 'Tomato')
('Onion', 'Tomato')


combo = ('Lettuce', 'Onion')

Merge('Lettuce-Onion', 'Plate', [Merged('Lettuce-Onion'), Fresh('Plate')], None)
    当 item = 'Lettuce', rem = ['Onion'], rem_str = 'Onion'
        Merge('Lettuce', 'Onion', [Chopped('Lettuce'), Chopped('Onion')], None)
        Merge('Onion', 'Lettuce-Plate')
        Merge('Lettuce', 'Onion-Plate')

    当 item = 'Onion', rem = ['Lettuce'], rem_str = 'Lettuce'
        Merge('Onion', 'Lettuce', [Chopped('Onion'), Chopped('Lettuce')], None)
        Merge('Lettuce', 'Onion-Plate')
        Merge('Onion', 'Lettuce-Plate')


combo = ('Lettuce', 'Tomato')
Merge('Lettuce-Tomato', 'Plate', [Merged('Lettuce-Tomato'), Fresh('Plate')], None)
    Merge('Lettuce', 'Tomato', [Chopped('Lettuce'), Chopped('Tomato')], None)
        Merge('Tomato', 'Lettuce-Plate')
        Merge('Lettuce', 'Plate-Tomato')

    Merge('Tomato', 'Lettuce', [Chopped('Tomato'), Chopped('Lettuce')], None)  
        Merge('Lettuce', 'Plate-Tomato') 
        Merge('Tomato', 'Lettuce-Plate')
        
    
combo = ('Onion', 'Tomato'):   
Merge('Onion-Tomato', 'Plate', [Merged('Onion-Tomato'), Fresh('Plate')], None)
    Merge('Onion', 'Tomato', [Chopped('Onion'), Chopped('Tomato')], None)
        Merge('Tomato', 'Onion-Plate')
        Merge('Onion', 'Plate-Tomato')

    Merge('Tomato', 'Onion', [Chopped('Tomato'), Chopped('Onion')], None) 
        Merge('Onion', 'Plate-Tomato')
        Merge('Tomato', 'Onion-Plate')
        
        
        
当 i = 3 时 (考虑三个食材的组合):   唯一组合 ('Lettuce', 'Onion', 'Tomato') 
Merge('Lettuce-Onion-Tomato', 'Plate', [Merged('Lettuce-Onion-Tomato'), Fresh('Plate')], None)  

当 item = 'Lettuce', rem = ['Onion', 'Tomato'], rem_str = 'Onion-Tomato':
len(rem) == 2，进入 else 分支。
plate_str = 'Lettuce-Plate', rem_plate_str = 'Onion-Plate-Tomato'
    Merge('Lettuce', 'Onion-Tomato')
    Merge('Lettuce-Plate', 'Onion-Tomato', [Merged('Lettuce-Plate'), Merged('Onion-Tomato')], None)
    Merge('Lettuce', 'Onion-Plate-Tomato')
    
当 item = 'Onion', rem = ['Lettuce', 'Tomato'], rem_str = 'Lettuce-Tomato':
len(rem) == 2，进入 else 分支
plate_str = 'Onion-Plate', rem_plate_str = 'Lettuce-Plate-Tomato'
    Merge('Onion', 'Lettuce-Tomato')
    Merge('Onion-Plate', 'Lettuce-Tomato', [Merged('Onion-Plate'), Merged('Lettuce-Tomato')], None)
    Merge('Onion', 'Lettuce-Plate-Tomato')
    
当 item = 'Tomato', rem = ['Lettuce', 'Onion'], rem_str = 'Lettuce-Onion':   
len(rem) == 2，进入 else 分支
plate_str = 'Plate-Tomato', rem_plate_str = 'Lettuce-Onion-Plate'
    Merge('Tomato', 'Lettuce-Onion')
    Merge('Plate-Tomato', 'Lettuce-Onion', [Merged('Plate-Tomato'), Merged('Lettuce-Onion')], None)
    Merge('Tomato', 'Lettuce-Onion-Plate')
        '''
        for i in range(2, len(self.contents)+1):
            # for any combo of i ingredients to be merged
            for combo in combinations(self.contents_names, i):
                # can merge all with plate
                self.actions.add(recipe.Merge('-'.join(sorted(combo)), 'Plate',\
                    [recipe.Merged('-'.join(sorted(combo))), recipe.Fresh('Plate')], None))

                # for any one item to be added to the i-1 rest
                for item in combo:
                    rem = list(combo).copy()
                    rem.remove(item)
                    rem_str = '-'.join(sorted(rem))
                    plate_str = '-'.join(sorted([item, 'Plate']))
                    rem_plate_str = '-'.join(sorted(rem + ['Plate']))

                    # can merge item with remaining
                    if len(rem) == 1:
                        self.actions.add(recipe.Merge(item, rem_str,\
                            [recipe.Chopped(item), recipe.Chopped(rem_str)], None))
                        self.actions.add(recipe.Merge(rem_str, plate_str))
                        self.actions.add(recipe.Merge(item, rem_plate_str))
                    else:
                        self.actions.add(recipe.Merge(item, rem_str))
                        self.actions.add(recipe.Merge(plate_str, rem_str,\
                            [recipe.Merged(plate_str), recipe.Merged(rem_str)], None))
                        self.actions.add(recipe.Merge(item, rem_plate_str))

class SimpleTomato(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'Tomato')
        self.add_ingredient(Tomato(state_index=-1))
        self.add_goal()
        self.add_merge_actions()

class SimpleLettuce(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'Lettuce')
        self.add_ingredient(Lettuce(state_index=-1))
        self.add_goal()
        self.add_merge_actions()

class Salad(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'Salad')
        self.add_ingredient(Tomato(state_index=-1))
        self.add_ingredient(Lettuce(state_index=-1))
        self.add_goal()
        self.add_merge_actions()

class OnionSalad(Recipe):
    def __init__(self):
        Recipe.__init__(self, 'OnionSalad')
        self.add_ingredient(Tomato(state_index=-1))
        self.add_ingredient(Lettuce(state_index=-1))
        self.add_ingredient(Onion(state_index=-1))
        self.add_goal()
        self.add_merge_actions()


