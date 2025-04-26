# modules for game
from misc.game.game import Game
from misc.game.utils import *
from utils.core import *
from utils.interact import interact

# helpers
import pygame
import numpy as np
import argparse
from collections import defaultdict
from random import randrange
import os
from datetime import datetime


class GamePlay(Game):
    def __init__(self, filename, world, sim_agents):
        """
        初始化 GamePlay 对象。

        Args:
            filename (str): 用于截图命名的基础文件名。
            world (World): 当前环境的 World 对象。
            sim_agents (list): 包含环境中所有 SimAgent 对象的列表。
        """
        Game.__init__(self, world, sim_agents, play=True)
        self.filename = filename
        self.save_dir = 'misc/game/screenshots'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # tally up all gridsquare types
        self.gridsquares = []
        self.gridsquare_types = defaultdict(set) # {type: set of coordinates of that type}
        for name, gridsquares in self.world.objects.items():
            for gridsquare in gridsquares:
                self.gridsquares.append(gridsquare)
                self.gridsquare_types[name].add(gridsquare.location)


    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
            
             # --- 处理键盘按下事件 ---
        elif event.type == pygame.KEYDOWN:
            # Save current image 保存截图 
            if event.key == pygame.K_RETURN:
                image_name = '{}_{}.png'.format(self.filename, datetime.now().strftime('%m-%d-%y_%H-%M-%S'))
                pygame.image.save(self.screen, '{}/{}'.format(self.save_dir, image_name))
                print('just saved image {} to {}'.format(image_name, self.save_dir))
                return
            
            # Switch current agent 切换当前控制的智能体
            if pygame.key.name(event.key) in "1234":
                try:
                    self.current_agent = self.sim_agents[int(pygame.key.name(event.key))-1]
                except:
                    pass
                return

            # Control current agent  移动/交互当前智能体
            x, y = self.current_agent.location
            if event.key in KeyToTuple.keys():
                action = KeyToTuple[event.key]
                self.current_agent.action = action
                interact(self.current_agent, self.world)

    def on_execute(self):
        """
        游戏的主执行循环。
        负责初始化、事件处理、渲染和清理。
        """
        if self.on_init() == False:
            self._running = False

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
            self.on_render()
        self.on_cleanup()


