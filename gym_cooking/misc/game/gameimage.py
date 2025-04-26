import pygame
import os
import numpy as np
from PIL import Image
from misc.game.game import Game
# from misc.game.utils import *


class GameImage(Game):
    def __init__(self, filename, world, sim_agents, record=False):
        Game.__init__(self, world, sim_agents)
        self.game_record_dir = 'misc/game/record/{}/'.format(filename)
        self.record = record


    def on_init(self):
        """
        初始化 GameImage 对象，包括调用父类的初始化和设置记录目录。
        """
        super().on_init()

        if self.record:
            # Make game_record folder if doesn't already exist
            if not os.path.exists(self.game_record_dir):
                os.makedirs(self.game_record_dir)

            # Clear game_record folder
            for f in os.listdir(self.game_record_dir):
                os.remove(os.path.join(self.game_record_dir, f))

    def get_image_obs(self):
        """
        获取当前游戏画面的 NumPy RGB 数组表示。
        这对于需要图像输入的机器学习模型非常有用。

        Returns:
            np.ndarray: 一个形状为 (height, width, 3) 的 uint8 NumPy 数组，
                        表示当前屏幕的 RGB 图像。通道顺序是 R, G, B。
                        (注意：代码实现中 color.g, color.b, color.r 赋值给了 0, 1, 2 通道，
                         所以实际通道顺序是 G, B, R，这可能需要根据后续使用调整)
        """
        self.on_render()
        img_int = pygame.PixelArray(self.screen)
        img_rgb = np.zeros([img_int.shape[1], img_int.shape[0], 3], dtype=np.uint8)
        for i in range(img_int.shape[0]):
            for j in range(img_int.shape[1]):
                color = pygame.Color(img_int[i][j])
                img_rgb[j, i, 0] = color.g
                img_rgb[j, i, 1] = color.b
                img_rgb[j, i, 2] = color.r
        return img_rgb


    def save_image_obs(self, t):
        """
        将当前游戏画面保存为 PNG 图像文件。

        Args:
            t (int): 当前的时间步 (timestep)，用于命名文件。
        """
        self.on_render()
        pygame.image.save(self.screen, '{}/t={:03d}.png'.format(self.game_record_dir, t))
