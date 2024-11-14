#######################################################################
# Name: Robot.py

#######################################################################

import os
import numpy as np
import numpy.ma as ma
from skimage import io
import matplotlib.pyplot as plt
from scipy import spatial
from random import shuffle

import Sensor
from build.astar import *

class Robot:
    def __init__(self,index_map, train, plot):
        """
        属性:
        global_map
        robot_position: 机器人当前位置
        start_position: 机器人开始时的位置
        robot_belief: 机器人已观测(observe)的地图
        map_size

        x2frontier: 机器人探索到的前沿x
        y2frontier: 机器人探索到的前沿y

        senseor_range: 传感器角度
        robot_size: 

        方法:
        map_setup: 初始化地图
        begin: 
        plot
        
        """
        self.mode = train
        self.plot = plot

        if self.mode:
            self.map_dir = './DungeonMaps/train'
        else:
            self.map_dir = './DungeonMaps/test'

        self.map_list = os.listdir(self.map_dir)
        self.map_number = np.size(self.map_list)
        if self.mode:
            shuffle(self.map_list)
        self.li_map = index_map

        self.global_map, self.robot_position = self.map_setup(self.map_dir + '/' + self.map_list[self.li_map])
        # self.start_position=self.robot_position
        self.robot_belief = np.ones(self.global_map.shape) * 127  # 127为中间值[0, 255]
        self.map_size = np.shape(self.global_map)
        self.t = self.map_points(self.global_map)

        self.free_tree = spatial.KDTree(self.free_points(self.global_map).tolist())

        self.old_position = np.zeros([2])
        self.old_robot_belief = np.empty([0])

        if self.plot:
            self.xPoint = np.array([self.robot_position[0]])
            self.yPoint = np.array([self.robot_position[1]])
            self.x2frontier = np.empty([0])
            self.y2frontier = np.empty([0])

        self.sensor_range = 80
        self.robot_size = 6
        self.local_size = 40
        self.finish_percent = 0.98
        self.resolution = 1

        # 获取当前脚本所在目录的路径
        # os.path.realpath(__file__) 返回脚本的绝对路径（去掉任何符号链接）
        # os.path.dirname() 获取文件的目录部分，即当前脚本所在的文件夹路径
        current_dir = os.path.dirname(os.path.realpath(__file__)) 
        self.action_space = np.genfromtxt(current_dir + '/action_points.csv', delimiter=",")

        self.fig=plt.figure()

    def begin(self):
        """
        机器人开始探索
        
        """
        self.robot_belief = Sensor.sensor_work(self.robot_position, 
                                          self.sensor_range, 
                                          self.robot_belief, 
                                          self.global_map)
        
        step_map = self.robot_in_map(self.robot_position, 
                                     self.robot_size, 
                                     self.t, 
                                     self.robot_belief)
        
        map_local = self.local_map(self.robot_position, 
                                   step_map, 
                                   self.map_size, 
                                   self.sensor_range + self.local_size)

        return map_local 

    def step(self, action_index):
        """
        模拟了机器人在地图中执行一步动作的过程
        """
        terminal = False
        complete = False
        new_location = False
        all_map = False

        # 备份机器人的当前位置和感知地图
        self.old_position = self.robot_position.copy()
        self.old_robot_belief = self.robot_belief.copy()

        # take action
        # self.robot_position已经更新
        self.take_action(action_index, self.robot_position)

        # collision check
        collision_points, collision_index = self.collision_check(
            self.old_position, self.robot_position, self.map_size,self.global_map)
        
        # 根据碰撞情况更新机器人状态
        if collision_index:
            # 寻找碰撞点附近的一个可行位置
            self.robot_position = self.nearest_free(self.free_tree, collision_points)

        # 更新机器人传感器地图信息
        self.robot_belief = Sensor.sensor_work(self.robot_position, 
                                                self.sensor_range, 
                                                self.robot_belief, 
                                                self.global_map)
        
        # 在地图上表示出更新的机器人所在的位置和占用区域
        step_map = self.robot_in_map(self.robot_position, 
                                        self.robot_size, 
                                        self.t, 
                                        self.robot_belief)
        
        map_local = self.local_map(self.robot_position, 
                                   step_map, 
                                   self.map_size, 
                                   self.sensor_range + self.local_size)
        
        # 根据旧的感知地图 当前的感知地图 和 碰撞信息 计算reward
        reward = self.get_reward(self.old_robot_belief, self.robot_belief, collision_index)

        if reward <= 0.02 and not collision_index: # 地图更新很小
            reward = -0.8
            new_location = True
            terminal = True
        
        if collision_index: # 有碰撞
            if not self.mode:
                # 如果在测试模式下，机器人会根据碰撞信息选择适当的动作来避免碰撞
                new_location = False
                terminal = False
            else:
                new_location = True
                terminal = True
            
            if self.plot and self.mode:
                self.xPoint = ma.append(self.xPoint, self.robot_position[0])
                self.yPoint = ma.append(self.yPoint, self.robot_position[1])
                self.plot_all()

            self.robot_position = self.old_position.copy()
            self.robot_belief = self.old_robot_belief.copy()

            if self.plot and self.mode:
                self.xPoint[self.xPoint.size-1] = ma.masked
                self.yPoint[self.yPoint.size-1] = ma.masked
        else: # 无碰撞
            if self.plot:
                self.xPoint = ma.append(self.xPoint, self.robot_position[0])
                self.yPoint = ma.append(self.yPoint, self.robot_position[1])
                self.plot_all()

        # check if exploration is finished
        if np.size(np.where(self.robot_belief == 255)
                   )/np.size(np.where(self.global_map == 255)) > self.finish_percent:
            self.li_map += 1
            if self.li_map == self.map_number:
                self.li_map = 0
                all_map = True
            self.__init__(self.li_map, self.mode, self.plot)
            complete = True
            new_location = False
            terminal = True
            self.plot_all()

        return map_local, reward, terminal, complete, new_location, collision_index, all_map

    def rescuer(self):
            complete = False
            all_map = False

            pre_position = self.robot_position.copy()

            self.robot_position = self.frontier(self.robot_belief, 
                                                self.map_size, 
                                                self.t)
            # 更新机器人的传感器范围内的信息
            self.robot_belief = Sensor.sensor_work(self.robot_position, 
                                                   self.sensor_range, 
                                                   self.robot_belief, 
                                                   self.global_map)
            # 更新带有机器人模型的地图
            step_map = self.robot_in_map(self.robot_position, 
                                         self.robot_size, 
                                         self.t, 
                                         self.robot_belief)
            # 提取局部地图
            map_local = self.local_map(self.robot_position, 
                                       step_map, 
                                       self.map_size, 
                                       self.sensor_range + self.local_size)

            # 路径规划和可视化
            if self.plot:
                # 使用 A* 算法计算机器人从 pre_position 到 robot_position 的路径，并保存在 path 中
                path = self.astar_path(self.robot_belief, 
                                       pre_position.tolist(), 
                                       self.robot_position.tolist())
                
                self.x2frontier = ma.append(self.x2frontier, ma.masked)
                self.y2frontier = ma.append(self.y2frontier, ma.masked)
                self.x2frontier = ma.append(self.x2frontier, path[1, :])
                self.y2frontier = ma.append(self.y2frontier, path[0, :])
                self.xPoint = ma.append(self.xPoint, ma.masked)
                self.yPoint = ma.append(self.yPoint, ma.masked)
                self.xPoint = ma.append(self.xPoint, self.robot_position[0])
                self.yPoint = ma.append(self.yPoint, self.robot_position[1])
                self.plot_all()

            if np.size(np.where(self.robot_belief == 255))/np.size(np.where(self.global_map == 255)) > self.finish_percent:
                self.li_map += 1
                if self.li_map == self.map_number:
                    self.li_map = 0
                    all_map = True
                self.__init__(self.li_map, self.mode, self.plot)
                complete = True
                new_location = False
                terminal = True
            return map_local, complete, all_map

    def frontier(self, robot_belief, map_size, points):
        """
        在给定地图中找到边界点 frontier points
        """
        y_len = map_size[0]
        x_len = map_size[1]
        mapping = robot_belief.copy()

        # 0-1 unknown area map
        # mapping = (mapping == 127) * 1: 将未知区域（假设值为 127）标记为 1，其他区域为 0
        # 生成仅包含未知区域的二值掩膜
        mapping = (mapping == 127) * 1
        # mapping = np.lib.pad(...): 在掩膜四周填充一圈 0 值，使得后续计算边界更加方便
        mapping = np.lib.pad(mapping, ((1, 1), (1, 1)), 'constant', constant_values=0)

        # 这一步通过对 mapping 进行多个方向上的相邻像素值求和来创建边界图 fro_map
        # 代码分成八个部分，每个部分分别检查未知区域的八个相邻位置
        # fro_map 中的每个元素表示该点周围未知区域的数量
        # 如果一个已知点附近有多个未知区域点，则它是潜在的边界点
        fro_map = mapping[2:][:, 1:x_len + 1] + mapping[:y_len][:, 1:x_len + 1] + mapping[1:y_len + 1][:, 2:] + \
                mapping[1:y_len + 1][:, :x_len] + mapping[:y_len][:, 2:] + mapping[2:][:, :x_len] + mapping[2:][:,
                                                                                                    2:] + \
                mapping[:y_len][:, :x_len]
        
        # ind_free: 提取操作地图中所有已知自由区域（值为 255）的索引
        ind_free = np.where(robot_belief.ravel(order='F') == 255)[0]

        # ind_fron_1: 提取 fro_map 中值大于 1 的索引（即未知区域数量大于 1）
        ind_fron_1 = np.where(1 < fro_map.ravel(order='F'))[0]

        # ind_fron_2: 提取 fro_map 中值小于 8 的索引（即不是被完全未知区域包围的点）
        ind_fron_2 = np.where(fro_map.ravel(order='F') < 8)[0]

        # ind_fron: 找到满足上述两条件的交集索引（即既有未知区域，又不是完全未知的点）
        ind_fron = np.intersect1d(ind_fron_1, ind_fron_2)

        # ind_to: ind_free 与 ind_fron 的交集，表示同时在 op_map 为自由区域且满足边界条件的点
        ind_to = np.intersect1d(ind_free, ind_fron)

        # f = points[ind_to]: 使用 ind_to 提取符合条件的边界点坐标
        f = points[ind_to]
        f = f.astype(int)

        # f[0]: 返回符合条件的第一个边界点坐标，以便机器人选择该点作为下一步探索目标
        return f[0]

    def map_setup(self, map_path):
        """
        初始化地图

        input:
        map_path: 地图文件的路径
        测试阶段为: ./img_6000.png

        output:
        global_map: 二值化地图图像，分可通行区域(255) 和 障碍区域(1)
        robot_location: 机器人在地图中的初始位置 [x0,y0]
        """
        
        # 使用 scikit-image 库中的 imread 函数加载图像
        # 第二个参数 1 表示以灰度模式读取图像
        # 图像数据通常加载为 0-1 的浮点数，所以这里乘以 255，将图像数据转换为 0-255 的整数范围
        # astype(int) 将图像矩阵的数据类型转换为整型
        global_map = (io.imread(map_path, 1) * 255).astype(int)

        # 地图中黄点位置 
        # np.nonzero(global_map == 208) 的结果是一个元组
        # 其中包含两个数组 分别是 global_map 中所有值为 208 的像素点的行和列索引
        robot_location = np.nonzero(global_map == 208)
        # 这两部分分别从二维数组 np.array(robot_location)中
        # 第 1 行（列索引）和第 0 行（行索引）选择第 127 个元素
        # 127 应该和点的像素大小有关
        # print(len(robot_location[0])) -> 256 其中正中间就是127
        robot_location = np.array([np.array(robot_location)[1, 127], 
                                   np.array(robot_location)[0, 127]])
        
        # global_map > 150 将像素值大于 150 的位置设为 True 
        # 小于等于 150 的设为 False
        # 将地图二值化: 分为“可通行区域”（True）和“不可通行区域”（False）
        global_map = (global_map > 150)

        # 这一步完成后，global_map 是一个只包含 1 和 255 的矩阵，用于表示地图。
        global_map = global_map * 254 + 1

        return global_map, robot_location
    
    def map_points(self, map_glo):
        """
        将一个二维地图的所有坐标点提取出来，返回一个包含地图中每个像素点位置的坐标数组
        """
        map_x = map_glo.shape[1]
        map_y = map_glo.shape[0]

        # 生成了 x 方向上的等间隔点，范围从 0 到 map_x - 1
        x = np.linspace(0, map_x - 1, map_x)
        y = np.linspace(0, map_y - 1, map_y)

        # 分别将 t1 和 t2 转置并展平（转化为一维数组）
        t1, t2 = np.meshgrid(x, y)

        # 将所有 x 坐标和 y 坐标堆叠在一起
        # .T 将数组转置，使其变成形状为 (map_x * map_y, 2) 的数组，每一行表示一个 (x, y) 坐标点
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        
        return points

    def robot_in_map(self, position, robot_size, points, robot_belief):
        """
        在地图上表示出机器人所在的位置和占用区域

        position:  机器人当前位置
        robot_size: 6, 机器人周围半径为6的像素点, 机器人本身的大小
        """
        map_copy = robot_belief.copy()

        # 调用 range_search 函数，找到以 position 为中心、robot_size 为半径的范围内的点
        # 用来表示机器人占用的区域
        robot_points = self.range_search(position, robot_size, points)

        # 在地图上标记机器人的位置
        # 半径为6个像素的圆, 圆心为机器人的位置position
        for i in range(0, robot_points.shape[0]):
            rob_loc = np.int32(robot_points[i, :])
            rob_loc = np.flipud(rob_loc)
            map_copy[tuple(rob_loc)] = 76
        
        robot_in_map = map_copy

        return robot_in_map

    def range_search(self, position, r, points):
        """
        找到指定位置为中心、指定半径范围内的点

        nvar: position 的维数，即坐标的维度数 一般是2
        r2: 范围半径的平方
        s: 用于存储每个点到 position 的平方距离

        inrange_points: 找到的范围中的点
        """
        nvar = position.shape[0]
        r2 = r ** 2
        s = 0

        for d in range(0, nvar):
            s += (points[:, d] - position[d]) ** 2  

        # 找到所有距离平方小于等于 r2 的点的索引，即在范围 r 内的点
        idx = np.nonzero(s <= r2)

        # 将 idx 转换为一维数组，并使用它从 points 中提取范围内的点
        idx = np.asarray(idx).ravel()
        # print(idx.shape)
        inrange_points = points[idx, :]

        return inrange_points

    def local_map(self, robot_location, map_glo, map_size, local_size):
        """
        从全局地图 (map_glo) 中提取出以机器人当前位置为中心的一个局部区域
        该区域的大小由 local_size 参数控制
        """
        # 确定局部地图的初始边界
        minX = robot_location[0] - local_size
        maxX = robot_location[0] + local_size
        minY = robot_location[1] - local_size
        maxY = robot_location[1] + local_size

        # 调整边界，确保局部地图在全局地图范围内
        if minX < 0:
            maxX = abs(minX) + maxX
            minX = 0
        if maxX > map_size[1]:
            minX = minX - (maxX - map_size[1])
            maxX = map_size[1]
        if minY < 0:
            maxY = abs(minY) + maxY
            minY = 0
        if maxY > map_size[0]:
            minY = minY - (maxY - map_size[0])
            maxY = map_size[0]

        minY = int(minY)
        maxY = int(maxY)
        minX = int(minX)
        maxX = int(maxX)
        # 从全局地图中提取局部区域
        map_loc = map_glo[minY:maxY][:, minX:maxX]

        return map_loc

    def take_action(self, action_index, robot_position):
        """
        根据给定的动作索引选择对应的动作，并更新机器人的位置
        """
        move_action = self.action_space[action_index, :]
        robot_position[0] = np.round(robot_position[0] + move_action[0])
        robot_position[1] = np.round(robot_position[1] + move_action[1])

    def collision_check(self, start_point, end_point, map_size, map_glo):
        x0, y0 = start_point.round()
        x1, y1 = end_point.round()
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        error = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx *= 2
        dy *= 2

        # 创建一个 1x2 的 NumPy 数组 coll_points，初始值为 -1，用于存储碰撞点的坐标
        # 如果路径上没有碰撞，它的值不会被更新
        coll_points = np.ones((1, 2), np.uint8) * -1

        while 0 <= x < map_size[1] and 0 <= y < map_size[0]:
            # k = map_glo.item(y, x)：获取当前位置在 map_glo 中的值
            # 如果值为 1，则表示当前位置是障碍物
            k = map_glo.item(y, x)
            if k == 1:
                coll_points.itemset((0, 0), x)  # 记录碰撞点的x坐标
                coll_points.itemset((0, 1), y)  # 记录碰撞点的y坐标
                break

            if x == end_point[0] and y == end_point[1]:
                break

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        
        # 计算 coll_points 数组的总和
        # 如果没有碰撞，coll_points 中的值保持为 -1 -> 因此总和为 -2
        if np.sum(coll_points) == -2:
            coll_index = False
        else:
            coll_index = True

        return coll_points, coll_index

    def free_points(self, op_map):
        """
        用于从占用地图()中提取出所有值为255的点-即可通行的地图区域
        """    
        index = np.where(op_map == 255)
        free = np.asarray([index[1], index[0]]).T

        return free

    def nearest_free(self, tree, point):
        """
        用于在给定的树结构(tree)中找到距离指定点(point)最近的一个可行位置
        通常用于机器人路径规划和碰撞检测等任务，其中需要找到某个点的最近可行位置
        """

        # 确保 point 被转换为至少二维数组
        # 这意味着即使输入的是一个一维的坐标数组（例如 [x, y]），它也会被转化为二维形式，形状为 (1, 2)
        pts = np.atleast_2d(point)

        # tree.query 是 KDTree 或类似树结构提供的方法，用于查询给定点（pts）的最近邻。
        # tree.query(pts) 会返回两个值：
        # 第一个值：距离数组，包含查询点到最近邻的距离。
        # 第二个值：索引数组，包含查询点到最近邻的索引。
        # 在这行代码中，[1] 获取了索引数组，表示查询点 pts 在树中最近邻的索引。
        index = tuple(tree.query(pts)[1])

        # tree.data 是存储树中所有数据点的属性
        # 通过查询返回的 index，我们可以找到与输入点 point 最近的点
        nearest = tree.data[index]

        return nearest

    def get_reward(self, old_op_map, op_map, coll_index):
        """
        计算reward
        """
        if not coll_index: # 没有发生碰撞
            # 计算两个地图之间自由空间的差异
            # 如果当前地图中的自由空间比之前多，说明机器人探索了新的区域
            reward = float(np.size(np.where(op_map == 255)) - 
                           np.size(np.where(old_op_map == 255))) / 14000
            # 如果机器人增加了自由空间（即进行了有效的探索），则奖励值为正
            # 最大不超过 1
            if reward > 1:
                reward = 1
        else: # 如果发生了碰撞（coll_index 为 True），则返回奖励 -1
            reward = -1

        return reward

    def unique_rows(self, a):
        a = np.ascontiguousarray(a)
        unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
        result = unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
        result = result[~np.isnan(result).any(axis=1)]
        return result

    def astar_path(self, weights, start, goal, allow_diagonal=True):
        temp_start = [start[1], start[0]]
        temp_goal = [goal[1], goal[0]]
        temp_weight = (weights < 150) * 254 + 1
        # For the heuristic to be valid, each move must cost at least 1.
        if temp_weight.min(axis=None) < 1.:
            raise ValueError("Minimum cost to move must be 1, but got %f" % (
                temp_weight.min(axis=None)))
        # Ensure start is within bounds.
        if (temp_start[0] < 0 or temp_start[0] >= temp_weight.shape[0] or
                temp_start[1] < 0 or temp_start[1] >= temp_weight.shape[1]):
            raise ValueError("Start lies outside grid.")
        # Ensure goal is within bounds.
        if (temp_goal[0] < 0 or temp_goal[0] >= temp_weight.shape[0] or
                temp_goal[1] < 0 or temp_goal[1] >= temp_weight.shape[1]):
            raise ValueError("Goal of lies outside grid.")

        height, width = temp_weight.shape
        start_idx = np.ravel_multi_index(temp_start, (height, width))
        goal_idx = np.ravel_multi_index(temp_goal, (height, width))

        path = astar(
            temp_weight.flatten(), height, width, start_idx, goal_idx, allow_diagonal,
        )
        return path

    def plot_all(self):
        """
        画图
        """
        plt.cla() # 清除当前的图形
        plt.imshow(self.robot_belief, cmap='gray') # 显示robot_belief(ob_map)作为灰度图像
        plt.axis((0, self.map_size[1], self.map_size[0], 0)) # 设置坐标轴范围

        # 绘制机器人历史路径（蓝色线）
        plt.plot(self.xPoint, self.yPoint, 'b', linewidth=2) 

        # 绘制机器人探索到的前沿（红色线）
        plt.plot(self.x2frontier, self.y2frontier, 'b', linewidth=2) 

        # 绘制机器人的当前位置（紫色圆点）
        plt.plot(self.robot_position[0], self.robot_position[1], 'mo', markersize=8)

        # 绘制机器人的起始位置（青色圆点）
        plt.plot(self.xPoint[0], self.yPoint[0], 'co', markersize=8)

        plt.pause(0.05) # 暂停一小段时间，允许图形更新
        
