#######################################################################
# Name: sensor.py
# Simulate the sensor model of Lidar.
#######################################################################

import numpy as np

def collision_check(x0, y0, x1, y1, ground_truth, robot_belief):
    """ 
    Checks if line is blocked by obstacle 
    检查给定路径上是否存在障碍物，更新机器人信念图中的探测结果
    """
    # round() 函数的主要功能是将一个浮点数四舍五入到最接近的整数
    x0 = x0.round()
    y0 = y0.round()
    x1 = x1.round()
    y1 = y1.round()

    # dx和dy计算x和y方向的距离差，用于判断横纵轴的步进大小
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = int(x0), int(y0)
    error = dx - dy

    # x_inc和y_inc用于决定在x和y方向上的步进方向
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1

    # dx *= 2 和 dy *= 2 将 x 和 y 方向的差值加倍，以便在控制步进时更加灵活地调整误差
    # 确保路径在整数网格上尽量接近直线
    dx *= 2
    dy *= 2

    collision_flag = 0
    max_collision = 10

    while 0 <= x < ground_truth.shape[1] and 0 <= y < ground_truth.shape[0]:
        k = ground_truth.item(y, x)
        if k == 1 and collision_flag < max_collision:
            collision_flag += 1
            if collision_flag >= max_collision:
                break

        if k !=1 and collision_flag > 0:
            break

        if x == x1 and y == y1:
            break

        # itemset() 是一个 NumPy 数组的方法，用于在特定位置设置单个元素的值
        # 表示将 robot_belief 数组在 (y, x) 位置的值设置为 k
        robot_belief.itemset((y, x), k)

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx

    return robot_belief


def sensor_work(robot_position, sensor_range, robot_belief, ground_truth):
    """
    Expands explored region on map belief
    根据激光雷达的角度和探测范围，在信念图中扩展机器人探测的区域。
    """
    # sensor_angle_inc是每次扫描增加的角度，0.5度（转换为弧度）
    sensor_angle_inc = 0.5 / 180 * np.pi
    sensor_angle = 0
    x0 = robot_position[0]
    y0 = robot_position[1]
    while sensor_angle < 2 * np.pi:
        x1 = x0 + np.cos(sensor_angle) * sensor_range
        y1 = y0 + np.sin(sensor_angle) * sensor_range
        robot_belief = collision_check(x0, y0, x1, y1, ground_truth, robot_belief)
        sensor_angle += sensor_angle_inc
    return robot_belief
