import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from skimage.transform import resize


import Networks
import Agent

# select mode
TRAIN = True
PLOT = False

# training environment parameters
ACTIONS = 50  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 1e4  # timesteps to observe before training
EXPLORE = 2e6  # frames over which to anneal epsilon

REPLAY_MEMORY = 10000  # 回放内存的大小

BATCH = 64  # size of minibatch

FINAL_RATE = 0  # final value of dropout rate
INITIAL_RATE = 0.9  # initial value of dropout rate

TARGET_UPDATE = 25000  # update frequency of the target network


network_dir = "./cnn_" + str(ACTIONS)
if not os.path.exists(network_dir):
    os.makedirs(network_dir)

trained_model_weight="cnn100000.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# 定义权重复制函数
def copy_weights(target_model, model):
    target_model.load_state_dict(model.state_dict())

def start():
    # Initialize models and optimizer
    model = Networks.CNN(ACTIONS).to(device)
    target_model = Networks.CNN(ACTIONS).to(device)

    # 如果非训练模式则加载已训练网络
    if not TRAIN:
        checkpoint = torch.load(os.path.join(network_dir, trained_model_weight))
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Successfully loaded:", os.path.join(network_dir, trained_model_weight))

    # 定义损失函数和优化器    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = torch.nn.MSELoss()

    # 初始化一个回放内存 D，用于存储机器人与环境交互的历史数据。
    D = deque(maxlen=REPLAY_MEMORY)

    # 初始化训练环境
    robot_explo = Agent.Robot(0, TRAIN, PLOT)
    step_t = 0
    drop_rate = INITIAL_RATE
    total_reward = np.empty([0, 0])
    finish_all_map = False

    x_t = robot_explo.begin()
    # print(x_t.shape)    240x240
    x_t = resize(x_t, (84, 84))
    # print(x_t.shape)
    s_t = torch.tensor(x_t, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    a_t_coll = []

    while TRAIN and step_t <= EXPLORE:
        # 缩减 dropout 率
        if drop_rate > FINAL_RATE and step_t > OBSERVE:
            drop_rate -= (INITIAL_RATE - FINAL_RATE) / EXPLORE
        
        # 在不确定性的基础上，从策略网络 (model) 中选择一个动作，并且该过程不进行梯度计算
        model.eval()
        with torch.no_grad():
            # 将当前状态 s_t 输入到策略网络 model 中，获取模型的预测输出
            # 输出是对每个动作的预期回报（或者 Q 值）
            # 如果模型运行在 GPU 上，这会将计算结果转移到 CPU 上
            # 因为接下来的操作是 NumPy 操作，而 NumPy 只能在 CPU 上运行
            readout_t = model(s_t,drop_rate).cpu().numpy()[0]
        readout_t[a_t_coll] = None # 排除这些动作不再被选择
        # 找到剩余值中的最大值的索引
        # 这样就选出了一个最大 Q 值的动作，并避免选择已经排除的动作
        action_index = np.nanargmax(readout_t) 
        a_t = np.zeros([ACTIONS])
        # 选择了 action_index 这个动作
        # 即选择了一个具有最大 Q 值的动作
        a_t[action_index] = 1

        x_t1, r_t, terminal, complete, re_locate, collision_index, _ = robot_explo.step(action_index)
        x_t1 = resize(x_t1, (84, 84))
        s_t1 = torch.tensor(x_t1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        finish = terminal

        # # 存储转换
        D.append((s_t, action_index, r_t, s_t1, terminal))

        # 从记忆库中采样一个小批量进行训练
        if step_t > OBSERVE:
            # 更新目标网络
            if step_t % TARGET_UPDATE == 0:
                copy_weights(target_model, model)

            minibatch = random.sample(D, BATCH)
            batch_s, batch_a, batch_r, batch_s1, batch_done = zip(*minibatch)

            # 获取批量变量
            batch_s = torch.cat(batch_s).to(device)
            batch_a = torch.tensor(batch_a).unsqueeze(1).to(device)
            batch_r = torch.tensor(batch_r).to(device)
            batch_s1 = torch.cat(batch_s1).to(device)
            batch_done = torch.tensor(batch_done, dtype=torch.float32).to(device)

            # 计算目标
            with torch.no_grad():
                target_q_values = target_model(batch_s1,drop_rate).max(1)[0]
                y_batch = batch_r + GAMMA * target_q_values * (1 - batch_done)
            
            # 执行梯度下降
            model.train()
            optimizer.zero_grad()
            q_values = model(batch_s,drop_rate).gather(1, batch_a).squeeze()
            loss = criterion(q_values, y_batch)
            loss.backward()
            optimizer.step()

        step_t += 1
        total_reward = np.append(total_reward, r_t)

        # 保存进度
        if step_t in [2e3,2e4, 1e5,2e5, 5e5,1e6,2e6]:
            torch.save({"model_state_dict": model.state_dict()}, os.path.join(network_dir, "cnn"+str(step_t)+".pth"))

        print("TIMESTEP", step_t, "/ DROPOUT", drop_rate, "/ ACTION", action_index, "/ REWARD", r_t, "/ Terminal", finish, "\n")
        print(robot_explo.li_map)

        # 重置环境
        if finish:
            if complete:
                x_t = robot_explo.begin()
            if re_locate:
                x_t, re_locate_complete, _ = robot_explo.rescuer()
                if re_locate_complete:
                    x_t = robot_explo.begin()
            x_t = resize(x_t, (84, 84))
            s_t = torch.tensor(x_t, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            a_t_coll = []
            continue

        s_t = s_t1

    while not TRAIN and not finish_all_map:
        # 通过策略选择动作
        model.eval()
        with torch.no_grad():
            readout_t = model(s_t,drop_rate).cpu().numpy()[0]
        readout_t[a_t_coll] = None
        action_index = np.nanargmax(readout_t)
        a_t = np.zeros([ACTIONS])
        a_t[action_index] = 1

        # 运行选定动作并观察下一个状态和奖励
        x_t1, r_t, terminal, complete, re_locate, collision_index, finish_all_map = robot_explo.step(action_index)
        x_t1 = resize(x_t1, (84, 84))
        s_t1 = torch.tensor(x_t1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        finish = terminal

        step_t += 1
        print("TIMESTEP", step_t, "/ ACTION", action_index, "/ REWARD", r_t, "/ Q_MAX %e" % np.max(readout_t), "/ Terminal", finish, "\n")

        if finish:
            a_t_coll = []
            if complete:
                x_t = robot_explo.begin()
            if re_locate:
                x_t, re_locate_complete, finish_all_map = robot_explo.rescuer()
                if re_locate_complete:
                    x_t = robot_explo.begin()
            x_t = resize(x_t, (84, 84))
            s_t = torch.tensor(x_t, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            continue

        # 避免下次碰撞
        if collision_index:
            a_t_coll.append(action_index)
            continue
        a_t_coll = []
        s_t = s_t1

if __name__=="__main__":
    start()
    # if PLOT:
    #     plt.show()
