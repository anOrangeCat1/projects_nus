import os
import torch
import numpy as np
from skimage.transform import resize

import Networks
import Agent

# select mode
TRAIN = False
PLOT = True

# training environment parameters
ACTIONS = 50  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 1e4  # timesteps to observe before training
EXPLORE = 2e6  # frames over which to anneal epsilon

REPLAY_MEMORY = 10000  # 回放内存的大小

FINAL_RATE = 0  # final value of dropout rate
INITIAL_RATE = 0.9  # initial value of dropout rate

TARGET_UPDATE = 25000  # update frequency of the target network


network_dir = "./cnn_" + str(ACTIONS)
trained_model_weight="cnn200000.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

def start():
    # Initialize models and optimizer
    model = Networks.CNN(ACTIONS).to(device)

    checkpoint = torch.load(os.path.join(network_dir, trained_model_weight))
    model.load_state_dict(checkpoint["model_state_dict"])
    # print("Successfully loaded:", os.path.join(network_dir, trained_model_weight))

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