import math
import uuid
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from static_obstacle_environment import Obstacle                   # 几个静态障碍物环境坐标


class Env:
    def __init__(self):
        env = 'Obstacle4'
        self.cylinder = Obstacle[env].cylinder      # 圆柱体障碍物坐标
        self.cylinderR = Obstacle[env].cylinderR    # 圆柱体障碍物半径
        self.cylinderH  = Obstacle[env].cylinderH   # 圆柱体高度

        self.numberOfCylinder = 1
        self.stepNum = 5000

        self.qgoal = Obstacle[env].qgoal          # 目标点
        # self.x0 = Obstacle[env].x0                # 轨迹起始点
        #-------------一些参考参数可选择使用-------------#
        # robot parameter
        self.max_speed = 1.4  # [m/s]  # 最大速度
        # self.min_speed = -0.5  # [m/s]  # 最小速度，设置为可以倒车
        self.min_speed = 0  # [m/s]  # 最小速度，设置为不倒车
        self.max_yawrate = 80.0 * math.pi / 180.0  # [rad/s]  # 最大角速度
        self.max_accel = 0.2  # [m/ss]  # 最大加速度
        self.max_dyawrate = 80.0 * math.pi / 180.0  # [rad/ss]  # 最大角加速度
        self.v_reso = 0.01  # [m/s]，速度分辨率
        self.yawrate_reso = 0.1 * math.pi / 180.0  # [rad/s]，角速度分辨率
        self.dt = 0.1  # [s]  # 采样周期

        self.start_point = np.array([0.0, 0.0, math.pi / 2.0, 0.2, 0.0])
        self.u = np.array([0.2, 0.0]) # 速度空间
        self.predict_time = 3.0  # [s]  # 向前预估三秒
        self.to_goal_cost_gain = 0.5  # 目标代价增益
        self.speed_cost_gain = 1.0  # 速度代价增益
        self.robot_radius = 0.1  # [m]  # 机器人半径
        # -------------路径（每次getqNext会自动往path添加路径）---------#
        self.path = self.start_point.copy()
        # 仿真DQN 加入迭代
        self.FINAL_EPSILON = 0.0001
        # epsilon 的初始值，epsilon 逐渐减小。
        self.INITIAL_EPSILON = 0.1

    def kears_reset(self):        # 重置环境
        return self.start_point, self.u

    def motion(self, x, u, dt):
        """
        :param x: 位置参数，在此叫做位置空间
        :param u: 速度和加速度，在此叫做速度空间
        :param dt: 采样时间
        :return:
        """
        # 速度更新公式比较简单，在极短时间内，车辆位移也变化较大
        # 采用圆弧求解如何？
        x[0] += u[0] * math.cos(x[2]) * dt  # x方向位移
        x[1] += u[0] * math.sin(x[2]) * dt  # y
        x[2] += u[1] * dt  # 航向角
        x[3] = u[0]  # 速度v
        x[4] = u[1]  # 角速度w
        return x

    def computeGoal(self, x, goal):
        ox = goal[0]
        oy = goal[1]
        dx = x[0] - ox
        dy = x[1] - oy
        r = np.hypot(dx, dy)
        return r

    def checkGoal(self, x, goal, robot_radius=0.1):
        # check goal
        if math.sqrt((x[0] - goal[0]) ** 2 + (x[1] - goal[1]) ** 2) <= robot_radius:
            return True
        else:
            return False

    def kears_step(self, env, state, action):

        vr = self.calc_dynamic_window(state, env)  # 改为根据action是一个具体的值
        u, trajectory, min_cost = self.calc_final_input(state, env.u, vr, env, action)

        if min_cost == float("Inf"):
            # 说明遇到障碍物了
            reward = -100
        else:
            reward = min_cost
        # next_state = self.motion(state, u, env.dt)
        next_state = trajectory[-1]
        reward = -reward
        done = self.checkGoal(next_state, self.qgoal[0:2], self.robot_radius)
        return next_state, reward, done, ''

    def calc_dynamic_window(self, x, config):
        """
        位置空间集合
        :param x:当前位置空间,符号参考硕士论文
        :param config:
        :return:目前是两个速度的交集，还差一个
        """
        # 车辆能够达到的最大最小速度
        vs = [config.min_speed, config.max_speed,
              -config.max_yawrate, config.max_yawrate]
        # 一个采样周期能够变化的最大最小速度
        vd = [x[3] - config.max_accel * config.dt,
              x[3] + config.max_accel * config.dt,
              x[4] - config.max_dyawrate * config.dt,
              x[4] + config.max_dyawrate * config.dt]
        # 求出两个速度集合的交集
        vr = [max(vs[0], vd[0]), min(vs[1], vd[1]),
              max(vs[2], vd[2]), min(vs[3], vd[3])]
        return vr

    def calc_trajectory(self, x_init, v, w, config):
        """
        预测3秒内的轨迹
        :param x_init:位置空间
        :param v:速度
        :param w:角速度
        :param config:
        :return: 每一次采样更新的轨迹，位置空间垂直堆叠
        """
        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= config.predict_time:
            x = self.motion(x, [v, w], config.dt)
            trajectory = np.vstack((trajectory, x))  # 垂直堆叠，vertical
            time += config.dt
        return trajectory

    def calc_to_goal_cost(self, trajectory, goal, config):
        """
        计算轨迹到目标点的代价
        :param trajectory:轨迹搜索空间
        :param goal:
        :param config:
        :return: 轨迹到目标点欧式距离
        """
        # calc to goal cost. It is 2D norm.

        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        goal_dis = math.sqrt(dx ** 2 + dy ** 2)
        cost = config.to_goal_cost_gain * goal_dis

        return cost

    def calc_obstacle_cost(self, traj, ob, config):
        min_r = float("inf")  # 距离初始化为无穷大
        for ii in range(0, len(traj[:, 1])):
            for i in range(len(ob[:, 0])):
                # ob 是 立方体时，需考虑半径的问题， 下面是几何距离
                ox = ob[i, 0]
                oy = ob[i, 1]
                dx = traj[ii, 0] - ox
                dy = traj[ii, 1] - oy
                r = np.hypot(dx, dy)
                if r < self.robot_radius:
                    return float("Inf")  # collision
                if min_r >= r:
                    min_r = r
        return min_r

    def calc_final_input(self, x, u, vr, dwa, action):
        x_init = x[:]
        final_cost_list = []
        trajectory_list = []
        u_list = []
        # evaluate all trajectory with sampled input in dynamic window
        # v,生成一系列速度，w，生成一系列角速度
        for v in np.arange(vr[0], vr[1], dwa.v_reso):
            for w in np.arange(vr[2], vr[3], dwa.yawrate_reso):
                trajectory = self.calc_trajectory(x_init, v, w, dwa)
                to_goal_cost = self.calc_to_goal_cost(trajectory, dwa.qgoal, dwa)
                speed_cost = dwa.speed_cost_gain * (dwa.max_speed - trajectory[-1, 3])
                ob_cost = self.calc_obstacle_cost(trajectory, dwa.cylinder, dwa)
                # 评价函数多种多样，看自己选择
                final_cost = to_goal_cost + speed_cost + ob_cost
                final_cost_list.append(final_cost)
                trajectory_list.append(trajectory)
                u_list.append([v, w])

        action_index = int(len(u_list) * action[0])
        action_index = action_index - 1
        min_u = u_list[action_index]
        best_trajectory = trajectory_list[action_index]
        min_cost = final_cost_list[action_index]
        return min_u, best_trajectory, min_cost

    def checkOb(self, state, ob):
        """
        检查是否与障碍物相撞
        """
        ob_state = False
        for i in range(len(ob[:, 0])):
            # ob 是 立方体时，需考虑半径的问题， 下面是几何距离
            ox = ob[i, 0]
            oy = ob[i, 1]
            dx = state[0] - ox
            dy = state[1] - oy
            r = np.hypot(dx, dy)
            if r <= 0.1:  # 按边长求得的值
                ob_state = True
                break
        return ob_state

    def plot_arrow(self, x, y, yaw, length=0.5, width=0.1):

        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  head_length=1.5 * width, head_width=width)
        plt.plot(x, y)


    def draw_path(self, trajectory, goal, ob, x, ob_r, title='Done'):
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111, aspect='equal')
        plt.plot(x[0], x[1], "ro", markersize=10)
        # plt.plot(0, 0, "og")
        plt.plot(goal[0], goal[1], "gh", markersize=10)
        # plt.plot(ob[:, 0], ob[:, 1], "bs")
        for i in range(ob.shape[0]):
            ax1.add_patch(
                patches.Rectangle(
                    ob[i] - [0.5, 0.5],  # (x,y)
                    ob_r[i],  # width
                    ob_r[i],  # height
                )
            )

        # plt.axis("equal")
        plt.xlim(-2, 12)
        plt.ylim(-2, 12)
        plt.grid(True, color='lightgray')
        plt.title('DDPG-DWA on {}'.format(title))
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'r')
        self.plot_arrow(trajectory[-1][0], trajectory[-1][1], trajectory[-1][2])

        pic_id = uuid.uuid4()
        plt.savefig('./result/DDPG_keras_' + str(pic_id) + '.jpg', dpi=300)
        plt.show()

