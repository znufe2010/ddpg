# 导入必要的库
import copy

import gym
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import chatgpt.DDPG_keras.Env as kearsEnv

# 定义一些超参数
num_states = 5 # 状态空间的维度
num_actions = 1 # 动作空间的维度
upper_bound = 1 # 动作空间的上界
lower_bound = 0 # 动作空间的下界
learning_rate = 0.001 # 学习率
gamma = 0.99 # 折扣因子
buffer_capacity = 100000 # 经验回放缓冲区的容量
batch_size = 64 # 训练时每批样本的数量
tau = 0.005 # 目标网络更新速率

# 定义一个噪声类，用于给Actor网络添加探索性噪声
class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # 使用Ornstein-Uhlenbeck过程生成噪声
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # 保持状态变量以便下次调用
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial.copy()
        else:
            self.x_prev = np.zeros_like(self.mean)

# 定义一个缓冲区类，用于存储和采样经验数据
class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # 缓冲区的容量和批量大小
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        # 初始化缓冲区中存储数据
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))
        # 记录缓冲区的使用量
        self.buffer_counter = 0

    # 记录新的经验数据
    def record(self, obs_tuple):
        # 设置索引为当前计数器值的位置
        index = self.buffer_counter % self.buffer_capacity
        # 将新的数据存入缓冲区
        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        # 增加计数器值
        self.buffer_counter += 1

    # 随机采样一批经验数据
    def sample(self):
        # 计算有效的采样范围
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # 随机选择一批索引
        batch_indices = np.random.choice(record_range, self.batch_size)

        # 返回对应的数据
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        return state_batch, action_batch, reward_batch, next_state_batch

# 定义一个网络类，用于构建和更新Actor和Critic网络
class Network:
    def __init__(self):
        # 初始化四个网络：Actor，Critic，目标Actor，目标Critic
        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        # 将目标网络的权重设置为原始网络的权重
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

    # 定义一个函数，用于构建Actor网络
    def get_actor(self):
        # 定义输入层（状态）
        inputs = layers.Input(shape=(num_states,))
        # 定义一个隐藏层，使用ReLU激活函数
        out = layers.Dense(32, activation="relu")(inputs)
        # 定义一个输出层，使用tanh激活函数，输出范围在[-1, 1]之间
        outputs = layers.Dense(1, activation="tanh")(out)
        # 将输出乘以动作空间的上界，得到最终的动作值
        outputs = outputs * upper_bound
        # 创建一个Keras模型，将输入和输出连接起来
        model = tf.keras.Model(inputs, outputs)
        return model

    # 定义一个函数，用于构建Critic网络
    def get_critic(self):
        # 定义两个输入层（状态和动作）
        state_input = layers.Input(shape=(num_states))
        action_input = layers.Input(shape=(num_actions))
        # 将状态和动作连接起来，形成一个输入向量
        inputs = layers.Concatenate()([state_input, action_input])
        # 定义一个隐藏层，使用ReLU激活函数
        out = layers.Dense(32, activation="relu")(inputs)
        # 定义一个输出层，输出一个标量值，表示状态-动作对的价值
        outputs = layers.Dense(1)(out)
        # 创建一个Keras模型，将输入和输出连接起来
        model = tf.keras.Model([state_input, action_input], outputs)
        return model

    # 定义一个函数，用于更新目标网络的权重
    def update_target(self, target_weights, weights):
        # 使用软更新的方法，即目标网络的权重等于原始网络的权重乘以更新速率加上目标网络的权重乘以（1-更新速率）
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))

# 定义一个代理类，用于实现DDPG算法的主要逻辑
class Agent:
    def __init__(self):
        # 初始化一个网络对象，用于创建和更新网络
        self.network = Network()
        # 初始化一个缓冲区对象，用于存储和采样经验数据
        self.buffer = Buffer(buffer_capacity, batch_size)
        # 初始化一个噪声对象，用于给动作添加探索性噪声
        self.noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(0.2) * np.ones(1))
        # 定义优化器，用于更新网络的权重
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate)
        # 定义一个函数，用于根据状态选择动作
    def policy(self, state):
        # 将状态转换为张量格式
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        # 使用Actor网络预测动作
        actions = self.network.actor_model(state)
        # 给动作添加噪声
        noise = self.noise()
        actions = actions.numpy() + noise
        # 将动作限制在合法的范围内
        actions = np.clip(actions, lower_bound, upper_bound)
        return actions[0]
    # 定义一个函数，用于实现网络的训练逻辑
    def train(self):
        # 从缓冲区中采样一批经验数据
        state_batch, action_batch, reward_batch, next_state_batch = self.buffer.sample()

        # 使用tf.GradientTape记录梯度信息
        with tf.GradientTape() as tape:
            # 使用目标Actor网络预测下一个状态的动作
            target_actions = self.network.target_actor(next_state_batch)
            # 使用目标Critic网络预测下一个状态-动作对的价值
            target_critic_values = self.network.target_critic([next_state_batch, target_actions])
            # 使用贝尔曼方程计算当前状态-动作对的目标价值
            target_values = reward_batch + gamma * target_critic_values
            # 使用Critic网络预测当前状态-动作对的价值
            critic_values = self.network.critic_model([state_batch, action_batch])
            # 计算Critic网络的均方误差损失
            critic_loss = tf.math.reduce_mean(tf.math.square(target_values - critic_values))

        # 计算Critic网络的梯度
        critic_grad = tape.gradient(critic_loss, self.network.critic_model.trainable_variables)
        # 使用优化器更新Critic网络的权重
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.network.critic_model.trainable_variables)
        )

        # 使用tf.GradientTape记录梯度信息
        with tf.GradientTape() as tape:
            # 使用Actor网络预测当前状态的动作
            new_actions = self.network.actor_model(state_batch)
            # 使用Critic网络评估当前状态-动作对的价值
            critic_values = self.network.critic_model([state_batch, new_actions])
            # 计算Actor网络的策略梯度损失，即最大化状态-动作对的价值
            actor_loss = -tf.math.reduce_mean(critic_values)

        # 计算Actor网络的梯度
        actor_grad = tape.gradient(actor_loss, self.network.actor_model.trainable_variables)
        # 使用优化器更新Actor网络的权重
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.network.actor_model.trainable_variables)
        )

        # 更新目标网络的权重，使其接近原始网络的权重
        self.network.update_target(self.network.target_actor.variables, self.network.actor_model.variables)
        self.network.update_target(self.network.target_critic.variables, self.network.critic_model.variables)

        return actor_loss, critic_loss

# 定义一个函数，用于模拟环境和代理的交互过程，并记录训练结果
def run(env, agent):
    # 初始化一个列表，用于存储每个回合的累积奖励
    total_reward_list = []
    avg_reward_list = []
    # 循环执行多个回合
    for ep in range(1000):
        # 创建一个列表，用于存储损失函数的值
        losses = []
        # 初始化一个变量，用于存储当前回合的累积奖励
        total_reward = 0
        # 重置环境，得到初始状态
        state, u = env.kears_reset()
        # 初始化一个变量，用于判断当前回合是否结束
        done = False
        step_num = 0
        trajectory = np.array(state)
        # 循环执行当前回合中的每一步
        while not done:
            step_num = step_num + 1
            # 使用代理的策略选择动作
            action = agent.policy(state)
            # 在环境中执行动作，得到下一个状态，奖励和结束标志
            next_state, reward, done, info = env.kears_step(env, state, action)
            if next_state[0] < 0 or next_state[1] < 0:  # 出界判断
                reward = reward - 100
            # print("next_state is {}".format(next_state))
            # 将经验数据存入缓冲区
            agent.buffer.record((state, action, reward, next_state))
            # 累加当前回合的奖励
            total_reward += reward
            # 更新当前状态
            state = copy.deepcopy(next_state)
            trajectory = np.vstack((trajectory, next_state))  # store state history
            # 如果缓冲区中有足够的数据，就开始训练网络
            if agent.buffer.buffer_counter >= batch_size:
                actor_loss, critic_loss = agent.train()
                losses.append((actor_loss, critic_loss))
            if step_num % 1000 == 0 and step_num > 0:
                env.draw_path(trajectory, env.qgoal[0:2], env.cylinder, env.start_point, env.cylinderR,
                          'Episode-{}-step-{}'.format(ep, step_num))
            if env.checkOb(next_state, env.cylinder) or next_state[0] < 0 or next_state[1] < 0:
                env.draw_path(trajectory, env.qgoal[0:2], env.cylinder, env.start_point, env.cylinderR,
                          'Episode-{}-ob'.format(ep))
                break
            if done:
                print("#########done###########")
                env.draw_path(trajectory, env.qgoal[0:2], env.cylinder, env.start_point, env.cylinderR, 'Episode-Done')
                break

        # 将当前回合的累积奖励存入列表
        total_reward_list.append(total_reward)
        avg_reward_list.append(np.round(total_reward/step_num, 2))
        # 打印当前回合的信息
        print("Episode {}: Total Reward = {}".format(ep, total_reward))

        if len(losses)>5:
            # 绘制损失函数的图形
            losses = np.array(losses)
            plt.plot(losses[:, 0], label='Q loss')
            plt.plot(losses[:, 1], label='pi loss')
            plt.legend()
            plt.title('DDPG-losses on {}'.format(ep))
            plt.xlabel('Updates')
            plt.ylabel('Loss')
            plt.savefig('./result/DDPG-loss-' + str(ep) + '.jpg', dpi=500)
            plt.show()
    # 绘制累积奖励的变化曲线
    plt.plot(total_reward_list)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title('DDPG Total Reward ')
    plt.show()

    plt.plot(avg_reward_list)
    plt.xlabel('Episode')
    plt.ylabel('Avg Avg')
    plt.title('DDPG Total Avg Reward ')
    plt.show()

env = kearsEnv.Env()
# 创建一个代理对象
agent = Agent()
# 调用run函数，开始模拟和训练过程
run(env, agent)

