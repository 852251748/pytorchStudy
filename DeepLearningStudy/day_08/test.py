# 创建网络
# 创建训练 采集样本 1000
# 开始训练 随机采样 放进经验池 经验池满了后 随机样本采用训练一会儿的网络给出的建议动作
# 增加探索 动态减小探索值

# import gym
# env = gym.make('CartPole-v0')
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         print(observation)
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break
# env.close()
import torch

a = [1, 2, 3, 4]
Qs = torch.tensor([[3, 4], [5, 9]])
action = torch.tensor([[0], [1]])

print(torch.gather(Qs, dim=1, index=action))
