import torch, random
from torch import nn, optim
import gym


# from torch import

class QNet(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )


class Trainer:
    def __init__(self, exp_pool_size, explore):
        self.exp_pool_size = exp_pool_size
        self.explore = explore
        self.exp_pool = []
        self.env = gym.make('CartPole-v1')
        self.q_net = QNet()

        self.opt = optim.Adam(self.q_net.parameters())
        self.loss_fn = nn.MSELoss()

    def __call__(self, num):
        is_render = False
        avg = 0
        for epoch in range(num):
            # 采样
            R = 0
            state = self.env.reset()
            while True:
                if is_render: self.env.render()

                if len(self.exp_pool) >= self.exp_pool_size:
                    self.exp_pool.pop(0)
                    self.explore += 0.00001
                    if random.random() > self.explore:
                        action = self.env.action_space.sample()
                    else:
                        Qs = self.q_net(torch.tensor(state)[None, ...].float())
                        action = torch.argmax(Qs, dim=1).item()
                else:
                    action = self.env.action_space.sample()

                next_state, reward, done, info = self.env.step(action)
                R += reward
                self.exp_pool.append([state, action, reward, next_state, done])
                state = next_state
                if done:
                    avg = 0.95 * avg + 0.05 * R
                    print(avg, R)
                    if avg > self.env.spec.reward_threshold:
                        is_render = True
                    break

            # 训练
            if len(self.exp_pool) >= self.exp_pool_size:
                exps = random.choices(self.exp_pool, k=100)

                _state = torch.tensor([exp[0] for exp in exps]).float()
                _action = torch.tensor([[exp[1]] for exp in exps])
                _reward = torch.tensor([[exp[2]] for exp in exps])
                _next_state = torch.tensor([exp[3] for exp in exps]).float()
                _done = torch.tensor([[exp[4]] for exp in exps]).int()

                # 估计值
                Qs = self.q_net(_state)
                _Qs = torch.gather(Qs, dim=1, index=_action)

                # 实际值
                _next_max_Qs = torch.max(self.q_net(_next_state), dim=1, keepdim=True)[0].detach()
                act_Qs = _reward + 0.9 * _next_max_Qs * (torch.tensor(1) - _done)

                loss = self.loss_fn(_Qs, act_Qs)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()


if __name__ == '__main__':
    train = Trainer(1000, 0.9)
    train(100000)
