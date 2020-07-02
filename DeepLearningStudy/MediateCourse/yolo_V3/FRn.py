import torch
from torch import nn


class FilterResponseNormalization(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        """
        Input Variables:
        ----------------
            beta, gamma, tau: Variables of shape [1, C, 1, 1].
            eps: A scalar constant or learnable variable.
        """

        super(FilterResponseNormalization, self).__init__()
        self.beta = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.gamma = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.tau = nn.parameter.Parameter(
            torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.eps = nn.parameter.Parameter(torch.Tensor([eps]))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)

    def forward(self, x):
        """
        Input Variables:
        ----------------
            x: Input tensor of shape [NxCxHxW]
        """

        n, c, h, w = x.shape
        assert (self.gamma.shape[1], self.beta.shape[1], self.tau.shape[1]) == (c, c, c)

        # Compute the mean norm of activations per channel
        nu2 = x.pow(2).mean(dim=(2, 3), keepdim=True)
        # Perform FRN
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        # Return after applying the Offset-ReLU non-linearity
        return torch.max(self.gamma * x + self.beta, self.tau)


if __name__ == '__main__':
    x = torch.transpose(torch.transpose(torch.arange(2, 34.).reshape([1, 4, 4, 2]), 1, 3), 2, 3)

    print(x[0, 0, :, :])
    frn = FilterResponseNormalization(2)

    y = frn(x)
    print('output:', y[0, 0, :, :])

    loss = torch.sum(y)
    loss.backward()
    print(frn.tau.grad)
    print("loss:", loss)
