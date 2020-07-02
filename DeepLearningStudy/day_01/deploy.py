from torch import jit
from day_01.net import *

if __name__ == '__main__':
    model = NetV3()
    model.load_state_dict(torch.load("./checkpoint/11.pkl"))
    input = torch.randn(1, 784)

    traced_script_module = jit.trace(model, input)
    traced_script_module.save("mnist.pt"     )
