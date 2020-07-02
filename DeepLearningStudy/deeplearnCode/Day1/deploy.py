from torch import jit
import torch
from deeplearnCode.Day1 import net

if __name__ == '__main__':
    input = torch.rand(1, 784)

    model = net.MlpNet()
    model.load_state_dict(torch.load(f"./param/19.pt"))
    pre_y = model(input)

    trace_script_module = jit.trace(model, input)
    trace_script_module.save("minist.pt")
