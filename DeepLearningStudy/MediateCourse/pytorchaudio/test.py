# import torch
# from torch.nn import functional as F
#
# a = torch.rand((1, 54080))
# y = F.adaptive_avg_pool1d(a[None, ...], 50)
# print(y.shape)
import torchaudio
root = r"D:\Alldata"
dataset = torchaudio.datasets.SPEECHCOMMANDS(root, url='speech_commands_v0.02', folder_in_archive='SpeechCommands',
                                   download=False)
print(len(dataset))