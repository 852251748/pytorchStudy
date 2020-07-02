import torch
import torchaudio
import matplotlib.pyplot as plt

# filename = r"D:\BaiduNetdiskDownload\feiQFile\feiq\Recv Files\waves_yesno/0_0_0_0_1_1_1_1.wav"
filename = r"D:\Alldata\YESNO\waves_yesno/0_0_0_0_1_1_1_1.wav"
waveform, sample_rate = torchaudio.load(filename)

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.figure()
plt.plot(waveform.t().numpy())


specgram = torchaudio.transforms.MelSpectrogram()(waveform)

print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
p = plt.imshow(specgram.log2()[0,:,:].detach().numpy(), cmap='gray')
# specgram = torchaudio.transforms.Spectrogram()(waveform)
#
# print("Shape of spectrogram: {}".format(specgram.size()))
#
# plt.figure()
# plt.imshow(specgram.log2()[0,:,:].numpy(), cmap='gray')

plt.show()
