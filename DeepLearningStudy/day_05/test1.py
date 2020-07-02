from PIL import ImageDraw, Image
#
imgpath = r"E:\BaiduNetdiskDownload\深度学习\celebA\CelebA\Img\img_celeba.7z\img_celeba"
lableBoxPath = r"E:\BaiduNetdiskDownload\深度学习\celebA\CelebA\Anno\list_bbox_celeba.txt"
#
boxFile = open(lableBoxPath)
for i, lable in enumerate(boxFile):
    if i > 1:
        lab = lable.split()
        imagePath = imgpath + "\\" + lab[0]
        img = Image.open(imagePath)

        imgDraw = ImageDraw.Draw(img)

        x, y, w, h = int(lab[1]), int(lab[2]), int(lab[3]), int(lab[4])
        # imgDraw.rectangle((int(lab[1]), int(lab[2]), int(lab[1]) + int(lab[3]), int(lab[2]) + int(lab[4])), width=3,
        #                   outline="blue")
        imgDraw.rectangle((int(x + 0.12 * w), int(y + 0.1 * h), int(x + 0.9 * w), int(y + 0.85 * h)), width=2,
                                                    outline="red")
        # ox1, oy1, ox2, oy2 = (int(lab[1]) - 0) / w1, int(lab[2]) / h1, (int(lab[1]) + int(lab[3])) / w1, (
        #         int(lab[2]) + int(lab[4])) / h1
        # print(ox1, oy1, ox2, oy2)
        # img = img.resize((100, 100))
        # crop = img.crop((int(ox1 * 100), int(oy1 * 100), int(ox2 * 100), int(oy2 * 100)))
        # crop.save(r"E:\BaiduNetdiskDownload\深度学习\celebA\CelebA\Img\img_celeba.7z\img_celeba\ddsds.jpg")
        # imgDraw = ImageDraw.Draw(img)
        # imgDraw.rectangle((int(ox1 * 100), int(oy1 * 100), int(ox2 * 100), int(oy2 * 100)), width=3,
        #                   outline="blue")
        img.show()

# from day_05.net import *
# import torch
#
# net = PNet()
# net.load_state_dict(torch.load("./pnet.pt"))
#
# print(net)
