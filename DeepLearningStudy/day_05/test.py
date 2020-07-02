from PIL import ImageDraw, Image
import torch

pre = torch.tensor([[1, 2, 3], [7, 5, 1]])
a = torch.tensor([[1, 2, 3], [7, 5, 1]])
mask = pre > 3
print(mask.nonzero())

imgpath = r"E:\BaiduNetdiskDownload\深度学习\celebA\CelebA\Img\img_celeba.7z\img_celeba"
lableBoxPath = r"E:\BaiduNetdiskDownload\深度学习\celebA\CelebA\Anno\list_bbox_celeba.txt"
lableLandMaPath = r"E:\BaiduNetdiskDownload\深度学习\celebA\CelebA\Anno\list_landmarks_celeba.txt"
path = r"D:\pycharm_workspace\mtcnn_data\12\positive.txt"
# img = Image.open(imgpath + "\\" + "000002.jpg")
# # imgDraw = ImageDraw.Draw(img)
# # imgDraw.rectangle((159, 443, 126, 380), width=2, outline="red")
# cropimg = img.crop([159, 383,204, 443])
# # cropimg = img.crop([95, 71, 321, 384])
# # cropimg = cropimg.resize((33, 263))
# cropimg.save(r"D:\pycharm_workspace\mtcnn_data\123.jpg")
# cropimg.show()
# # img.show()
# exit()
#
# landMarkfile = open(path).readlines()
# print(len(landMarkfile))
# landmarkList = []
# for i in range(len(landMarkfile)):
#     column_list = landMarkfile[i].split()
#     landmarkList.append(column_list)
# print(landmarkList[0])
# exit()
# boxFile = open(lableBoxPath)
# count = 0
# for i, lable in enumerate(boxFile):
#     if i > 1:
#         lab = lable.split()
#         count += 1
#         if count == 10:
#             break
#         imagePath = imgpath + "\\" + lab[0]
#         img = Image.open(imagePath)
#         print(img.size)
#         imgDraw = ImageDraw.Draw(img)
#         x, y, w, h = int(lab[1]), int(lab[2]), int(lab[3]), int(lab[4])
#         imgDraw.rectangle((122, 102, 298, 337), width=2,
#                           outline="red")
#         imgDraw.rectangle((377, 356, 401, 535), width=3,
#                           outline="blue")
#         # if landmarkList[i][0] == lab[0]:
#         #     imgDraw.ellipse((int(landmarkList[i][1]) - 3, int(landmarkList[i][2]) - 3, int(landmarkList[i][1]) + 3,
#         #                      int(landmarkList[i][2]) + 3),
#         #                     fill='red')
#         #     imgDraw.ellipse((int(landmarkList[i][3]) - 3, int(landmarkList[i][4]) - 3, int(landmarkList[i][3]) + 3,
#         #                      int(landmarkList[i][4]) + 3), fill='red')
#         #
#         #     imgDraw.ellipse((int(landmarkList[i][5]) - 3, int(landmarkList[i][6]) - 3, int(landmarkList[i][5]) + 3,
#         #                      int(landmarkList[i][6]) + 3), fill='red')
#         #     imgDraw.ellipse((int(landmarkList[i][7]) - 3, int(landmarkList[i][8]) - 3, int(landmarkList[i][7]) + 3,
#         #                      int(landmarkList[i][8]) + 3), fill='red')
#         #     imgDraw.ellipse((int(landmarkList[i][9]) - 3, int(landmarkList[i][10]) - 3, int(landmarkList[i][9]) + 3,
#         #                      int(landmarkList[i][10]) + 3), fill='red')
#         # else:
#         #     print("error", i, landmarkList[i][0], lab[0])
#
#         img.show()
