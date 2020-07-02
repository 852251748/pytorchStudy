from PIL import ImageDraw, Image

imgpath = r"E:\BaiduNetdiskDownload\深度学习\celebA\CelebA\Img\img_celeba.7z\img_celeba"
lableBoxPath = r"E:\BaiduNetdiskDownload\深度学习\celebA\CelebA\Anno\list_bbox_celeba.txt"
lableLandMaPath = r"E:\BaiduNetdiskDownload\深度学习\celebA\CelebA\Anno\list_landmarks_celeba.txt"

lableLandMaPathP = r"D:\pycharm_workspace\mtcnn_data\300\positive.txt"
imgpathP = r"D:\pycharm_workspace\mtcnn_data\300\positive"

# lableLandMaPathP = r"D:\pycharm_workspace\1123\48\positive.txt"
# imgpathP = r"D:\pycharm_workspace\1123\48"
imgsize = 300
boxFile = open(lableLandMaPathP)
count = 0
for i, lable in enumerate(boxFile):
    lab = lable.split()
    imagePath = imgpathP + "\\" + lab[0]
    img = Image.open(imagePath)
    imgDraw = ImageDraw.Draw(img)

    # x1, y1, x2, y2 = max(float(lab[2]) * imgsize, 0), max(float(lab[3]) * imgsize, 0), max(float(lab[4]) * imgsize,
    #                                                                                        0), max(
    #     float(lab[5]) * imgsize, 0)
    x1, y1, x2, y2 = float(lab[2]) * imgsize, float(lab[3]) * imgsize, float(lab[4]) * imgsize + imgsize, float(
        lab[5]) * imgsize + imgsize

    # x1, y1, x2, y2 = float(lab[2]) * imgsize, float(lab[3]) * imgsize, float(lab[4]) * imgsize + imgsize, float(
    #     lab[5]) * imgsize + imgsize
    # print(x1, y1, x2, y2)
    imgDraw.rectangle((x1, y1, x2, y2), width=2,
                      outline="red")
    # imgDraw.rectangle((int(lab[1]), int(lab[2]), int(lab[1]) + int(lab[3]), int(lab[2]) + int(lab[4])), width=3,
    #                   outline="blue")

    imgDraw.ellipse(
        (max(float(lab[6]) * imgsize, 0) - 3, max(float(lab[7]) * imgsize, 0) - 3, max(float(lab[6]) * imgsize, 0) + 3,
         max(float(lab[7]) * imgsize, 0) + 3),
        fill='red')
    imgDraw.ellipse(
        (max(float(lab[8]) * imgsize, 0) - 3, max(float(lab[9]) * imgsize, 0) - 3, max(float(lab[8]) * imgsize, 0) + 3,
         max(float(lab[9]) * imgsize, 0) + 3), fill='red')

    imgDraw.ellipse((max(float(lab[10]) * imgsize, 0) - 3, max(float(lab[11]) * imgsize, 0) - 3,
                     max(float(lab[10]) * imgsize, 0) + 3,
                     max(float(lab[11]) * imgsize, 0) + 3), fill='red')
    imgDraw.ellipse((max(float(lab[12]) * imgsize, 0) - 3, max(float(lab[13]) * imgsize, 0) - 3,
                     max(float(lab[12]) * imgsize, 0) + 3,
                     max(float(lab[13]) * imgsize, 0) + 3), fill='red')
    imgDraw.ellipse((max(float(lab[14]) * imgsize, 0) - 3, max(float(lab[15]) * imgsize, 0) - 3,
                     max(float(lab[14]) * imgsize, 0) + 3,
                     max(float(lab[15]) * imgsize, 0) + 3), fill='red')

    img.show()

#
# landMarkfile = open(lableLandMaPath).readlines()
# landmarkList = []
# for i in range(len(landMarkfile)):
#     column_list = landMarkfile[i].split()
#     landmarkList.append(column_list)
# # print(landmarkList[0])
# # exit()
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
#         imgDraw = ImageDraw.Draw(img)
#         x, y, w, h = int(lab[1]), int(lab[2]), int(lab[3]), int(lab[4])
#         imgDraw.rectangle((int(x + 0.12 * w), int(y + 0.1 * h), int(x + 0.9 * w), int(y + 0.85 * h)), width=2,
#                           outline="red")
#         imgDraw.rectangle((int(lab[1]), int(lab[2]), int(lab[1]) + int(lab[3]), int(lab[2]) + int(lab[4])), width=3,
#                           outline="blue")
#         if landmarkList[i][0] == lab[0]:
#             imgDraw.ellipse((int(landmarkList[i][1]) - 3, int(landmarkList[i][2]) - 3, int(landmarkList[i][1]) + 3,
#                              int(landmarkList[i][2]) + 3),
#                             fill='red')
#             imgDraw.ellipse((int(landmarkList[i][3]) - 3, int(landmarkList[i][4]) - 3, int(landmarkList[i][3]) + 3,
#                              int(landmarkList[i][4]) + 3), fill='red')
#
#             imgDraw.ellipse((int(landmarkList[i][5]) - 3, int(landmarkList[i][6]) - 3, int(landmarkList[i][5]) + 3,
#                              int(landmarkList[i][6]) + 3), fill='red')
#             imgDraw.ellipse((int(landmarkList[i][7]) - 3, int(landmarkList[i][8]) - 3, int(landmarkList[i][7]) + 3,
#                              int(landmarkList[i][8]) + 3), fill='red')
#             imgDraw.ellipse((int(landmarkList[i][9]) - 3, int(landmarkList[i][10]) - 3, int(landmarkList[i][9]) + 3,
#                              int(landmarkList[i][10]) + 3), fill='red')
#         else:
#             print("error", i, landmarkList[i][0], lab[0])
#
#         img.show()
