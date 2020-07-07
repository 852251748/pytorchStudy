from deeplearnCode.Day4.Mtcnn.train import Train

if __name__ == '__main__':
    trainer12 = Train(r"E:\mtcnn_data", 24)
    trainer12(10000)
