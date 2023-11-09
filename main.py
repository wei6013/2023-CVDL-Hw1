import sys
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchsummary
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mp
from torchvision import datasets, transforms
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtGui import QPixmap
from PIL import Image
import Ui_Hw1


class MyWindow(QMainWindow):
    
    def __init__(self):
        super(MyWindow, self).__init__()
        self.ui = Ui_Hw1.Ui_MainWindow()  # 創建UI對象
        self.ui.setupUi(self)
        self.ui.spinBox.setValue(1)
        self.ui.spinBox.setMinimum(1)
        self.ui.spinBox.setMaximum(15)
        self.ui.pushButton_1.clicked.connect(self.Load_folder)
        self.ui.pushButton_2.clicked.connect(self.Load_Image_L)
        self.ui.pushButton_3.clicked.connect(self.Load_Image_R)
        self.ui.pushButton_4.clicked.connect(self.Find_corner)
        self.ui.pushButton_5.clicked.connect(self.Find_Intrinsic)
        self.ui.pushButton_6.clicked.connect(self.Find_Extrinsic)
        self.ui.pushButton_7.clicked.connect(self.Find_Distortion)
        self.ui.pushButton_8.clicked.connect(self.Show_result)
        self.ui.pushButton_9.clicked.connect(self.Show_words_on_board)
        self.ui.pushButton_10.clicked.connect(self.Show_words_vertically)
        self.ui.pushButton_11.clicked.connect(self.Stereo_disparity_map)
        self.ui.pushButton_12.clicked.connect(self.Load_Image_L)
        self.ui.pushButton_13.clicked.connect(self.Load_Image_R)
        self.ui.pushButton_14.clicked.connect(self.Keypoints)
        self.ui.pushButton_15.clicked.connect(self.Match_Keypoints)
        self.ui.pushButton_16.clicked.connect(self.Load_Image)
        self.ui.pushButton_17.clicked.connect(self.Show_augmented_image)
        self.ui.pushButton_18.clicked.connect(self.Show_model_structure)
        self.ui.pushButton_19.clicked.connect(self.Show_accuracy_loss)
        self.ui.pushButton_20.clicked.connect(self.Inference)
        self.filestr=[]
        self.Image_L=""
        self.Image_R=""
        self.Image=""
        self.ImagePoints=[]
        self.ObjectPoints=[]
        self.disparity=[]
        self.img_augmented=list()
        self.img_name=list()

    def Load_folder(self):
        filestr = QFileDialog.getExistingDirectory(self,'開啟資料夾')
        self.filestr=[]
        if filestr:
            image_type=['.jpg','.jpeg','.png','.gif','.bmp']
            for root,dirs,files in os.walk(filestr):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_type):
                        image_path=os.path.join(root,file)
                        self.filestr.append(image_path)
            self.filestr = sorted(self.filestr, key=lambda x: int(os.path.basename(x).split('.')[0]))
            #print(self.filestr)
        #print(len(self.filestr))
                        
        
    def Load_Image_L(self): 
        self.Image_L=""
        filename,filetype = QFileDialog.getOpenFileName(self,'選擇左圖')
        if filename:
            self.Image_L=filename
            img=cv2.imread(filename)
            img=cv2.resize(img,(800,600))
            cv2.imshow('Img_Left',img)
            cv2.moveWindow('Img_Left',150,120)
            cv2.waitKey()
            cv2.destroyAllWindows()
    def Load_Image_R(self): 
        self.Image_R=""
        filename,filetype = QFileDialog.getOpenFileName(self,'選擇右圖')
        if filename:
            self.Image_R=filename
            img=cv2.imread(filename)
            img=cv2.resize(img,(800,600))
            cv2.imshow('Img_Right',img)
            cv2.moveWindow('Img_Right',950,120)
            cv2.waitKey()
            cv2.destroyAllWindows()
        
    def Find_corner(self):
        self.ImagePoints=list()
        self.ObjectPoints=list()
        width=11
        high=8
        objpoint=np.zeros((width*high,3),np.float32)
        objpoint[:,:2]=np.mgrid[0:width,0:high].T.reshape(-1, 2)
        #print(self.ObjectPoints)
        for i in range(len(self.filestr)):
            #print(self.filestr[i])
            img = cv2.imread(self.filestr[i])
            winname=str(i+1)+'.bmp'
            grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret,corners=cv2.findChessboardCorners(grayimg, (width, high), None)

            #print(corners)
            #print("ret",ret)
            #print("corners",corners)
            if ret:
                winSize = (5, 5)
                zeroZone = (-1, -1)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                self.ImagePoints.append(corners)
                self.ObjectPoints.append(objpoint)
                cv2.cornerSubPix(grayimg, corners, winSize, zeroZone, criteria)
                img=cv2.cvtColor(grayimg,cv2.COLOR_GRAY2RGB)
                cv2.drawChessboardCorners(img, (width, high), corners, ret)
                #print('hello')
                img=cv2.resize(img,(1000,800))
                cv2.imshow(winname,img)
                cv2.moveWindow(winname,150,120)
                cv2.waitKey(650)
                cv2.destroyAllWindows() 
        
    def Find_Intrinsic(self):
        width=11
        high=8
        #print('imagePoints:',self.ObjectPoints[0])
        #for i in range(len(self.ImagePoints)):
            #winname=str(i+1)+'.bmp'
            
        ret,mat_intri,cof_dist,v_rot,v_trans=cv2.calibrateCamera (self.ObjectPoints, self.ImagePoints,(width, high) ,None, None)
        if ret:
            print("Intrinsic:")
            print(mat_intri)
    
    def Find_Extrinsic(self):
        width=11
        high=8
        ret,mat_intri,cof_dist,v_rot,v_trans=cv2.calibrateCamera(self.ObjectPoints, self.ImagePoints,(width, high) ,None, None)
        num=self.ui.spinBox.value()-1 
        img = cv2.imread(self.filestr[num])
        grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        winSize = (5, 5)
        zeroZone = (-1, -1)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners=cv2.cornerSubPix(grayimg, self.ImagePoints[num], winSize, zeroZone, criteria)
        ret,v_rot,v_trans=cv2.solvePnP(self.ObjectPoints[num],corners,mat_intri,cof_dist)
        Rotation_matrix,_=cv2.Rodrigues(v_rot)
        Extrinsic_matrix=np.column_stack((Rotation_matrix,v_trans))
        print('Extrinsic {}.bmp:'.format(num+1))
        print(Extrinsic_matrix)
                    
    def Find_Distortion(self):
        width=11
        high=8
        #print('imagePoints:',self.ObjectPoints[0])
        #for i in range(len(self.ImagePoints)):
            #winname=str(i+1)+'.bmp'
            
        ret,mat_intri,cof_dist,v_rot,v_trans=cv2.calibrateCamera(self.ObjectPoints, self.ImagePoints,(width, high) ,None, None)
        if ret:
            print("Distortion:")
            print(cof_dist)
    
    def Show_result(self):
        width=11
        high=8
        ret,mat_intri,cof_dist,v_rot,v_trans=cv2.calibrateCamera (self.ObjectPoints, self.ImagePoints,(width, high) ,None, None) 
        for i in range(len(self.filestr)):
            img = cv2.imread(self.filestr[i])
            undistorted_img=cv2.undistort(img,mat_intri,cof_dist)
            concatenated_img=np.hstack([img,undistorted_img])
            concatenated_img=cv2.resize(concatenated_img,(1500,800))
            cv2.imshow('{}.bmp-Distorted(left) vs undistorted(right)'.format(i+1),concatenated_img)
            cv2.moveWindow('{}.bmp-Distorted(left) vs undistorted(right)'.format(i+1),150,120)
            cv2.waitKey(650)
            cv2.destroyAllWindows()  
        
    def Show_words_on_board(self):
        width=11
        high=8
        self.ImagePoints=list()
        self.ObjectPoints=list()
        objpoint=np.zeros((width*high,3),np.float32)
        objpoint[:,:2]=np.mgrid[0:width,0:high].T.reshape(-1, 2)
        #print(self.ObjectPoints)
        for i in range(len(self.filestr)):
            #print(self.filestr[i])
            img = cv2.imread(self.filestr[i])
            winname=str(i+1)+'.bmp'
            grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret,corners=cv2.findChessboardCorners(grayimg, (width, high), None)

            #print(corners)
            #print("ret",ret)
            #print("corners",corners)
            if ret:
                self.ImagePoints.append(corners)
                self.ObjectPoints.append(objpoint)
        offsets =[[7.0, 5.0, 0.0], [4.0, 5.0, 0.0], [1.0, 5.0, 0.0], [7.0, 2.0, 0.0], [4.0, 2.0, 0.0], [1.0, 2.0, 0.0]]
        ret,mat_intri,cof_dist,v_rot,v_trans=cv2.calibrateCamera (self.ObjectPoints, self.ImagePoints,(width, high) ,None, None)
        text=self.ui.textEdit.toPlainText()
        fs = cv2.FileStorage('alphabet_lib_onboard.txt', cv2.FILE_STORAGE_READ)
        if text:
            for j in range(len(self.filestr)):
                img = cv2.imread(self.filestr[j])
                for i in range(len(text)):
                    ch_mat = fs.getNode(text[i]).mat()
                    ch_mat=np.float32(ch_mat).reshape(-1,3)
                    #print('before',ch_mat)
                    ch_mat=ch_mat+offsets[i]
                    #print('after',ch_mat)
                    img_points, jac = cv2.projectPoints(ch_mat, v_rot[j], v_trans[j], mat_intri, cof_dist)
                    for k in range(len(img_points)//2):
                        pt1=tuple(map(int,img_points[2*k].ravel()))
                        pt2=tuple(map(int,img_points[2*k+1].ravel()))
                        img = cv2.line(img, pt1, pt2, (0, 0, 255), 5)
                        
                img=cv2.resize(img,(1000,800))
                cv2.imshow('AR {}.bmp'.format(j+1),img)
                cv2.moveWindow('AR {}.bmp'.format(j+1),150,120)
                cv2.waitKey(1000)
                cv2.destroyAllWindows() 
    
    def Show_words_vertically(self):
        width=11
        high=8
        self.ImagePoints=list()
        self.ObjectPoints=list()
        objpoint=np.zeros((width*high,3),np.float32)
        objpoint[:,:2]=np.mgrid[0:width,0:high].T.reshape(-1, 2)
        #print(self.ObjectPoints)
        for i in range(len(self.filestr)):
            #print(self.filestr[i])
            img = cv2.imread(self.filestr[i])
            winname=str(i+1)+'.bmp'
            grayimg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret,corners=cv2.findChessboardCorners(grayimg, (width, high), None)

            #print(corners)
            #print("ret",ret)
            #print("corners",corners)
            if ret:
                self.ImagePoints.append(corners)
                self.ObjectPoints.append(objpoint)
        offsets =[[7.0, 5.0, 0.0], [4.0, 5.0, 0.0], [1.0, 5.0, 0.0], [7.0, 2.0, 0.0], [4.0, 2.0, 0.0], [1.0, 2.0, 0.0]]
        ret,mat_intri,cof_dist,v_rot,v_trans=cv2.calibrateCamera (self.ObjectPoints, self.ImagePoints,(width, high) ,None, None)
        text=self.ui.textEdit.toPlainText()
        fs = cv2.FileStorage('alphabet_lib_vertical.txt', cv2.FILE_STORAGE_READ)
        if text:
            for j in range(len(self.filestr)):
                img = cv2.imread(self.filestr[j])
                for i in range(len(text)):
                    ch_mat = fs.getNode(text[i]).mat()
                    ch_mat=np.float32(ch_mat).reshape(-1,3)
                    #print('before',ch_mat)
                    ch_mat=ch_mat+offsets[i]
                    #print('after',ch_mat)
                    img_points, jac = cv2.projectPoints(ch_mat, v_rot[j], v_trans[j], mat_intri, cof_dist)
                    for k in range(len(img_points)//2):
                        pt1=tuple(map(int,img_points[2*k].ravel()))
                        pt2=tuple(map(int,img_points[2*k+1].ravel()))
                        img = cv2.line(img, pt1, pt2, (0, 0, 255), 5)
                        
                img=cv2.resize(img,(1000,800))
                cv2.imshow('AR {}.bmp'.format(j+1),img)
                cv2.moveWindow('AR {}.bmp'.format(j+1),150,120)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()          
    
    def Stereo_disparity_map(self):
        left_image=cv2.imread(self.Image_L)
        right_image=cv2.imread(self.Image_R)
        grayimg_left=cv2.cvtColor(left_image,cv2.COLOR_BGR2GRAY)
        grayimg_right=cv2.cvtColor(right_image,cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=256, blockSize=9)
        disparity = stereo.compute(grayimg_left, grayimg_right)
        self.disparity=disparity
        print(disparity.astype(np.float32)/16.0)
        disparity=cv2.resize(disparity,(800,600))
        cv2.imshow('Disparity', disparity)
        cv2.moveWindow('Disparity',550,120)
        cv2.setMouseCallback('Img_Left', self.on_mouse_click)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            left_img=cv2.imread(self.Image_L)
            right_img=cv2.imread(self.Image_R)
            disparity_value = self.disparity[y, x].astype(np.float32) / 16.0
            #print(disparity_value)
            while disparity_value > 60 or (disparity_value < 40 and disparity_value > 0):
                if disparity_value > 60:
                    disparity_value/=2
                elif disparity_value < 40:
                    disparity_value*=2.5
            #print(disparity_value)
            right_x = x-int((round(disparity_value)))
            right_y = y
            
            if disparity_value > 0:
                #print('click')
                #left_img=cv2.resize(left_img,(800,600))
                right_img=cv2.resize(right_img,(800,600))
                cv2.circle(right_img, (right_x, right_y), 3, (0, 255, 0), 5)
                cv2.imshow('Img_Right',right_img)
                cv2.moveWindow('Img_Right',950,120)
                #cv2.circle(left_img, (x, y), 3, (0, 255, 0), 5)
                #cv2.imshow('Img_Left',left_img)
                #cv2.moveWindow('Img_Left',150,120)
                print(f"left: ({x}, {y}), right: ({right_x}, {right_y}), Disparity: {disparity_value}")
            else:
                print("Fail",disparity_value)

    def Keypoints(self):
        sift = cv2.SIFT_create()
        img = cv2.imread(self.Image_L)
        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = sift.detectAndCompute(grayimg,None)
        out_img = cv2.drawKeypoints(grayimg, keypoints,None,color=(0,255,0))
        #print('succeed')
        out_img=cv2.resize(out_img,(800,600))
        cv2.imshow('SIFT_Keypoints', out_img)
        cv2.moveWindow('SIFT_Keypoints',950,120)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def Match_Keypoints(self):
        sift = cv2.SIFT_create()
        img_l = cv2.imread(self.Image_L)
        img_r = cv2.imread(self.Image_R)
        grayimg_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY)
        grayimg_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
        keypoints_l, descriptors_l = sift.detectAndCompute(grayimg_l,None)
        keypoints_r, descriptors_r = sift.detectAndCompute(grayimg_r,None)
        out_img_l = cv2.drawKeypoints(grayimg_l, keypoints_l,None,color=(0,255,0))
        out_img_r = cv2.drawKeypoints(grayimg_r, keypoints_r,None,color=(0,255,0))
        bf = cv2.BFMatcher()
        print('calculate knn')
        matches = bf.knnMatch(descriptors_l, descriptors_r, k=2)
        print('done')
        good_match=list()
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good_match.append([m])
        #out_img = cv2.drawMatchesKnn(grayimg_l,keypoints_l,grayimg_r,keypoints_r,good_match)
        print('draw')
        out_img = cv2.drawMatchesKnn(grayimg_l,keypoints_l,grayimg_r,keypoints_r,good_match,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        print('done')
        out_img=cv2.resize(out_img,(1200,900))
        cv2.imshow('Match_points', out_img)
        cv2.moveWindow('Match_points',550,120)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def Load_Image(self):
        self.Image=""
        filename,filetype = QFileDialog.getOpenFileName(self,'選擇圖片')
        if filename:
            self.Image=filename
            scene = QGraphicsScene()

            pixmap = QPixmap(filename)
            pixmap=pixmap.scaled(128, 128)
            pixmap_item = QGraphicsPixmapItem(pixmap)

            scene.addItem(pixmap_item)

            self.ui.graphicsView.setScene(scene)
    
    def Show_augmented_image(self):
        augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ]
        self.img_augmented=list()
        self.img_name=list()
        img_file=list()
        filestr = QFileDialog.getExistingDirectory(self,'開啟資料夾')
        if filestr:
            image_type=['.jpg','.jpeg','.png']
            for root,dirs,files in os.walk(filestr):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_type):
                        image_path=os.path.join(root,file)
                        img_file.append(image_path)

        for i in img_file:
            img=Image.open(i)
            name=os.path.basename(i)
            self.img_name.append(os.path.splitext(name)[0])
            for aug in augmentations:
                img=aug(img)
            #print('meow')
            self.img_augmented.append(img)
            #print(img_augmented)
        plt.figure(figsize=(12, 8))
        for i, image in enumerate(self.img_augmented):
            ax=plt.subplot(3, 3, i + 1)
            ax.set_title(self.img_name[i])
            plt.imshow(image)
            plt.axis('on')
            image_width, image_height = self.img_augmented[i].size
            ax.set_xticks([0,image_width])
            ax.set_yticks([0,image_height])
        plt.show()
    
    def Show_model_structure(self):
        vgg19_bn=models.vgg19_bn(num_classes=10,weights=False)
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #vgg19_bn.to(device)
        torchsummary.summary(vgg19_bn,(3,32,32))
    
    def Show_accuracy_loss(self):
        img = mp.imread('Accuracy_Loss.png')
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        """        train_loss = np.loadtxt('./train_loss.txt')
        val_loss = np.loadtxt('val_loss.txt')
        train_accuracy = np.loadtxt('train_accuracy.txt')
        val_accuracy = np.loadtxt('val_accuracy.txt')

        #Loss
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 1, 1)
        plt.plot(train_loss, label='Train Loss')
        plt.plot(val_loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss')

        #準確
        plt.subplot(2, 1, 2)
        plt.plot(train_accuracy, label='Train Accuracy')
        plt.plot(val_accuracy, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy(%)')
        plt.legend()
        plt.title('Accuracy')

        plt.tight_layout()
        plt.show()
        plt.savefig('Accuracy_Loss.png', dpi=300)
        """
        
    def Inference(self):
        model = torch.load('Wei_Vgg19-bn_weights.pth', map_location=torch.device('cpu'))
        model.eval()
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        image = Image.open(self.Image)
        image = transform(image)
        image = image.unsqueeze(0)
        #print("meow")
        # 用自己的模型預測
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs,dim=1)
        # 預測結果
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        class_dict = {i: classes[i] for i in range(len(classes))}
        predicted_class = classes[predicted.item()]

        #print(f'Predicted Class: {predicted_class}')
        self.ui.label.setText(f'Predict : {predicted_class}')
        
        plt.figure()
        plt.bar(range(10),probabilities[0])
        plt.xlabel('Class')
        plt.ylabel('Probability(%)')
        plt.xticks(range(10),[class_dict[i] for i in range(10)], rotation=45)
        plt.title("Probability of each class")
        plt.show()
        
        
        
        
        
if __name__=='__main__':
    app=QApplication(sys.argv)
    MainWindow=MyWindow()
    #ui=Ui_Hw1.Ui_MainWindow()
    #ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())