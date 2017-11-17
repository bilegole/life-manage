#coding=utf-8
#test on distance 1m
#the size of picture is 3246x2448 photoed by iphone 5
'''
author:Crossroads
'''
import numpy as np 
import time 
import os
import datetime
import argparse  
import cv2

import math
#from sklearn import svm
#import scipy as sp
import Training_KNN
#import ANN_training_classification.get_feature
from cv2 import contourArea, waitKey, resize, imwrite, imread
from matplotlib.contour import ContourSet
from math import ceil
from matplotlib import pyplot as plt
from numpy import char
start_time=time.clock()
#-------------------------------------------粗定位阶段----------------------------------------------
#print'识别时间:%s'%datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
'''
颜色筛选阶段
在原始图像中筛选出蓝色像素点
''' 
#image = cv2.imread('c:\\plate\\3m.jpg')
image = cv2.imread('/root/git_project/HyperLPR/test/0.jpg')
images = image.copy()#获得当前彩色原图像的一份备份，留给后面使用
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#转换成为灰度图像
imagehsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)#将处于RGB色彩空间的图像转换到HSV，以获得更好的色彩提取效果 
color = [([100, 90, 90], [140, 255, 255])]#经过试验之后的蓝色范围，可以根据实际需要进行调整  

for (lower, upper) in color:  
    # 创建NumPy N维数组ndarray  
    lower = np.array(lower, dtype = "uint8")#蓝色颜色下限  
    upper = np.array(upper, dtype = "uint8")#蓝色颜色上限  
    # 根据阈值找到对应颜色  
    mask = cv2.inRange(imagehsv, lower, upper)  
    output = cv2.bitwise_and(imagehsv, imagehsv, mask = mask)#不太懂，按位与？？？
    plate = cv2.cvtColor(output, cv2.COLOR_HSV2BGR) #plate中是提取出的原图像中的蓝色区域（很可能就是车牌）
'''
------色块强化阶段------
经过上面的操作，我们已经得到了一张图，这张图里面包含：
       一些相距较远的离散的蓝色点（是周围环境中的小区域蓝色，不是我们需要的）
       一大群距离较近的蓝色点，它们的群簇近似成矩形（是车牌区域，是我们需要的）
    本阶段的目的，就是通过对该图进行形态学操作，将里面的蓝色色块进行强化，使得距离较近的蓝色像素点相互连接（那么离散的蓝色噪声点
    点将不能相互连接成为大面积区域，而车牌部分的蓝色点将相互连接成为一个矩形色块），等所有的点都连接成为大块封闭区域之后，我们就
    可以检测图像中封闭区域的轮廓（而刚才的散点是没有办法检测轮廓的），来找出面积最大的封闭区域，这个区域就是车牌区域。
'''
kernel_plate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6))#创建一个结构元素，66是实验值，可以进行适当修改
grayplate = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
equalizedHist_plate = cv2.equalizeHist(grayplate)#对提取到的蓝色区域做灰度均衡，以初步去除光照的影响
ret,plate_binary = cv2.threshold(equalizedHist_plate, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)#采用大津法进行自适应二值化
dilated = cv2.dilate(plate_binary, kernel_plate, iterations = 7)#对图像进行7次膨胀操作，使得微小的像素点相互连接成为色块
contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #找出这幅图中所有色块的外轮廓
'''
轮廓筛选阶段
''' 
#下面的步骤：根据色块的轮廓计算色块的面积，选取出其中面积最大的色块，这个色块便是车牌
imagecopy = image.copy()
contours_number = len(contours)#轮廓数量
if contours_number > 0:#判断蓝色轮廓是否存在
    max_contours = contours[0]#将最大轮廓初始化为第一个轮廓
    max_index = 0
    max_area = contourArea(contours[0])#将最大面积初始化为第一个轮廓的面积
    
    for i in range(contours_number):        
        contours_area_i = contourArea(contours[i]) #求取这个轮廓的面积
        cv2.drawContours(image, contours[i], -1, (0, 255, 0), 3)#画出每一个轮廓
        a, b, c, d = cv2.boundingRect(contours[i])#将这个轮廓转换成矩形元组并解包给四个元素
        cv2.rectangle(image, (a, b), (a + c, b + d), (255, 0, 0), 8)#画出每一个轮廓的矩形包围
        everminrec = cv2.minAreaRect(contours[i])#求得包含点集最小面积的矩形，，这个矩形是可以有偏转角度的，可以与图像的边界不平行
        box = cv2.cv.BoxPoints(everminrec)
        box = np.int0(box)#？？？？numpy.int0是什么？？？
        img = cv2.drawContours(image, [box], 0, (0, 0, 255), 2)
        if contours_area_i > max_area:#如果该轮廓是最大轮廓
            max_area = contourArea(contours[i])
            max_contours = contours[i]#将此轮廓代替为最大轮廓
            max_index = i
        else:pass
        
    #下面的步骤：将最大的色块从图像中分割出来    
    selected_rc_tuple = cv2.boundingRect(max_contours)
    x, y, z, o = selected_rc_tuple#x, y是左上角坐标，z，o是长和宽
    cv2.rectangle(imagecopy, (x, y), (x + z, y + o), (0, 0, 255), 3)
    is_plate=image_gray[selected_rc_tuple[1]:(selected_rc_tuple[1]+selected_rc_tuple[3]),selected_rc_tuple[0]:(selected_rc_tuple[0]+selected_rc_tuple[2])]
    is_color_plate=images[selected_rc_tuple[1]:(selected_rc_tuple[1]+selected_rc_tuple[3]),selected_rc_tuple[0]:(selected_rc_tuple[0]+selected_rc_tuple[2])]
else:print('no plates been detective!please adjust the distance between cars and camera!')#改成当轮廓面积小于xxx时输出，代表当前原图像里并没有发现车牌                    
    
    
#----------------------------------------------------细定位阶段--------------------------------------------
'''
细定位阶段需要完成的主要任务是：
在经过粗定位阶段之后，我们已经提取出了车牌的大致图像，但是这样提取到的车牌图像往往具有一些无用的边缘，具有一些明显的噪声点，而且可能是倾斜的。这些问题会对后期的
字符识别造成一定干扰，经过考虑之后，我们先进行车牌的倾斜矫正，所以我们需要将车牌上的一个个字符从车牌内“扣”出来，直接做识别。这就是细定位。
'''
#首先获取提取到的矩形轮廓的倾斜角度
rect = cv2.minAreaRect(contours[max_index])
plate_angle=rect[2]#这个参数代表提取出的矩形轮廓的倾斜角度，详情可以查看contours函数的返回值的含义
#print plate_angle 打印这个矩形轮廓的偏转角度
if plate_angle<-45:
    plate_angle=-(90-abs(plate_angle))
else:pass
#print('the rotation of plate is: ')
#print(plate_angle)#获得车牌的旋转角度


'''
倾斜矫正阶段
'''
#定义图像旋转函数对倾斜的车牌进行倾斜矫正
def rotate_about_center(image_being_rotate, angle, scale=1.):
    width = image_being_rotate.shape[1]
    height = image_being_rotate.shape[0]
    #print ('after transform angle: %s'%angle)
    #print('width is: %s'%width)
    #print('height is: %s'%height)
    r_angle = abs(np.deg2rad(angle))
    #print('r_angle is %s'%r_angle)
    new_width = (abs(np.sin(r_angle)*height) + abs(np.cos(r_angle)*width))*scale
    new_height = (abs(np.cos(r_angle)*height) + abs(np.sin(r_angle)*width))*scale
    #print('new width is: %s'%new_width)
    #print('new height is: %s'%new_height)
    rot_mat = cv2.getRotationMatrix2D((new_width*0.5, new_height*0.5), angle, scale)
    rot_move = np.dot(rot_mat, np.array([(new_width-width)*0.5, (new_height-height)*0.5,0]))
    rot_mat[0,2] += rot_move[0]
    rot_mat[1,2] += rot_move[1]
    return cv2.warpAffine(image_being_rotate, rot_mat, (int(math.ceil(new_width)), int(math.ceil(new_height))), flags=cv2.INTER_LANCZOS4)
if abs(plate_angle)>1.5:#1.5为实验值，实际过程中可以调整
    rotated_plate=rotate_about_center(is_plate, plate_angle)
else:rotated_plate=is_plate.copy()



'''
自适应二值化阶段
对旋转过后的车牌进行自适应二值化，以消除光照阴影的影响
'''
equalizedHist_rotateplate=cv2.equalizeHist(rotated_plate)
#此处需要一个自适应二值化算法，以克服光照阴影对于识别状况的影响
#这个算法的思想是：统计整张图中每个像素点的灰度值（0-255），将灰度值进行排序，取全体灰度值排列的62%处作为阈值。
#首先确定自适应二值化的阈值  
pixels=[]
for eachrow in equalizedHist_rotateplate:
    for eachpixel in eachrow:
        if eachpixel not in pixels: 
            pixels.append(eachpixel)
        else:pass
pixels.sort()
pixels_number=float(len(pixels))
thres_gray_index_float=(pixels_number/100)*62#此比例有较好的效果，可以继续改进
thres_gray_index=int(math.ceil(thres_gray_index_float))
thres_gray=pixels[thres_gray_index]

ret,thres_rotated_plate=cv2.threshold(equalizedHist_rotateplate,thres_gray,255,cv2.THRESH_BINARY)#利用计算出的阈值对图像进行二值化

'''
去除无用空洞阶段
这个阶段的主要任务是：车牌图像上往往存在一些小的空洞（与噪点不同，这些空洞也是噪声，但是比噪声面积大），这些空洞的存在将极大
影响后期提取到字符图像的质量，所以必须进行空洞填充。
'''
after_noise_plate=thres_rotated_plate.copy()
after_noise_plate2=after_noise_plate.copy()
#对图像的噪点进行填充，以便提高边界分割和字符识别的精度
#对带有噪点的二值化图像提取轮廓，筛选面积小于一定值的轮廓，并在另一幅二值图上对这些轮廓进行填充绘制，达到去噪目的
new_contours, new_hierarchy = cv2.findContours(thres_rotated_plate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
sum_noiserea=0
for every_c in new_contours:
    sum_noiserea+=cv2.contourArea(every_c)
threshold_area_down=2*(sum_noiserea/1000)#规定面积下限
#threshold_area_up=20*(sum_noiserea/1000)#规定面积上限
outrange_contours=[]
for tours in new_contours:
    tours_area=cv2.contourArea(tours)
    #if tours_area<threshold_area_down or tours_area>threshold_area_up:outrange_contours.append(tours)
    if tours_area<threshold_area_down:
        outrange_contours.append(tours)

'''
切除上下边框
车牌存在一些无用的上下边框，这些边框里存在着一些干扰：固定车牌的螺丝钉等等。需要进行切除。
'''
#下列函数通过统计每一行像素的黑白跳变次数来确定哪些是无用边缘。（字符区域的横向跳变值往往较多，而边缘区域的横向跳变往往教少）
def up_down_border_segment(img):
    width=img.shape[1]#获得图片长度
    height=img.shape[0]#获得图片宽度
    flag=False
    pixels=0
    y_up=0
    y_down=height-1
    for y in range(height/2):#纵向扫描范围：从顶端到一半
        cur_pixels=0
        for x in range(width):#横向扫描范围：整个横向范围
            if img[y,x]==255:cur_pixels+=1#统计黑色像素总数
        if not flag:#在flag为假false的状态下
            if (cur_pixels>pixels):pixels=cur_pixels
            if pixels-cur_pixels>0.6*pixels:
                flag=True
                pixels=cur_pixels
        else:
            if (cur_pixels<pixels):pixels=cur_pixels
            if cur_pixels-pixels>0.4*cur_pixels:
                y_up=y
                flag=False
    flag=False
    pixels=0
    for y in range(height-1,height/2-1,-1):
        cur_pixels=0
        for x in xrange(width):
            if img[y,x]==255:cur_pixels+=1
        if not flag:
            if (cur_pixels>pixels):pixels=cur_pixels
            if pixels-cur_pixels>0.6*pixels:
                flag=True
                pixels=cur_pixels
        else:
            if (cur_pixels<pixels):pixels=cur_pixels
            if cur_pixels-pixels>0.4*cur_pixels:
                y_down=y
                flag=False
    return y_up,y_down,img[y_up:y_down]
up,down,cut_up_down_plate=up_down_border_segment(after_noise_plate)

'''
切除上下边框之后给图像加上纯黑色上下边框
加上纯黑色上下边框的作用是为了便于后面字符的提取
'''
size=(1920,480)#这个大小是我们规定的大小，所有的车牌经过前面的预处理之后都被归一化到这个尺寸
change_size=cv2.resize(cut_up_down_plate,size)
blackcolor=(0,0,0)
add_border= cv2.copyMakeBorder(change_size, 100,100,0,0, cv2.BORDER_CONSTANT, value=blackcolor)#这个步骤即是加边框步骤


'''
提取非中文字符与中文字符
非中文字符（数字和字母）往往成长方形，对车牌图像提取所有的外轮廓，找出满足面积要求的长方形就是字符区域，而中文字符用提取轮廓的方法
往往效果很差。在下面的步骤中，我们使用模板匹配完成非中文字符的识别，使用sift特征匹配完成中文字符的识别。
'''

'''
非中文字符提取
在图像中寻找满足条件的轮廓提取并切割形成模板
'''
string_contours=[]
without_chinese=add_border[:,280:1920]
without_chinese2=without_chinese.copy()
contours_1, hierarchy_1 = cv2.findContours(without_chinese,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
all_contours_sum_area=0
picturesarea=1115200.0#经过对图片的测试，字符面积所占比集中在5-6之间
for everycontour in contours_1:
    this_area=cv2.contourArea(everycontour)
    all_contours_sum_area+=this_area    
for everycontour in contours_1:
    x,y,z,q=cv2.boundingRect(everycontour)
    rect_tuple=cv2.boundingRect(everycontour)
    rect_area=z*q
    #print float(float(rect_area)/picturesarea)*100  
    if 0.04<float(float(rect_area)/picturesarea)<0.09:
        string_contours.append(rect_tuple)
#print '检测到的可能是字符的区域有：%s个'%len(string_contours)
sortedlist=sorted(string_contours,key=lambda d:d[0])#对这些矩形元素按照坐标进行排序
#print sortedlist
if len(sortedlist)==6:   
    city_symbol=without_chinese2[sortedlist[0][1]:(sortedlist[0][1]+sortedlist[0][3]),sortedlist[0][0]:(sortedlist[0][0]+sortedlist[0][2])]
    first_char=without_chinese2[sortedlist[1][1]:(sortedlist[1][1]+sortedlist[1][3]),sortedlist[1][0]:(sortedlist[1][0]+sortedlist[1][2])]
    second_char=without_chinese2[sortedlist[2][1]:(sortedlist[2][1]+sortedlist[2][3]),sortedlist[2][0]:(sortedlist[2][0]+sortedlist[2][2])]
    third_char=without_chinese2[sortedlist[3][1]:(sortedlist[3][1]+sortedlist[3][3]),sortedlist[3][0]:(sortedlist[3][0]+sortedlist[3][2])]
    forth_char=without_chinese2[sortedlist[4][1]:(sortedlist[4][1]+sortedlist[4][3]),sortedlist[4][0]:(sortedlist[4][0]+sortedlist[4][2])]
    fifth_char=without_chinese2[sortedlist[5][1]:(sortedlist[5][1]+sortedlist[5][3]),sortedlist[5][0]:(sortedlist[5][0]+sortedlist[5][2])]
    #后期需要完善对数字1的分割步骤
    newsize=(12,24)
    city_symbol2=cv2.resize(city_symbol,newsize,interpolation=cv2.INTER_CUBIC)
    first_char2=cv2.resize(first_char,newsize,interpolation=cv2.INTER_CUBIC)
    second_char2=cv2.resize(second_char,newsize,interpolation=cv2.INTER_CUBIC)
    third_char2=cv2.resize(third_char,newsize,interpolation=cv2.INTER_CUBIC)
    forth_char2=cv2.resize(forth_char,newsize,interpolation=cv2.INTER_CUBIC)
    fifth_char2=cv2.resize(fifth_char,newsize,interpolation=cv2.INTER_CUBIC)
    cv2.namedWindow('city_symbol',cv2.WINDOW_NORMAL)
    cv2.imshow('city_symbol',city_symbol)
    cv2.namedWindow('first_char',cv2.WINDOW_NORMAL)
    cv2.imshow('first_char',first_char)
    cv2.namedWindow('second_char',cv2.WINDOW_NORMAL)
    cv2.imshow('second_char',second_char)
    cv2.namedWindow('third_char',cv2.WINDOW_NORMAL)
    cv2.imshow('third_char',third_char)
    cv2.namedWindow('forth_char',cv2.WINDOW_NORMAL)
    cv2.imshow('forth_char',forth_char)
    cv2.namedWindow('fifth_char',cv2.WINDOW_NORMAL)
    cv2.imshow('fifth_char',fifth_char) 
 
else:print'未检测完整的车牌区域，请调整拍照距离、角度或光照以再次识别.'
#非中文字符提取阶段结束

'''
提取中文字符
中文字符与非中文字符不同，中文字符存在左右结构和上下结构，直接对汉字进行轮廓提取将有可能提取不到完整的总轮廓，所以中文字符采取其他方式进行切割和识别
'''
#中文字符往往是车牌的第一个字符，取车牌的前面一小部分，即可以成功切割处中文字符
plate_1_chinese=rotate_about_center(is_color_plate,plate_angle)
plate_2_chinese=plate_1_chinese[up:down,:]
plate_3_chinese=cv2.resize(plate_2_chinese,size)
plate_4_chinese=plate_3_chinese[:,80:320]

'''
用sift检测中文字符
'''
feature_list=np.load('D:\\program\\plateRc\\offlineDataDoNotDelet\\chinese_model_sift_features.npy')#这是汉字sift特征库。
#车牌中可能出现的汉字一共有28个，将这28个汉字分别提取出sift特征，存在一个文件里，这个文件就叫汉字sift特征库。
#有了特征库，我们就可以将待识别汉字的特征与特征库中的特征进行比对，从而找出最相似的汉字。

#下面开始计算待识别汉字的sift特征，计算好之后与特征库进行比对
sift=cv2.SIFT()
keypoints_self, descirbe_self = sift.detectAndCompute(plate_4_chinese,None)
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5) 
search_params = dict(checks=50) 
flann = cv2.FlannBasedMatcher(index_params,search_params) 
max_number_matches=0
max_matches_index=0
for x in range(28): 
    matches = flann.knnMatch(descirbe_self,feature_list[x],k=2)   
    goodones=[] 
    for m,n in matches: 
        if m.distance < 0.75*n.distance: 
            goodones.append(m)
            if len(goodones)>max_number_matches:
                max_number_matches=len(goodones)
                max_matches_index=x+1
char_dictionary={1:'藏',2:'甘',3:'赣',4:'贵',5:'桂',6:'黑',7:'沪',
                 8:'吉',9:'冀',10:'津',11:'晋',12:'京',13:'鲁',14:'辽',
                 15:'蒙',16:'闽',17:'青',18:'琼',19:'陕',20:'苏',21:'皖',
                 22:'湘',23:'新',24:'渝',25:'豫',26:'粤',27:'云',28:'浙'}
got_chinese_char=char_dictionary[max_matches_index]
print '省份信息:%s'%got_chinese_char #汉字识别结束
#--------------------------------------------识别阶段-------------------------------------------------
'''
非中文字符待识别样本二值化阶段
'''
#全部转化为0和255的值
ret,city_symbol3=cv2.threshold(city_symbol2,125,255,cv2.THRESH_BINARY)
ret,first_char3=cv2.threshold(first_char2,125,255,cv2.THRESH_BINARY)
ret,second_char3=cv2.threshold(second_char2,125,255,cv2.THRESH_BINARY)
ret,third_char3=cv2.threshold(third_char2,125,255,cv2.THRESH_BINARY)
ret,forth_char3=cv2.threshold(forth_char2,125,255,cv2.THRESH_BINARY)
ret,fifth_char3=cv2.threshold(fifth_char2,125,255,cv2.THRESH_BINARY)
'''
特征提取阶段
这里将我们切割出的非中文字符块转化为向量（特征），用一个向量代表一张图片，便于后面的匹配
'''

#定义一个函数对图像特征进行计算（经过计算之后，每个图像被转化成一个72维的向量）
def get_feature(picture_name):
    char_maxtrix=picture_name.astype(np.float64)#转换其数据深度
    total_features=[]#嵌套列表，12层列表，每层6个数字，dtype=float64
    for row in range(0,23,2):
        features=[]#每两行产生一个6维的特征列表，存放进该列表中
        sum_list=char_maxtrix[row]+char_maxtrix[row+1]#两行的像素值之和，12维
        count_list=[]#两行的255像素数目，12维
        for c in sum_list:
            count=c/255.0
            count_list.append(count)
        for m in range(0,11,2):
            feature=count_list[m]+count_list[m+1]
            features.append(feature)
        total_features.append(features)
    featurearr=np.array(total_features)#一个numpy array，维数同total_features
    feature_reshape=featurearr.reshape(1,72)
    feature_back2list=np.ndarray.tolist(feature_reshape)
    return np.array(feature_back2list[0]).reshape(1,-1)

'''
对训练集进行特征提取并预测
'''
#使用KNN进行预测
#在计算出每个字符块的特征后，送入我们的KNN进行特征向量匹配。找到最相近的数字或字母进行返回。
x1=get_feature(city_symbol3)
char1=Training_KNN.roc(x1)
x2=get_feature(first_char3)
char2=Training_KNN.roc(x2)
x3=get_feature(second_char3)
char3=Training_KNN.roc(x3)
x4=get_feature(third_char3)
char4=Training_KNN.roc(x4)
x5=get_feature(forth_char3)
char5=Training_KNN.roc(x5)
x6=get_feature(fifth_char3)
char6=Training_KNN.roc(x6)
print '车牌信息:%s'%got_chinese_char,char1,'·',char2,char3,char4,char5,char6
end_time=time.clock()
time_cost=end_time-start_time
print '识别耗时:%sS'%time_cost
cv2.waitKey(0)

'''
目前需要解决的问题：
（1）需要距离腐蚀参数，在参数不确定的情况下将不能保证提取出完整的车牌，更不能实现好的字符识别
（2）需要对倾斜矫正函数进行进一步修改和细化，在倾斜没有被完全矫正的情况下，KNN将无法准确对字符识别
（3）搞清楚如何将训练好的ANN进行保存。
'''

