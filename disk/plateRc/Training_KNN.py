#coding=utf-8
import numpy as np  
from sklearn import neighbors  
knn = neighbors.KNeighborsClassifier(n_neighbors=1)   
def roc(x):
    data =np.load('D:\\program\\plateRc\\offlineDataDoNotDelet\\trainingset features.npy')
    labels=np.load('D:\\program\\plateRc\\offlineDataDoNotDelet\\trainingset lables.npy')
    knn.fit(data, labels)
    result=knn.predict(x)
    c1=np.ndarray.tolist(result)
    index=c1[0].index(1)
    english_char_dic={0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
                      10:'A',11:'B',12:'C',13:'D',14:'E',15:'F',
                      16:'G',17:'H',18:'J',19:'K',20:'L',21:'M',
                      22:'N',23:'P',24:'Q',25:'R',26:'S',27:'T',
                      28:'U',29:'V',30:'W',31:'X',32:'Y',33:'Z'}
    result_char=english_char_dic[index]
    return result_char
