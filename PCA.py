import cv2
import numpy as np
#Vectorize
def vectorize(img):
    w,h=img.shape
    tmp=img.reshape(w*h)
    img_mean=np.mean(tmp)
    tmp=np.subtract(tmp,img_mean)
    tmp=tmp/np.linalg.norm(tmp)
    return tmp

#load image
def load_img(filepath,N_class,N_image):
    X=[]
    for i in range(0,N_class):
        for j in range(0,N_image):
            if (i+1)<10:
                str1='0'+str(i+1)
            else: str1=str(i+1)
            if (j+1)<10:
                str2='0'+str(j+1)
            else: str2=str(j+1)
            img=cv2.imread(filepath+str1+'_'+str2+'.png')
            gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            X_tmp=vectorize(gray)
            X.append(X_tmp)
    return np.transpose(X)
#PCA
def PCA(X,K):
    mean_total=np.mean(X,axis=1)
    for i in range(0,np.shape(X)[1]):
        X[:,i]=X[:,i]-mean_total
    U = np.dot(np.transpose(X),X)
    d,u = np.linalg.eig(U)
    w = np.dot(X,u)
    for i in range(0,np.shape(w)[1]):
        w[:,i]=w[:,i]/np.linalg.norm(w[:,i])
    return w[:,:K]
    
#Project to p-dimensional subspace
def project_to_subspace(W,X):
    mean_total=np.mean(X,axis=1)
    for i in range(0,np.shape(X)[1]):
        X[:,i]=X[:,i]-mean_total
    Y=np.dot(np.transpose(W),X)
    return Y
#Classify
def classify(Y_train,Y_test):
    k,n=np.shape(Y_test)
    result=np.zeros(n)
    for i in range(0,n):
        y=Y_test[:,i]
        tmp=np.zeros(n)
        for j in range(0,n):
            y_train=Y_train[:,j]
            tmp[j]=np.linalg.norm(y_train-y)
        result[i]=int(np.argmin(tmp))
    return result
# Compute Accuracy
def cal_accuracy(X,N_image):
    correct=0
    wrong=0
    for i in range(len(X)):
        
        if int(X[i]/N_image) == int(i/N_image):
            correct+=1
        else:
            wrong+=1
    return float(correct)/(float(correct)+float(wrong))
#load images
filepath_train='ECE661_2016_hw11_DB1/train/'
filepath_test='ECE661_2016_hw11_DB1/test/'
N_class=30
N_image=21
X_train=load_img(filepath_train,N_class,N_image)
X_test=load_img(filepath_test,N_class,N_image)
#training
accurate=[]
for K in range(1,20):
    print K
    W=PCA(X_train,K)
    #Project to p-dimensional subspace
    Y_train=project_to_subspace(W,X_train)
    Y_test=project_to_subspace(W,X_test)
    #Predict
    res=classify(Y_train,Y_test)
    #Calculate accuracy
    acc_rate=cal_accuracy(res,N_image)
    accurate.append(acc_rate)
    print acc_rate
np.savetxt('PCA.txt', accurate)
