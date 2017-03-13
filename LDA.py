import cv2
import numpy as np
from scipy.linalg import sqrtm
#Vectorize
def vectorize(img):
    w,h=img.shape
    tmp=img.reshape(w*h)
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
# Compute eigenvectors
def cal_eigenvector(X,N_image,N_class,K):
    ## Calculate Between-class scatter
    mean_total=np.mean(X,axis=1)
    mean=np.zeros((len(X),N_class))
    SB=np.zeros((len(X),len(X)))
    for i in range(N_class):
        tmp=X[:,i*N_image:(i+1)*(N_image)]
        mean[:,i]=np.mean(tmp,axis=1)-mean_total
    SB = np.dot(mean,np.transpose(mean))
    D,U = np.linalg.eig(np.dot(np.transpose(mean),mean))
    Y=np.dot(mean,U[:,:K])
    DB=np.dot(np.transpose(Y),np.dot(SB,Y))
    Z=np.dot(Y,np.linalg.inv(sqrtm(DB)))
    
    ## Calculate Within class scatter
    X_new=X.astype(np.float64)
    for i in range(N_class):
        for j in range(N_image):
            X_new[:,i*N_image+j] -=mean[:,i]
    tmp=np.dot(np.transpose(Z),X_new)
    DW=np.dot(tmp,np.transpose(tmp))
    V,U=np.linalg.eig(DW)
    W=np.dot(np.transpose(U),np.transpose(Z))
    return np.transpose(W)
#Project to p-dimensional subspace
def project_to_subspace(W,X):
    mean_total=np.mean(X,axis=1)
    Y=np.zeros((np.shape(W)[1],np.shape(X)[1]))
    for i in range(np.shape(X)[1]):
        Y[:,i]=np.dot(np.transpose(W),(X[:,i]-mean_total))
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
for K in range(1, 20):
    print K
    W=cal_eigenvector(X_train,N_image,N_class,K)
    #Project to p-dimensional subspace
    Y_train=project_to_subspace(W,X_train)
    Y_test=project_to_subspace(W,X_test)
    #Predict
    res=classify(Y_train,Y_test)
    #Calculate accuracy
    acc_rate=cal_accuracy(res,N_image)
    accurate.append(acc_rate)
    print acc_rate
np.savetxt('LDA.txt', accurate)
