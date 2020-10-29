import cv2
import numpy as np
import time
from numba import jit
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as pl
from math import sqrt,cos,sin,radians
from skimage.transform import rescale, resize, downscale_local_mean

frearr=np.zeros([1,180])
t=np.array([int(0/2),int(15/2)])
label=np.zeros([768,1024])
pt=np.array([[int(345/2),int(15/2)],[int(255/2),int(285/2)]]) #順時針
sa=0
sb=0
def setsec(i):
    global pt,sa,sb
    if i==0:
        pt[0]=[int(255/2),int(285/2)]
        pt[1]=[int(255/2),int(285/2)]
        sa=int(pt[0,1]-pt[0,0])/2
        sb=int(pt[1,1]-pt[1,0])/2
    elif i ==1:
        pt[1]=[int(255/2),int(285/2)]
        pt[0]=[int(75/2),int(105/2)]
        sa=int(pt[0,1]-pt[0,0])/2
        sb=int(pt[1,1]-pt[1,0])/2
    elif i ==2:
        pt[1]=[int(225/2),int(315/2)]
        pt[0]=[int(75/2),int(105/2)]
        sa=int(pt[0,1]-pt[0,0])/2
        sb=int(pt[1,1]-pt[1,0])/2
    elif i ==3:
        pt[1]=[int(225/2),int(315/2)]
        pt[0]=[int(45/2),int(135/2)]
        sa=int(pt[0,1]-pt[0,0])/2
        sb=int(pt[1,1]-pt[1,0])/2
    elif i ==4:
        pt[0]=[int(235/2),int(305/2)]
        pt[1]=[int(235/2),int(305/2)]
        # sa=int(pt[0,1]-pt[0,0])/2
        # sb=int(pt[1,1]-pt[1,0])/2
        p=0
    elif i==5:
        # sa=90
        # sb=0
        # pt[0]=[int(270/2),int(90/2)]
        # pt[1]=[int(270/2),int(90/2)]
        p=0
    elif i==6:
        pt[0]=[int(315/2),int(45/2)]
        pt[1]=[int(255/2),int(285/2)]
        sa=int(pt[0,1]-pt[0,0])/2
        sb=int(pt[1,1]-pt[1,0])/2

def ini(x,y,h):
    global label,frearr
    for i in range(180):
        frearr[0,i]=len(h[h==i])
    label=np.zeros([x,y])  
# def distance(border,h,s,i):
#     hue=np.copy(h)
#     mask=False
#     for k in border:
#         mask=mask | ((hue>k[0])&(hue<k[1]))
#     hue[mask]=0
#     f=pt[0,0]
#     a=list(np.abs(hue[~mask]-(f+i)))
#     f=pt[0,1]
#     b=list(np.abs(hue[~mask]-(f+i)))
#     f=pt[1,0]
#     c=list(np.abs(hue[~mask]-(f+i)))
#     f=pt[1,1]
#     d=list(np.abs(hue[~mask]-(f+i)))
#     x=np.minimum(a,b)
#     y=np.minimum(c,d)
#     v=np.minimum(x,y)*s[~mask]/255
#     # res=s*hue
#     return sum(v)

def distance1(h,i,s):
    global label
    [x,y]=h.shape
    label=np.zeros([x,y])
    h=h.astype(np.int16)
    hue=np.copy(h)
    border=[]
    s1b1=(pt[0,0]+i)%180
    s1b2=(pt[0,1]+i)%180
    s2b1=(pt[1,0]+i)%180
    s2b2=(pt[1,1]+i)%180
    dis1=s1b2-s1b1
    dis2=s2b2-s2b1
    if dis1<0 and dis2<0:
        dis1=(dis1+180)%180
        dis2=(dis2+180)%180
        border=[[0,s1b2],[s1b1,180]]
    elif s1b1==s2b1 and s1b2==s2b2:
        border=[[s2b1,s2b2]]
    elif dis1<0:
        dis1=(dis1+180)%180
        border=[[0,s1b2],[s2b1,s2b2],[s1b1,180]]
    elif dis2<0:
        dis2=(dis2+180)%180
        border=[[0,s2b2],[s1b1,s1b2],[s2b1,180]]
    else:
        border=[[s1b1,s1b2],[s2b1,s2b2]]
#    acc=distance(border,h,s,i)
    acc=0
    for i in border:
         mask=(hue>i[0])&(hue<i[1])
         acc+=-len((h[mask]))

    label=label1(hue,i,s1b1,s1b2,s2b1,s2b2)
    diff=calE2(s1b2,s2b1,s,label,h)
    diff+=calE2(s2b2,s1b1,s,label,h)
    return 0.1*acc+30*diff
    
def label1(h,i,s1b1,s1b2,s2b1,s2b2):
    lab1=-1
    lab2=-2
    [x,y]=h.shape
    label=np.zeros([x,y])
    h=h.astype(np.int16)
    hue=np.copy(h)
    dis1=s2b1-s1b2
    dis2=s1b1-s2b2
    if dis1<0:
        dis1=(dis1+180)%180
    elif dis2<0:
        dis2=(dis2+180)%180
    mid1=int(dis1/2)
    mid2=int(dis2/2)
#    if s1b2+mid1>s2b2+mid2:
#        label[:]=lab2
#        for i in range(s2b2+mid2,s1b2+mid1,1):
#            label[hue==i]=lab1
#    else:
#        label[:]=lab1
#        for i in range(s1b2+mid1,s2b2+mid2,1):
#            label[hue==i]=lab2
            
    minval=10000000000
    minidx1=(s1b2+mid1)%180
    minidx2=(s2b2+mid2)%180
    # if s2b1>s1b2:
    #     minidx1=s1b2+1+np.argmin(frearr[0,s1b2+1:s2b1])
    #     # minidx1=s1b2+1+np.argmin(frearr[0,int((s1b2+minidx1)/2):int((s2b1+minidx1)/2)])
    #     # for i in range(s1b2+1,s2b1,1):
    #     #     if len((hue[hue==i]))<minval:
    #     #         minval=len((hue[hue==i]))
    #     #         minidx1=i
    # else:
    #     if s1b2+1==180:
    #         s1b2=178 
    #     a=s1b2+1+np.argmin(frearr[0,(s1b2+1)%180:180])
    #     if s2b1==0:
    #         s2b1=1 
    #     b=np.argmin(frearr[0,0:s2b1])
    #     if a<b:
    #         minidx1=a
    #         # minidx1=s1b2+1+np.argmin(frearr[0,int((s1b2+minidx1)%180/2):int((s2b1+minidx1)%180/2)])
    #     else:
    #         minidx1=b
    #         # minidx1=s1b2+1+np.argmin(frearr[0,int((s1b2+minidx1)%180/2):int((s2b1+minidx1)%180/2)])

    #     # for i in range(s1b2+1,s1b2+dis1,1):
    #     #     if len((hue[hue==(i%180)]))<minval:
    #     #         minval=len((hue[hue==(i%180)]))
    #     #         minidx1=i
    # minval=10000000000
    # if s1b1>s2b2:
    #     minidx2=s2b2+1+np.argmin(frearr[0,(s2b2+1):s1b1])
    #     # minidx2=s2b2+1+np.argmin(frearr[0,int((s2b2+minidx2)/2):int((s1b1+minidx2)/2)])
    #     # for i in range(s2b2+1,s1b1,1):
    #     #     if len((hue[hue==i]))<minval:
    #     #         minval=len((hue[hue==i]))
    #     #         minidx2=i
    # else:
    #     if s2b2+1==180:
    #         s2b2=178 
    #     a=s2b2+1+np.argmin(frearr[0,(s2b2+1)%180:180])
    #     if s1b1==0:
    #         s1b1=1 
    #     b=np.argmin(frearr[0,0:s1b1])
    #     if a<b:
    #         minidx2=a
    #     else:
    #         minidx2=b
    #     # for i in range(s2b2+1,s2b2+dis2,1):
    #     #     if len((hue[hue==(i%180)]))<minval:
    #     #         minval=len((hue[hue==(i%180)]))
    #     #         minidx2=i
    step1=minidx1-s1b2
    step2=minidx2-s2b2
    if step1<0:
        step1=(step1+180)%180
    if step2<0:
        step2=(step2+180)%180
    for i in range(dis1):
        if i>step1:
            label[np.where(hue==(s1b2+i)%180)]=lab2
        else:
            label[np.where(hue==(s1b2+i)%180)]=lab1
            
    for i in range(dis2):
        if i>step2:
            label[np.where(hue==(s2b2+i)%180)]=lab1
        else:
            label[np.where(hue==(s2b2+i)%180)]=lab2
        
    return label
    
@jit(nopython=True)
def calE2(v1,v2,s,label,h): #兩扇形中間點的hue value +-10度
    h=h.astype(np.int16)
    close=[]
    if v2-v1<0:
        dis=(v2-v1+180)%180
    else:
        dis=(v2-v1)%180
        
    for i in range(10):
        close.append((v1+int(dis/2)+i)%180)
        close.append((v1+int(dis/2)-i)%180)
    [x,y]=h.shape
    diff=0
    for i in range(1,x-1,1):
        for j in range(1,y-1,1):
            if h[i,j] in close and label[i,j]!=0:
                if h[i-1,j] in close and label[i-1,j]!=0 and label[i-1,j]!=label[i,j]:
                    diff+=1*s[i,j]/255/np.abs(h[i,j]-h[i-1,j])
                if h[i+1,j] in close and label[i+1,j]!=0 and label[i+1,j]!=label[i,j]:
                    diff+=1*s[i,j]/255/np.abs(h[i,j]-h[i+1,j])
                if h[i,j-1] in close and label[i,j-1]!=0 and label[i,j-1]!=label[i,j]:
                    diff+=1*s[i,j]/255/np.abs(h[i,j]-h[i,j-1])
                if h[i,j+1] in close and label[i,j+1]!=0 and label[i,j+1]!=label[i,j]:
                    diff+=1*s[i,j]/255/np.abs(h[i,j]-h[i,j+1])
    return diff



if __name__== "__main__":
    t1=time.time()
    # cap = cv2.VideoCapture(0) #open camera
    # while(True):
    #     ret, frame = cap.read()
    #     cv2.imshow('frame', frame)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         cv2.imwrite('D:/test.jpg',frame)
    #         break
    # cap.release()
    # cv2.destroyAllWindows()
    img = cv2.imread('D:/testpic/3723.jpg')
    [x,y,z]=img.shape
    #==============for downsize=====================
    img1 = cv2.resize(img, dsize=(int(y/8), int(x/8)), interpolation=cv2.INTER_CUBIC)
    [x,y,z]=img1.shape
    #==============for downsize=====================
    hsv=np.zeros([x,y,z])
    targetlabel=np.zeros([x,y])
    hsv=cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    ini(x,y,h)
    print(np.max(h))
    mini=0
    best=0
    setsec(0)
    th=distance1(h,0,s)
    for j in range(7):
        setsec(j)
        for i in range(180):
            sectorval=distance1(h,i,s)
            if sectorval < th:
                targetlabel=np.copy(label)
                th=sectorval
                mini=i
                best=j
    t2=time.time()
    # =====================================back============================
    [x,y,z]=img.shape
    hsv=np.zeros([x,y,z])
    targetlabel=np.zeros([x,y])
    hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    ini(x,y,h)
    distance1(h,mini,s)
    targetlabel=np.copy(label)
    setsec(best)
    # =====================================back============================
    print('best sector:',best)
    print('process time:',t2-t1)
    print('rotate angle:',mini)
    shift_h = h
    cp=(int((t[0]+t[1])/2)+mini)%180
    w=2*np.pi*15/180
    w=w/2
    d=0
    
    for i in range(len(h)):
        for j in range(len(h[i])):
            if targetlabel[i,j]==-1:
                if pt[0,0]>pt[0,1]:
                    d=(pt[0,0]+int(180+pt[0,1]-pt[0,0])/2)%180
                else:
                    d=int(pt[0,1]+pt[0,0])/2
                cp=(d+mini)%180
            elif targetlabel[i,j]==-2:
                if pt[1,0]>pt[1,1]:
                    d=(pt[1,0]+int(180+pt[1,0]-pt[1,1])/2)%180
                else:
                    d=int(pt[1,1]+pt[1,0])/2
                cp=(d+mini)%180
            if targetlabel[i,j]!=0:
                shift_h[i,j]=(cp+w*(1-np.random.normal(0,w,1)*np.abs(h[i,j]-cp).astype(np.int16)))%180
    hsv[:,:,0]=shift_h
    convert = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    pl.figure(1)
    pl.imshow(convert)
    convert = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite('D:/testpic/test.jpg',convert)
    pl.show()
    
