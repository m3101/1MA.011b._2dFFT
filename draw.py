import cv2 as cv2
import scipy.fft
import scipy.signal
import numpy as np

points = []

window="2DFFT"
cv2.namedWindow(window,cv2.WINDOW_GUI_NORMAL)

size=600

drawing = False
prev = -1
radius = 10
points = []
def click(event, x, y, flags, param):
    global drawing,points,prev,radius
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing = True
        prev=-1
    elif event==cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event==cv2.EVENT_MOUSEMOVE and drawing:
        x = min(size,max(0,x))
        if prev==-1:
            prev=x
            points.append((x,y))
        else:
            if ((x-points[-1][0])**2+(y-points[-1][1])**2)**.5>radius:
                pp,p = (np.array(points[-1]),np.array((x,y)))
                points.append(tuple((pp+(radius*(p-pp)/np.linalg.norm((p-pp)))).astype(int)))
                prev = x
cv2.setMouseCallback(window,click)

while 1:
    screen = np.zeros((size,size))

    for i in range(1,len(points)):
        cv2.line(screen,tuple(points[i-1]),tuple(points[i]),(255,255,255),2,cv2.LINE_AA)

    cv2.imshow(window,screen)

    k=cv2.waitKey(1)&0xff
    if k==ord('q'):
        break
    elif k==ord('s'):
        with open('DRAW.NP','wb') as o_f:
            np.save(o_f,np.array(points))