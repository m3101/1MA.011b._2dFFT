import cv2 as cv2
import scipy.fft
import scipy.signal
import numpy as np

with open("DRAW.NP",'rb') as i_f:
    points = np.load(i_f)

fwindow="2DFFT"
cv2.namedWindow(fwindow,cv2.WINDOW_GUI_NORMAL)
window="ORIG"
cv2.namedWindow(window,cv2.WINDOW_GUI_NORMAL)

size=600

num = 1
def fourier(pts,num):
    pts = pts.astype(float)
    #2d coords to imaginary
    pts = pts[:,0]+1j*pts[:,1]
    pts-= pts.mean()
    fft = scipy.fft.fftshift(scipy.fft.fft(pts))
    x   = scipy.fft.fftshift(scipy.fft.fftfreq(len(fft),1/len(pts)))
    ffft= fft.real**2+fft.imag**2
    #peaks,_ = scipy.signal.find_peaks(ffft)
    peaks = np.arange(pts.shape[0])
    print(len(peaks),"MAX PEAKS")
    peaks = [p[1] for p in sorted([(ffft[p],p) for p in peaks],reverse=True)[:num]]
    print(len(peaks),'peaks')
    return (fft[peaks]/len(pts),x[peaks])

peaks = np.array(fourier(points,num)).T
print(peaks)

centre = points.mean(axis=0).astype(int)

def pstime(divisions,peaks):
    time = np.linspace(0,1,divisions)
    #pts = np.cumsum(np.array([1]+(1/np.linspace(1,len(peaks),len(peaks))).tolist())[:,np.newaxis]*np.array([0*time]+[(peak[0]*np.exp(2*np.pi*1j*time*peak[1]))for peak in peaks]),axis=0).T
    pts = np.cumsum(np.array([0*time]+[(peak[0]*np.exp(2*np.pi*1j*time*peak[1]))for peak in peaks]),axis=0).T
    #waves = np.array([0*time]+[(peak[0]*np.exp(2*np.pi*1j*time*peak[1]))for peak in peaks])
    #pts = np.zeros((len(waves),divisions),dtype=np.complex64)
    #for i in range(1,len(waves)):
        #pts[i,:] = waves[:i+1,:].sum(axis=0)
    #pts = pts.T
    print("Shape",pts.shape)
    rpts = np.zeros((pts.shape[0],pts.shape[1],2))
    rpts[:,:,0] = pts.real
    rpts[:,:,1] = pts.imag
    return rpts.astype(int)

timediv = 240
passtime = 3
ptst = pstime(timediv,peaks)
print(ptst[0])
print(len(ptst))
ctime = 0
counter = 0
done = 0
rec = 1

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter('output.mp4',fourcc, 45, (size,size), True)

clar = 0
cols = [(.5,0,.5),(.5,.5,.5),(.05,0,.05),(.10,.10,.10)]

while 1:
    screen = np.zeros((size,size))
    fscreen = np.zeros((size,size,3))

    for i in range(1,len(ptst[ctime])):
        cv2.line(fscreen,tuple(ptst[ctime][i-1]+centre),tuple(ptst[ctime][i]+centre),cols[clar],1)
        cv2.circle(fscreen,tuple(ptst[ctime][i-1]+centre),int(np.linalg.norm(ptst[ctime][i]-ptst[ctime][i-1])),cols[clar+1],1)
    for i in range(1,len(points)):
        cv2.line(screen,tuple(points[i-1]),tuple(points[i]),(1,1,1),2,cv2.LINE_AA)
    for i in range(1,ctime if not done else len(ptst)):
        cv2.line(fscreen,tuple(ptst[i-1][-1]+centre),tuple(ptst[i][-1]+centre),(1,1,1),1,cv2.LINE_AA)

    cv2.imshow(window,screen)
    cv2.imshow(fwindow,fscreen)

    if rec:
        #print("BANANA",fscreen.max())
        out.write((fscreen*255).astype(np.uint8))

    counter+=1
    if counter==passtime:
        ctime+=4
        if ctime == len(ptst):
            done=1
        ctime%=len(ptst)
        counter=0
    k=cv2.waitKey(1)&0xff
    if k==ord('q'):
        break
    elif k==ord('+') or (ctime==0 and counter==0 and num<60):
        num+=1
        if num==4:
            passtime = 1
            timediv = 1
        if num==60:
            passtime = 3
            timediv = 240
        peaks = np.array(fourier(points,num)).T
        ptst = pstime(timediv,peaks)
        ctime=0
        counter=0
        done=0
    elif k==ord('-'):
        num-=1
        peaks = np.array(fourier(points,num)).T
        ptst = pstime(timediv,peaks)
        ctime=0
        counter=0
        done=0
    elif k==ord('s'):
        ctime=0
        counter=0
        done=0
        num=0
        rec=1
out.release()
cv2.imwrite("img.jpg",screen.astype(np.uint8))
cv2.destroyAllWindows()
