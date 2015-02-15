 # Active confocal imaging for visual prostheses 
import numpy as np
import cv2

def nothing(x):
    pass

def PixilateGray(frame, dynamicRange):
    blockWidth = max(width / pixelCount,1);
    blockHeight = max(height / pixelCount,1);
    if blockHeight == 1 and dynamicRange == 256:
        return frame
        
    divisionsy = height / blockHeight;
    divisionsx = width  / blockWidth;
    canvas = np.zeros((height,width,1), np.uint8)
    for x in xrange(0, divisionsx):
        for y in xrange(0,divisionsy):
            xLoop=x*blockWidth
            yLoop=y*blockHeight
            roi = frame[yLoop:yLoop+blockHeight, xLoop:xLoop+blockWidth]
            Sum = cv2.mean(roi)
            if dynamicRange!=256.0:
                # grey = (128+(Sum[0] - 128.0) * dynamicRange/256.0,0.0,0.0,0.0)
                num =  int(Sum[0])
                grey = ( (float) ( (num/(256/dynamicRange))*(256/(dynamicRange-1)) ) ,0.0,0.0,0.0)
            else:
                grey = Sum
           
            cv2.rectangle(frame,(xLoop,yLoop),(xLoop+blockWidth,yLoop+blockHeight),grey, -1)

    sigma = 1 + (int) (100/pixelCount)
    if blockHeight == 1:
        gray = cv2.GaussianBlur(gray,  (0, 0), sigma);

    return frame	


# def calculateDivisions(newBlockSize):
# 	global divisionsy
# 	global divisionsx
# 	global blockSize
# 	global pixelCount
# 	blockSize = newBlockSize
# 	divisionsy = height / blockSize
# 	divisionsx = width  / blockSize
# 	pixelCount = blockSize * 



# cap = cv2.VideoCapture('/Users/alexandermartinez/Downloads/videoplayback.mov')
cap = cv2.VideoCapture(0)
global height;
global width;
global pixelCount;
ret, frame = cap.read()
height, width, dummy = frame.shape
# frame = cv2.resize(frame, (width/4, height/4))
frame = cv2.resize(frame, (width/2, height/2))

height, width, dummy = frame.shape

# pixilize initialize
blockSize = 80
DFScale = 8
waitDurationMultiplier = 8
numMasks = 3
dynamicRange = 256.0
divisionsy = (height) / blockSize
divisionsx = (width)  / blockSize
pixelCount = blockSize /2

cv2.namedWindow('Image')
cv2.namedWindow('Original')
cv2.namedWindow('Gradient Map')


cv2.createTrackbar('Pixel Count', 'Image',0,90,nothing)
cv2.createTrackbar('Threshold Value', 'Gradient Map',0,160,nothing)
cv2.createTrackbar('Pixelate', 'Image',0,1,nothing)
cv2.createTrackbar('Gradient Kernel Size (0 = No Gradient)', 'Gradient Map',0,10,nothing)
cv2.createTrackbar('Dynamic Range 2^(n+1)', 'Image',0,7,nothing)
cv2.createTrackbar('CLAHE', 'Gradient Map',0,10,nothing)
cv2.createTrackbar('Pause', 'Gradient Map',0,1,nothing)




while(cap.isOpened()):
    threshold_value = cv2.getTrackbarPos('Threshold Value','Gradient Map')
    pixelCount = cv2.getTrackbarPos('Pixel Count','Image')+50
    togglePixilate = cv2.getTrackbarPos('Pixelate','Image')
    kernelsize = 2*(cv2.getTrackbarPos('Gradient Kernel Size (0 = No Gradient)','Gradient Map')-1) + 1
    histogram = cv2.getTrackbarPos('CLAHE','Gradient Map')
    pause = cv2.getTrackbarPos('Pause','Gradient Map')
    dynamicRange = (cv2.getTrackbarPos('Dynamic Range 2^(n+1)','Image') + 1)
    dynamicRange = 2 ** dynamicRange

    if pause == 1:
        if cv2.waitKey(0) == 27:
            break






    ret, frame = cap.read()
    frame = cv2.resize(frame, (width, height))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if kernelsize!= -1 :
        sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=kernelsize)
        sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=kernelsize)
        abs_grad_x = cv2.convertScaleAbs( sobelx  );
        abs_grad_y = cv2.convertScaleAbs( sobely  );
        grad = cv2.addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0 );
        ret, gray = cv2.threshold( grad, threshold_value, 255, cv2.THRESH_TOZERO);
    # cv2.imshow('frame',gray)
    # grad = cv2.cvtColor(grad, cv2.COLOR_BGR2GRAY)
	# if(histogramEq) {
	# 	if (src.channels() > 1){
	# 		cvtColor(src, src, CV_BGR2GRAY);
	# 	}
	# 	Ptr<CLAHE> clahe = createCLAHE();
	# 	clahe->setClipLimit(4);
	# 	clahe->apply(src,src);
	# 	cvtColor(src, src, CV_GRAY2BGR);
	# }

    if histogram !=0:
		clahe = cv2.createCLAHE(clipLimit= 2.0 * histogram , tileGridSize=(8,8))
		gray = clahe.apply(gray)

    cv2.imshow('Gradient Map', gray)
   

    if (togglePixilate == 1):
        gray = PixilateGray(gray,dynamicRange)

    cv2.imshow('Image', gray)
    cv2.imshow('Original', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()