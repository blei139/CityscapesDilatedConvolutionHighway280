import cv2
import sys
import numpy as np

def videoAnnotate(vin):
    
    print('video in file: {}'.format(vin))
    inFile = cv2.VideoCapture(vin)
    fOutname = '_'.join(['combined', vin])
   
    print('video in file: {}'.format(vin))
    print('video out file: {}'.format(fOutname))
 
    #check if the input file opened successfully
    if (inFile.isOpened() == False):
        print("Error opening video stream on file")

    #define the codec and create videowriter object
    imgCnt = 0
    fps = 20
    frame_size = (int(inFile.get(3)), int(inFile.get(4)))    #tuple(result.shape[1::-1])
    print("frame_size: {}".format(frame_size))
    writer = cv2.VideoWriter(fOutname,
         cv2.VideoWriter_fourcc(*'MP4V'), fps, frame_size, True)  


    #read until video is completed
    while(inFile.isOpened()):
        #Capture frame by frame
        ret, frame = inFile.read()

        if ret == True:
            #display frame
            #plt.imshow(frame)
            #plt.show()
            #increment the image count
            imgCnt = imgCnt + 1
            """
            if (imgCnt < 10): 
                padding = "00"
            
            elif (imgCnt < 100):
                padding = "0"
            else: 
                padding = ""
            """

            fname = str(imgCnt) #"".join([padding, str(imgCnt)])
            
            fname = "_".join(["image", fname])

            imgFname = "/".join(["videoImgs", fname])
            imgFname = ".".join([imgFname, "jpg"])

            #read 2nd image from another directory
            imgFname2 = "/".join(["imgs", fname])
            imgFname2 = ".".join([imgFname2, "jpg"])
            print("2nd image file name: {}".format(imgFname2))
            frame2 = cv2.imread(imgFname2)

            #combine 2 images into 1 image
            #create an empty matrix
            vis = np.uint8(np.zeros([1440,2560,3]))

            #paste input image over to the empty matrix           
            vis[:720, :1280, :3] = frame
            #paste 2nd input image into the matrix
            vis[:720, 1280:2560, :3] = frame2[:720, :1280, :3]

            #copy vis to frame
            frame = vis
            
            
            cv2.imwrite(imgFname, frame)
            #reduce the size of the frame to (720, 1280)
            frame = cv2.resize(frame, (1280,720))
            writer.write(frame)
        else:
            #if no frame break while loop
            writer.release() 
            print("end of mp4 video file conversion") 
            break

#main
#vin = "MOvI0003.mp4"
#videoAnnotate(vin)

if (sys.argv[1] is not None):
    vin = sys.argv[1] 
    videoAnnotate(vin)
else:
    print("Please input the path of the input file")
