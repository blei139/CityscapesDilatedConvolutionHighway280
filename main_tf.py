import tensorflow as tf
import pickle
import cv2
import os
import os.path as path
from utils import predict
from model import dilation_model_pretrained
from datasets import CONFIG
import numpy as np
import time
import sys
from matplotlib import pyplot as plt
       
###########################################################
def process_image(input_image):
    # Read and predict on a test image
    print("At time: {} sec, start to read and predict on a video frame.".format(str(time.clock())))
    print("image shape: {}".format(input_image.shape))
    #create an empty matrix
    vis = np.uint8(np.zeros([1024,2048,3]))

    #past input image over to the empty matrix           
    vis[:input_image.shape[0],:input_image.shape[1], :input_image.shape[2]] = input_image
    #copy vis to input_image
    input_image = vis

    input_tensor = graph.get_tensor_by_name('input_placeholder:0')
    predicted_image = predict(input_image, input_tensor, model, dataset, sess)

    # Convert colorspace (palette is in RGB) and save prediction result
    predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
    print("At time: {} sec, finished reading and predicting on a video frame.".format(str(time.clock())))

    #create an empty matrix
    vis = np.uint8(np.zeros([720,1280,3]))

    #paste input image over to the empty matrix           
    vis = predicted_image[:720, :1280, :3]
    #copy vis to input_image
    predicted_image = vis
    return predicted_image

def videoAnnotate(vin):
    vout = '_'.join(['Cityscapes', vin])
    fname = '/'.join(['./data/test_videos', vin])
    fOutname = '/'.join(['./data/test_videos', vout])
   
    print('video in file: {}'.format(vin))
    print('video out file: {}'.format(vout))

    inFile = cv2.VideoCapture(fname) 
    #check if the input file opened successfully
    if (inFile.isOpened() == False):
        print("Error opening video stream on file")

    #define the codec and create videowriter object
    fps = 20
    frame_size = (int(inFile.get(3)), int(inFile.get(4)))    
    print("frame_size: {}".format(frame_size))
    writer = cv2.VideoWriter(fOutname,
         cv2.VideoWriter_fourcc(*'MP4V'), fps, frame_size, True)  

    frameCnt = 0
    #read until video is completed
    while(inFile.isOpened()):
        #Capture frame by frame
        ret, frame = inFile.read()

        if ret == True:
            #display frame
            #plt.imshow(frame)
            #plt.show()
            result = process_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            print("video frame number: {}".format(frameCnt))
            frameCnt = frameCnt + 1
            #plt.imshow(result)
            #plt.show()
            #convert result image to (720,1280,3)
            result = cv2.resize(result, (1280,720))

            writer.write(result)
            fileOutname = '/'.join(['./data/test_videos/imgs', "_".join(["image", str(frameCnt)])])
            cv2.imwrite('.'.join([fileOutname, 'jpg']), result)
        else:
            #if no frame break while loop
            writer.release() 
            print("end of mp4 video file conversion") 
            break

#########################################################
if __name__ == '__main__':
    
    # Choose between 'cityscapes' and 'camvid'
    dataset = 'cityscapes'

    # Load dict of pretrained weights
    print('Loading pre-trained weights...')
    with open(CONFIG[dataset]['weights_file'], 'rb') as f:
        w_pretrained = pickle.load(f)
    print('Done.')

    # Create checkpoint directory
    checkpoint_dir = path.join('data/checkpoint', 'dilation_' + dataset)
    if not path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Image in / out parameters
    input_image_path  = path.join('data', dataset + '.png')
    output_image_path = path.join('data', dataset + '_out.png')

    # Build pretrained model and save it as TF checkpoint
    #limit tensorflow cpu and memory usage so the machine won't freeze up
    with tf.Session(config=
          tf.ConfigProto(inter_op_parallelism_threads=1,
          intra_op_parallelism_threads=1)) as sess:

        # Choose input shape according to dataset characteristics
        input_h, input_w, input_c = CONFIG[dataset]['input_shape']
        input_tensor = tf.placeholder(tf.float32, shape=(None, input_h, input_w, input_c), name='input_placeholder')

        # Create pretrained model
        model = dilation_model_pretrained(dataset, input_tensor, w_pretrained, trainable=False)

        sess.run(tf.global_variables_initializer())

        # Save both graph and weights
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
        saver.save(sess, path.join(checkpoint_dir, 'dilation'))

    # Restore both graph and weights from TF checkpoint
    with tf.Session() as sess:

        saver = tf.train.import_meta_graph(path.join(checkpoint_dir, 'dilation.meta'))
        saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))

        graph = tf.get_default_graph()
        model = graph.get_tensor_by_name('softmax:0')
        model = tf.reshape(model, shape=(1,)+CONFIG[dataset]['output_shape'])

########################################
        #save model to savedModel directory
        #print("At time: {} sec, saving model to savedModel directory.".format(str(time.clock())))
        #model.save_model(sess, "./savedModel")

########################################
        """
        # Read and predict on a test image
        print("At time: {} sec, start to read and predict on a test image.".format(str(time.clock())))
        input_image = cv2.imread(input_image_path)
        #reshape the input image to (1024, 2048, 3) always
        #input_image = cv2.resize(input_image, (2048, 1024))

        #create an empty matrix
        vis = np.uint8(np.zeros([1024,2048,3]))

        #past input image over to the empty matrix           
        vis[:input_image.shape[0],:input_image.shape[1], :input_image.shape[2]] = input_image
        #copy vis to input_image
        input_image = vis

        input_tensor = graph.get_tensor_by_name('input_placeholder:0')
        predicted_image = predict(input_image, input_tensor, model, dataset, sess)

        # Convert colorspace (palette is in RGB) and save prediction result
        predicted_image = cv2.cvtColor(predicted_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_image_path, predicted_image)

        print("At time: {} sec, finished reading and predicting on a test image.".format(str(time.clock())))
        """

        if (sys.argv[1] == "testing"):
            cv2.imwrite("data/cityscapes_augmented.png", process_image(cv2.imread("data/cityscapes.png")))
        elif (sys.argv[1] is not None):
            vin = sys.argv[1] 
            videoAnnotate(vin)

