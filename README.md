# CityscapesDilatedConvolutionHighway280
I was inspired by David Abati's dilated convolutional github on semantic segmentation, trained on Cityscapes dataset.  I just want to reproduce his work using my own video.  

It is a very interesting topic.  According to "Semantic Segmentation Using Fully Convolutional Networks over the Years", written by
meetshah1995, feature upsampling the low-resolution segmentation maps to input image resolution using learned deconvolutions will totally
or partially avoid the reduction of resolution altogether in the encoder, using dilated convolutions.

The result of my duplication of David Abati's github using my own images and video is sort of similar, but it is not as perfect as his own augmented image.  I modified his python code to generate a dilated convolutional video of my own.

Here are the requirements to run this repository:

1. Download pretrained weights from here:

    [CityScapes weights](https://drive.google.com/open?id=0Bx9YaGcDPu3XR0d4cXVSWmtVdEE)
    
    [CamVid weights](https://drive.google.com/open?id=0Bx9YaGcDPu3Xd0JrcXZpTEpkb0U)
    
2. Move weights file into [`data`](data) directory.

3. Configuration requirements:

This model has been tested with the following configuration:
- Ubuntu 16.04
- python 3.5.2
- tensorflow 1.1.0
- cv2 3.2.0


Here is the instruction to generate the videos:
For just the dilated video:
1) In Anaconda prompt, type these 2 commands:  
  python main_tf.py input.mp4
  
  
  where input.mp4 is the input video file name which is located in the test_videos directory
  If you are interested in the input video file, just email me at blei139@gmail.com.  I will be happy to send the copy of the input
  video or you can use your own video by pasting it into the testvideos directory.
  
For both the input video and dilated video combine together into one video for comparison:
1) In Anaconda prompt, type these 2 commands:  

  python main_tf.py input.mp4
  
  python data/test_videos/video2imgs.py input.mp4
  where input.mp4 is the input video file name which is located in the test_videos directory
  If you are interested in the input video file, just email me at blei139@gmail.com.  I will be happy to send the copy of the input
  video or you can use your own video by pasting it into the testvideos directory.
  
Here are youtube links generated from my own camera:

https://youtu.be/mlYPk4kFxmk

https://youtu.be/MFRHjIOUMNU
