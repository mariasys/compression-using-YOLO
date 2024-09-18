# Image and Video Compression using YOLO

========================================================================

The purpose of this repository is to demonstrate an image and video compression method using the state-of-the-art algorithm YOLO (You Only Look Once) that performs detection and segmentation.
The YOLO algorithm was used to seperate the foreground and the background with confidence higher than 90%



##  Project Language and Techniques
The project is written in Python. 
In order to handle the multimedia data, the FFmpeg suite was used.

On images we used Wavelet transform and LLoyd-max quantization.

On videos we applied Wavelet transform and LLoyd-max quantization to the I-frames and Huffman quantizationn to the motion vectors.
