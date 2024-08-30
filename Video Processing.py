#!/usr/bin/env python
# coding: utf-8

# Functions' definitions

# In[1]:


def apply_wavelet_transform(image_segment, wavelet='haar', mode='symmetric'):
    """
    Apply a 2D wavelet transform to the image segment.

    Parameters:
    - image_segment: 2D numpy array, the image or segment to transform.
    - wavelet: string, the type of wavelet to use.
    - mode: string, the mode of the wavelet transform.

    Returns:
    - coeffs: coefficients from the wavelet transform.
    """
    coeffs = pywt.dwt2(image_segment, wavelet, mode)
    return coeffs


def apply_inverse_wavelet_transform(coeffs, wavelet='haar', mode='symmetric'):
    """
    Apply the inverse 2D wavelet transform.

    Parameters:
    - coeffs: coefficients from the wavelet transform.
    - wavelet: string, the type of wavelet to use.
    - mode: string, the mode of the wavelet transform.

    Returns:
    - reconstructed_segment: the reconstructed image segment.
    """
    reconstructed_segment = pywt.idwt2(coeffs, wavelet, mode)
    return reconstructed_segment

def quantize_wavelet_coeffs(coeffs, n_levels):
    """
    Quantize all wavelet coefficients using Lloyd-Max quantization.

    Parameters:
    - coeffs: Wavelet coefficients returned by pywt.dwt2 or similar, in the form (cA, (cH, cV, cD)).
    - n_levels: Number of quantization levels.
    
    - cA: Approximation coefficient, are the lowpass representation of the signal.
    - cH: Horizantal detail coefficient
    - cV: Vertical detail coefficient
    - cD: Diagonal detail coefficient
    
    Αpproximation coefficients typically capture the broader, more general features of a signal.
    Detail coefficients capture smaller details and noise.

    Returns:
    - Quantized wavelet coefficients in the same structure.
    """
    cA, (cH, cV, cD) = coeffs
    quantized_cA = lloyd_max_quantization(cA.flatten(), n_levels).reshape(cA.shape)
    quantized_cH = lloyd_max_quantization(cH.flatten(), n_levels).reshape(cH.shape)
    quantized_cV = lloyd_max_quantization(cV.flatten(), n_levels).reshape(cV.shape)
    quantized_cD = lloyd_max_quantization(cD.flatten(), n_levels).reshape(cD.shape)
    return (quantized_cA, (quantized_cH, quantized_cV, quantized_cD))

def calculate_shannon_entropy(coeffs):
    """
    Calculate Shannon entropy of quantized wavelet coefficients.

    Parameters:
    - coeffs: Quantized wavelet coefficients in the form (cA, (cH, cV, cD)).

    Returns:
    - Average Shannon entropy of all coefficients.
    """
    cA, (cH, cV, cD) = coeffs
    entropies = [shannon_entropy(cA), shannon_entropy(cH), shannon_entropy(cV), shannon_entropy(cD)]
    return np.mean(entropies)



def lloyd_max_quantization(signal, n_levels):
    # Initialize quantization levels (centroids) and decision boundaries
    min_val, max_val = np.min(signal), np.max(signal)
    quant_levels = np.linspace(min_val, max_val, n_levels)
    decision_boundaries = (quant_levels[:-1] + quant_levels[1:]) / 2
    prev_quant_levels = np.copy(quant_levels)
    
    # Iteratively update quantization levels and decision boundaries
    for _ in range(100):  # number of iterations
        # Assign each sample in the signal to the closest quantization level
        indices = np.digitize(signal, decision_boundaries, right=True)
        
        # Update quantization levels to be the mean of the assigned samples
        for i in range(n_levels):
            assigned_samples = signal[indices == i]
            if len(assigned_samples) > 0:
                quant_levels[i] = np.mean(assigned_samples)
        
        # Update decision boundaries
        decision_boundaries = (quant_levels[:-1] + quant_levels[1:]) / 2
        
        # Check for convergence
        if np.linalg.norm(prev_quant_levels - quant_levels) < 1e-5:
            break
        prev_quant_levels = np.copy(quant_levels)
    
    # Quantize the signal
    quantized_signal = np.zeros_like(signal)
    for i in range(n_levels):
        quantized_signal[indices == i] = quant_levels[i]
    
    return quantized_signal



def calculate_psnr(original, reconstructed):
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

#def create_mask_from_polygon(image_shape, polygon_coords):
    # Create a mask for a segment of the image based on polygon coordinates
    #rr, cc = polygon(polygon_coords[:, 1], polygon_coords[:, 0], image_shape)
    #mask = np.zeros(image_shape, dtype=np.uint8)
    #mask[rr, cc] = 1
    #return mask


# Opening the cif file 

# In[2]:


import cv2
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# Load the video
video_path = '/Users/maria/Downloads/akiyo_cif.y4m'
cap = cv2.VideoCapture(video_path)

try:
    while True:
        # Read frame from the video
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no frames left to read

        # Convert the frame from BGR to RGB (OpenCV uses BGR by default)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame
        plt.figure(figsize=(10, 6))
        plt.imshow(frame_rgb)
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.show()
        
        # Clear the output for dynamic display
        clear_output(wait=True)

finally:
    # Release the video capture object
    cap.release()



# Load the YOLO model

# In[3]:


from ultralytics import YOLO

model = YOLO('yolov8m-seg.pt')


# Fixing the path of ffmeg in order to use it in Jupyter

# In[4]:


import os
print(os.environ['PATH'])


# In[5]:


import os
# Example: Adding a common location for Homebrew-installed binaries on macOS
os.environ['PATH'] += os.pathsep + '/opt/homebrew/bin/ffmpeg'


# Convert the video from cif to mp4 

# In[6]:


get_ipython().system(' /opt/homebrew/bin/ffmpeg -y -i /Users/maria/Downloads/akiyo_cif.y4m -c:v libx264 -crf 23 -preset fast /Users/maria/Downloads/akiyo_cif.avi')


# Run the inference 

# In[7]:


results = model.predict('/Users/maria/Downloads/akiyo_cif.avi', show = True, save =True)


# Print the masks coords of the segmemts

# In[8]:


for r in results:
      print(r.masks)


# Frame Details:
# 
# Parsed_showinfo_1: This is the label for a filter instance that reports detailed information about each frame.
# 
# pts: This stands for Presentation Time Stamp; it's the time at which a frame is meant to be shown. It's expressed in a unit that depends on the stream's time base.
# 
# pts_time: This represents the presentation timestamp converted into seconds.
# 
# duration_time: This is the duration of the frame in seconds.
# 
# fmt: Frame format (e.g., yuv420p, which is a common format where the brightness is sampled at every pixel but the 
# 
# color is sampled less frequently).
# 
# iskey:1 type:I: Indicates that the frame is an I-frame (Intra-coded frame, which is a keyframe and can be independently decoded).
# 
# checksum: This is a checksum value for the frame data, useful for data integrity checks.
# 
# 
# Stream and Encoding Details:
# 
# The information about the video stream (Video: h264...) describes the codec used (H.264), pixel format (yuv420p), and other encoding parameters like bitrate and frame rate.
# 
# encoder: Specifies the library used for encoding, Lavc61.3.100 libx264 indicates Libavcodec version 61.3.100 with libx264 (an H.264/MPEG-4 AVC encoder).
# 
# The Stream mapping: section shows how the streams in the input are processed and mapped to outputs. Here, video is taken from stream #0:0 and processed with the wrapped_avframe encoder.
# 
# 
# Performance and Miscellaneous Details:
# 
# speed= 574x: This indicates the speed of the processing relative to real-time playback. A speed of 574x means FFmpeg processed the video much faster than it would play in real time.
# 
# frame= 2 fps=0.0 q=-0.0 Lsize=N/A time=00:00:08.37 bitrate=N/A: This shows the total frames processed, average processing speed in frames per second, quality factor, and other output details like the size and bitrate.
# 
# 
# Conclusion
# 
# The output confirms that FFmpeg successfully processed the video to analyze its frames, focusing on identifying I-frames (keyframes). It extracted two I-frames (as indicated by n: 0 and n: 1) at different presentation times, giving you a basis to understand where the keyframes are located in your video. This can be particularly useful for tasks like video editing or if you need to segment the video based on scene changes, which often align with I-frames.

# Display information about the video

# In[9]:


get_ipython().system('/opt/homebrew/bin/ffmpeg -i /Users/maria/Downloads/akiyo_cif.avi -vf "select=\'eq(pict_type,I)\',showinfo" -f null -')


# Check how many frames are in the video

# In[10]:


import cv2

# Load the video
video = cv2.VideoCapture('/Users/maria/Downloads/akiyo_cif.avi')

# Get the total number of frames
frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# Print the frame count
print("Total number of frames:", frame_count)

# Release the video capture object
video.release()


# Extract the frame rate

# In[11]:


get_ipython().system('/opt/homebrew/bin/ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1:nokey=1 /Users/maria/Downloads/akiyo_cif.avi')


# Split the video into blocks of (12) frames (I used this)

# In[12]:


import subprocess
import os

input_video = '/Users/maria/Downloads/akiyo_cif.avi'
output_directory = './video_segments/'
output_template = 'segment_%03d.avi'

# Number of frames per segment
frames_per_segment = 12  
frame_rate = 29.97 

# Calculate segment duration
segment_duration = frames_per_segment / frame_rate

# Use ffmpeg to re-encode and ensure keyframes are placed every 12 frames
command = [
    '/opt/homebrew/bin/ffmpeg',
    '-y',  # Automatically overwrite existing files without asking
    '-i', input_video,
    '-an',  # Remove audio for simplicity
    '-c:v', 'libx264',  # Re-encode video using x264
    '-x264-params', f'keyint={frames_per_segment}:min-keyint={frames_per_segment}',  # Force keyframes at every 12 frames
    '-map', '0:v',  # Map only the video stream
    '-f', 'segment',
    '-segment_time', str(frames_per_segment / 29.97),  # Time duration for each segment, assuming frame rate is 29.97
    '-reset_timestamps', '1',
    '-segment_format', 'avi',  # Output format of each segment
    os.path.join(output_directory, output_template)
]

# Execute the command
subprocess.run(command, check=True)

print(f"Video segments have been saved in {output_directory}")


# Isolate the I-frame of each segment

# In[15]:


import subprocess
import os

# Directory where the segments are stored
output_directory = './video_segments/'
# Directory to save I-frames as images
iframes_output_directory = './iframes_output/'
os.makedirs(iframes_output_directory, exist_ok=True)  # Ensure the output directory exists

# List all files in the directory
segments = [os.path.join(output_directory, f) for f in os.listdir(output_directory) if f.endswith('.avi')]

# Analyze each segment for I-frames and save them as images
for segment in segments:
    segment_name = os.path.basename(segment).split('.')[0]  # Get the base name without extension
    print(f"Analyzing and extracting I-frames for {segment}")
    command = [
        '/opt/homebrew/bin/ffmpeg',
        '-i', segment,
        '-vf', "select='eq(pict_type,I)'",  # Select I-frames
        '-vsync', 'vfr',  # Variable frame rate to keep only selected frames
        '-frame_pts', 'true',  # Use presentation timestamp for filenames
        os.path.join(iframes_output_directory, f'{segment_name}_iframe_%d.png')  # Save frames as JPEG
    ]
    subprocess.run(command, check=True)  # Execute the command and check for errors

print("I-frames have been saved in the output directory.")


# In[14]:


import subprocess
import os

# Directory where the segments are stored
output_directory = './video_segments/'
# Directory to save I-frames as images
iframes_output_directory = './iframes_output/'
os.makedirs(iframes_output_directory, exist_ok=True)  # Ensure the output directory exists

# List all files in the directory
segments = [os.path.join(output_directory, f) for f in os.listdir(output_directory) if f.endswith('.avi')]

# Analyze each segment for I-frames and save them as images
for segment in segments:
    segment_name = os.path.basename(segment).split('.')[0]  # Get the base name without extension
    print(f"Analyzing and extracting I-frames for {segment}")
    command = [
    '/opt/homebrew/bin/ffmpeg',
    '-y',
    '-i', input_video,
    '-g', '12',  # Group of Picture (GOP) size (set keyframes every 12 frames)
    '-keyint_min', '12',  # Minimum GOP size, avoid scene cut detection
    '-sc_threshold', '0',  # Disable scene cut detection
    'output_video.avi'
]
subprocess.run(command, check=True)

print("I-frames have been saved in the output directory.")


# Run YOLO on the I-frames 

# In[15]:


from ultralytics import YOLO

model = YOLO('yolov8m-seg.pt')


# Find the mask(s) with the higher confidence
# For each I-frame, seperate foreground and background based on confidence > 90%

# In[16]:


from PIL import Image
import numpy as np
import torch
from skimage.draw import polygon 

def create_mask_from_polygon(image_shape, polygon_coords):
    # polygon_coords should be an Nx2 array where each row is [y, x]
    rr, cc = polygon(polygon_coords[:, 1], polygon_coords[:, 0], image_shape)
    mask = np.zeros(image_shape[:2], dtype=np.uint8)  # Only use height and width for the mask
    mask[rr, cc] = 1
    return mask

def tensor_to_numpy_mask(tensor_mask):
    # Assuming the mask is in the first element of the batch
    return tensor_mask.squeeze().numpy()



# Directory containing I-frame images
iframes_directory = './iframes_output/'

# List all JPEG images in the directory
iframes = [os.path.join(iframes_directory, f) for f in os.listdir(iframes_directory) if f.endswith('.png')]

for i_frame in iframes:
    image_frame = Image.open(i_frame)
    image_np = np.array(image_frame)
    results = model.predict(image_frame, show = True, save =True)
    
    highest_score = 0
    best_mask = None

    for r in results:
        # Assuming each r has multiple detections and associated scores
        for idx, score in enumerate(r.boxes.conf):
            if score > highest_score:
                highest_score = score
                best_mask = r.masks[idx]  # Access the corresponding mask

                print("Highest Score:", highest_score)
                print("Best Mask Coordinates:", best_mask)
                
    if best_mask is not None:
        # Check if 'best_mask' is a torch Tensor and has 'xy' coordinates
        if isinstance(best_mask, torch.Tensor):
            mask_numpy = tensor_to_numpy_mask(best_mask.data)
        elif hasattr(best_mask, 'xy'):
            polygon_coords = np.array(best_mask.xy[0])  # Assuming xy is a list of arrays
            mask_numpy = create_mask_from_polygon(image_np.shape, polygon_coords)

        # Separate the image into foreground and background using the numpy mask
        foreground = image_np.copy()
        background = image_np.copy()
        foreground[mask_numpy == 0] = 0  # Apply mask to the foreground
        background[mask_numpy == 1] = 0  # Apply inverse mask to the background

        # Show or save these images
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(foreground)
        plt.title('Foreground')
        plt.subplot(1, 2, 2)
        plt.imshow(background)
        plt.title('Background')
        plt.show()


# Seperate foreground and background using segment(s) with confidence > 90% and apply wavelet transform to both of them

# In[16]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pywt
from ultralytics import YOLO

def apply_wavelet_transform(image, wavelet='db1', level=1):
    # Apply a 2D Discrete Wavelet Transform
    coeffs = pywt.dwt2(image, wavelet)
    return coeffs

# Initialize the YOLO model (ensure the model path is correct)
model = YOLO('yolov8m-seg.pt')

# Directory containing I-frame images
iframes_directory = './iframes_output/'
iframes = [os.path.join(iframes_directory, f) for f in os.listdir(iframes_directory) if f.endswith('.png')]

for i_frame in iframes:
    image_frame = Image.open(i_frame)
    image_np = np.array(image_frame)

    if len(image_np.shape) == 2:  # The image is already grayscale
        gray_image = image_np
    elif len(image_np.shape) == 3 and image_np.shape[2] == 3:  # The image is RGB
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Unsupported image format")

    height, width = gray_image.shape
    
    results = model.predict(image_frame, show=True, save=True)
    
    highest_score = 0
    best_mask = None

    foreground_mask = np.zeros((height, width), dtype=np.uint8)
    background_mask = np.ones((height, width), dtype=np.uint8) * 255

    for r in results:
        for idx, score in enumerate(r.boxes.conf):
            if score > highest_score:
                highest_score = score
                best_mask = r.masks[idx]
    
    if best_mask:
        for segment in best_mask.xy:
            segment = np.array(segment, dtype=np.int32)
            cv2.fillPoly(foreground_mask, [segment], 255)
            cv2.fillPoly(background_mask, [segment], 0)

        foreground = cv2.bitwise_and(gray_image, gray_image, mask=foreground_mask)
        background = cv2.bitwise_and(gray_image, gray_image, mask=background_mask)

        coeffs_foreground = apply_wavelet_transform(foreground)
        cA, (cH, cV, cD) = coeffs_foreground

        coeffs_background = apply_wavelet_transform(background)
        cA_back, (cH_back, cV_back, cD_back) = coeffs_background

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 4, 1)
        plt.imshow(cA, cmap='gray')
        plt.title('Foreground cA')
        plt.subplot(2, 4, 2)
        plt.imshow(cH, cmap='gray')
        plt.title('Foreground cH')
        plt.subplot(2, 4, 3)
        plt.imshow(cV, cmap='gray')
        plt.title('Foreground cV')
        plt.subplot(2, 4, 4)
        plt.imshow(cD, cmap='gray')
        plt.title('Foreground cD')
        plt.subplot(2, 4, 5)
        plt.imshow(cA_back, cmap='gray')
        plt.title('Background cA')
        plt.subplot(2, 4, 6)
        plt.imshow(cH_back, cmap='gray')
        plt.title('Background cH')
        plt.subplot(2, 4, 7)
        plt.imshow(cV_back, cmap='gray')
        plt.title('Background cV')
        plt.subplot(2, 4, 8)
        plt.imshow(cD_back, cmap='gray')
        plt.title('Background cD')

        plt.tight_layout()
        plt.show()


# In[35]:


import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pywt
from ultralytics import YOLO

def apply_wavelet_transform(image, wavelet='db1', level=1):
    coeffs = pywt.dwt2(image, wavelet)
    return coeffs

# Initialize the YOLO model
model = YOLO('yolov8m-seg.pt')

# Directory containing I-frame images
iframes_directory = './iframes_output/'
# Directories to save the masks
foreground_masks_directory = './foreground_masks/'
background_masks_directory = './background_masks/'

os.makedirs(foreground_masks_directory, exist_ok=True)
os.makedirs(background_masks_directory, exist_ok=True)

iframes = [os.path.join(iframes_directory, f) for f in os.listdir(iframes_directory) if f.endswith('.png')]

for i_frame in iframes:
    image_frame = Image.open(i_frame)
    image_np = np.array(image_frame)

    if len(image_np.shape) == 2:  # The image is already grayscale
        gray_image = image_np
    elif len(image_np.shape) == 3 and image_np.shape[2] == 3:  # The image is RGB
        gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Unsupported image format")

    height, width = gray_image.shape

    results = model.predict(image_frame, show=True, save=True)

    highest_score = 0
    best_mask = None

    foreground_mask = np.zeros((height, width), dtype=np.uint8)
    background_mask = np.ones((height, width), dtype=np.uint8) * 255

    for r in results:
        for idx, score in enumerate(r.boxes.conf):
            if score > highest_score:
                highest_score = score
                best_mask = r.masks[idx]

    if best_mask:
        for segment in best_mask.xy:
            segment = np.array(segment, dtype=np.int32)
            cv2.fillPoly(foreground_mask, [segment], 255)
            cv2.fillPoly(background_mask, [segment], 0)

        # Save the masks
        fg_mask_path = os.path.join(foreground_masks_directory, os.path.basename(i_frame).replace('.png', '_fg_mask.png'))
        bg_mask_path = os.path.join(background_masks_directory, os.path.basename(i_frame).replace('.png', '_bg_mask.png'))
        cv2.imwrite(fg_mask_path, foreground_mask)
        cv2.imwrite(bg_mask_path, background_mask)

        foreground = cv2.bitwise_and(gray_image, gray_image, mask=foreground_mask)
        background = cv2.bitwise_and(gray_image, gray_image, mask=background_mask)

        coeffs_foreground = apply_wavelet_transform(foreground)
        cA, (cH, cV, cD) = coeffs_foreground

        coeffs_background = apply_wavelet_transform(background)
        cA_back, (cH_back, cV_back, cD_back) = coeffs_background

        # Visualization
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 4, 1)
        plt.imshow(cA, cmap='gray')
        plt.title('Foreground cA')
        plt.subplot(2, 4, 2)
        plt.imshow(cH, cmap='gray')
        plt.title('Foreground cH')
        plt.subplot(2, 4, 3)
        plt.imshow(cV, cmap='gray')
        plt.title('Foreground cV')
        plt.subplot(2, 4, 4)
        plt.imshow(cD, cmap='gray')
        plt.title('Foreground cD')
        plt.subplot(2, 4, 5)
        plt.imshow(cA_back, cmap='gray')
        plt.title('Background cA')
        plt.subplot(2, 4, 6)
        plt.imshow(cH_back, cmap='gray')
        plt.title('Background cH')
        plt.subplot(2, 4, 7)
        plt.imshow(cV_back, cmap='gray')
        plt.title('Background cV')
        plt.subplot(2, 4, 8)
        plt.imshow(cD_back, cmap='gray')
        plt.title('Background cD')

        plt.tight_layout()
        plt.show()


# Use motion vectors for encoding (Fixed size of motion vectors)

# In[17]:


import os
import cv2
import numpy as np

# Directories for segments and motion vectors
output_directory = './video_segments/'
motion_vectors_output_directory = './motion_vectors_output/'

# Ensure the output directories exist
os.makedirs(motion_vectors_output_directory, exist_ok=True)

# List all .avi files in the directory
segments = [os.path.join(output_directory, f) for f in os.listdir(output_directory) if f.endswith('.avi')]

# Number of frames to process for each segment
frames_to_encode = 12  # Example fixed number of frames per segment

# Function to save motion vectors
def save_motion_vectors(motion_vectors, filename):
    np.save(filename, motion_vectors)

# Analyze each segment and extract motion vectors
for segment in segments:
    segment_name = os.path.basename(segment).split('.')[0]  # Get the base name without extension
    print(f"Analyzing and extracting motion vectors for {segment}")

    # Open the video file
    cap = cv2.VideoCapture(segment)
    if not cap.isOpened():
        print(f"Failed to open {segment}")
        continue

    # Prepare for motion estimation
    prev_gray = None
    frame_idx = 0
    all_motion_vectors = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # Skip I-frames (example check, adjust as needed)
        if frame_number % 10 != 0:
            if prev_gray is not None:
                # Calculate motion vectors using Farneback's method
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                all_motion_vectors.append(flow)
                print(f"Processed motion vectors for frame {frame_idx}")

            prev_gray = gray_frame

        frame_idx += 1
        if frame_idx >= frames_to_encode:
            break

    # Concatenate all motion vectors into one array and save
    if all_motion_vectors:
        all_motion_vectors = np.array(all_motion_vectors)
        motion_vectors_filename = os.path.join(motion_vectors_output_directory, f'{segment_name}_motion_vectors.npy')
        save_motion_vectors(all_motion_vectors, motion_vectors_filename)
        print(f"Saved all motion vectors for segment {segment_name} to {motion_vectors_filename}")

    cap.release()

print("Motion vectors have been saved in the motion vectors output directory.")


# Print the contents of the .npy files 

# In[18]:


import os
import numpy as np

# Directory where the motion vectors are saved
motion_vectors_output_directory = './motion_vectors_output/'

# List all .npy files in the directory
motion_vector_files = [os.path.join(motion_vectors_output_directory, f) for f in os.listdir(motion_vectors_output_directory) if f.endswith('.npy')]

# Function to load and print motion vectors
def print_motion_vectors(filename):
    motion_vectors = np.load(filename)
    print(f"Motion vectors for {filename}:")
    print(motion_vectors)

# Iterate over each .npy file and print the motion vectors
for motion_vector_file in motion_vector_files:
    print_motion_vectors(motion_vector_file)


# Visualize the motion vectors

# In[26]:


import os
import numpy as np
import matplotlib.pyplot as plt

# Directory where the motion vectors are saved
motion_vectors_output_directory = './motion_vectors_output/'

# List all .npy files in the directory
motion_vector_files = [os.path.join(motion_vectors_output_directory, f) for f in os.listdir(motion_vectors_output_directory) if f.endswith('.npy')]

# Function to load and plot motion vectors
def plot_motion_vectors(filename, step=10, scale_factor=50):
    motion_vectors = np.load(filename)
    print(f"Motion vectors for {filename}:")
    print(f"Shape of motion vectors array: {motion_vectors.shape}")

    if motion_vectors.ndim == 4:
        seq, height, width, _ = motion_vectors.shape
        y, x = np.mgrid[0:height, 0:width]
        u, v = motion_vectors[0, ..., 0], motion_vectors[0, ..., 1]
    elif motion_vectors.ndim == 3:
        height, width, _ = motion_vectors.shape
        y, x = np.mgrid[0:height, 0:width]
        u, v = motion_vectors[..., 0], motion_vectors[..., 1]
    else:
        print(f"Unexpected shape: {motion_vectors.shape}")
        return

    # Print some statistics about the motion vectors
    print(f"u min: {u.min()}, u max: {u.max()}")
    print(f"v min: {v.min()}, v max: {v.max()}")

    # Subsample the motion vectors to reduce density
    y, x, u, v = y[::step, ::step], x[::step, ::step], u[::step, ::step], v[::step, ::step]
    magnitude = np.sqrt(u**2 + v**2)

    # Plotting the motion vectors with color indicating magnitude and applying scale factor
    plt.figure(figsize=(10, 10))
    plt.quiver(x, y, u, v, magnitude, angles='xy', scale_units='xy', scale=scale_factor, cmap='jet')
    plt.colorbar(label='Magnitude')
    plt.gca().invert_yaxis()
    plt.title(f'Motion Vectors for {os.path.basename(filename)}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Iterate over each .npy file and plot the motion vectors
for motion_vector_file in motion_vector_files:
    plot_motion_vectors(motion_vector_file, step=10, scale_factor=150)


# Apply Huffman encoding and Shannon Entropy calculation to the motion vectors extracted above.

# In[42]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pywt
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
import heapq
from collections import Counter

# Directory to save the output videos and Huffman encoded files
output_directory = './output_videos/'
huffman_directory = './huffman_encoded/'
motion_vectors_directory = './motion_vectors_output/'
os.makedirs(output_directory, exist_ok=True)
os.makedirs(huffman_directory, exist_ok=True)
os.makedirs(motion_vectors_directory, exist_ok=True)

class HuffmanCoding:
    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    class HeapNode:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None

        def __lt__(self, other):
            return self.freq < other.freq

    def make_frequency_dict(self, data):
        return Counter(data)

    def build_heap(self, frequency):
        for key in frequency:
            node = HuffmanCoding.HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = HuffmanCoding.HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, node, current_code):
        if node is None:
            return

        if node.char is not None:
            self.codes[node.char] = current_code
            self.reverse_mapping[current_code] = node.char

        self.make_codes_helper(node.left, current_code + "0")
        self.make_codes_helper(node.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        self.make_codes_helper(root, "")

    def get_encoded_data(self, data):
        encoded_text = ""
        for character in data:
            encoded_text += self.codes[character]
        return encoded_text

    def compress(self, data):
        frequency = self.make_frequency_dict(data)
        self.build_heap(frequency)
        self.merge_nodes()
        self.make_codes()
        encoded_data = self.get_encoded_data(data)
        return encoded_data

def apply_wavelet_transform(image, wavelet='db1', level=1):
    coeffs = pywt.dwt2(image, wavelet)
    return coeffs

def apply_inverse_wavelet_transform(coeffs):
    return pywt.idwt2(coeffs, 'db1')

def lloyd_max_quantization(subband, n_levels):
    flat_subband = subband.ravel()
    min_val, max_val = flat_subband.min(), flat_subband.max()
    quant_levels = np.linspace(min_val, max_val, n_levels)
    decision_boundaries = (quant_levels[:-1] + quant_levels[1:]) / 2
    indices = np.digitize(flat_subband, decision_boundaries, right=True)
    quantized_subband = quant_levels[indices - 1]
    return quantized_subband.reshape(subband.shape)

def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return round(psnr_value, 2)

def calculate_entropy(data):
    flat_data = data.flatten()
    frequency_dict = Counter(flat_data)
    total_count = sum(frequency_dict.values())
    entropy = -sum((count / total_count) * np.log2(count / total_count) for count in frequency_dict.values())
    return entropy

def process_iframes(iframes_directory, motion_vectors_directory, fg_bits, bg_bits):
    iframes = [os.path.join(iframes_directory, f) for f in os.listdir(iframes_directory) if f.endswith('.png')]

    psnr_list = []
    ssim_list = []
    entropy_list = []
    bitrate_list = []
    mv_entropy_list = []
    total_bits = 0

    huffman = HuffmanCoding()

    # Load pre-extracted motion vectors
    motion_vectors_files = [os.path.join(motion_vectors_directory, f) for f in os.listdir(motion_vectors_directory) if f.endswith('.npy')]
    all_motion_vectors = []
    for mv_file in motion_vectors_files:
        motion_vectors = np.load(mv_file)
        all_motion_vectors.extend(motion_vectors.flatten())

    for i_frame in iframes:
        image_frame = Image.open(i_frame)
        gray_image = np.array(image_frame)

        # Ensure the image is grayscale
        if len(gray_image.shape) == 3 and gray_image.shape[2] == 3:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGB2GRAY)
        elif len(gray_image.shape) == 3 and gray_image.shape[2] == 4:
            gray_image = cv2.cvtColor(gray_image, cv2.COLOR_RGBA2GRAY)

        height, width = gray_image.shape
        segment_name = os.path.basename(i_frame).split('.')[0]

        fg_mask_path = os.path.join('./foreground_masks', f'{segment_name}_fg_mask.png')
        bg_mask_path = os.path.join('./background_masks', f'{segment_name}_bg_mask.png')

        fg_mask = cv2.imread(fg_mask_path, cv2.IMREAD_GRAYSCALE)
        bg_mask = cv2.imread(bg_mask_path, cv2.IMREAD_GRAYSCALE)

        if fg_mask is None or bg_mask is None:
            print(f"Skipping frame {segment_name} due to missing masks.")
            continue

        foreground = cv2.bitwise_and(gray_image, gray_image, mask=fg_mask)
        background = cv2.bitwise_and(gray_image, gray_image, mask=bg_mask)

        coeffs_foreground = apply_wavelet_transform(foreground)
        cA, (cH, cV, cD) = coeffs_foreground

        quantized_cA = lloyd_max_quantization(cA, 2 ** fg_bits)
        quantized_cH = lloyd_max_quantization(cH, 2 ** fg_bits)
        quantized_cV = lloyd_max_quantization(cV, 2 ** fg_bits)
        quantized_cD = lloyd_max_quantization(cD, 2 ** fg_bits)

        quantized_coeffs_foreground = (quantized_cA, (quantized_cH, quantized_cV, quantized_cD))
        reconstructed_foreground = apply_inverse_wavelet_transform(quantized_coeffs_foreground)
        normalized_reconstructed_foreground = np.clip(reconstructed_foreground, 0, 255)

        coeffs_background = apply_wavelet_transform(background)
        cA_back, (cH_back, cV_back, cD_back) = coeffs_background

        quantized_cA_back = lloyd_max_quantization(cA_back, 2 ** bg_bits)
        quantized_cH_back = lloyd_max_quantization(cH_back, 2 ** bg_bits)
        quantized_cV_back = lloyd_max_quantization(cV_back, 2 ** bg_bits)
        quantized_cD_back = lloyd_max_quantization(cD_back, 2 ** bg_bits)

        quantized_coeffs_background = (quantized_cA_back, (quantized_cH_back, quantized_cV_back, quantized_cD_back))
        reconstructed_background = apply_inverse_wavelet_transform(quantized_coeffs_background)
        normalized_reconstructed_background = np.clip(reconstructed_background, 0, 255)

        reconstructed_image = np.where(fg_mask > 0, normalized_reconstructed_foreground, normalized_reconstructed_background)

        # Huffman encode I-frame
        flattened_reconstructed_image = reconstructed_image.flatten().astype(np.uint8)
        encoded_data = huffman.compress(flattened_reconstructed_image)
        encoded_bits = len(encoded_data)  # Length of the binary string in bits
        total_bits += encoded_bits

        # Calculate metrics directly using the quantized and reconstructed image
        current_psnr = calculate_psnr(gray_image, reconstructed_image)
        current_ssim = ssim(gray_image, reconstructed_image, data_range=255)
        current_entropy = calculate_entropy(reconstructed_image)

        psnr_list.append(current_psnr)
        ssim_list.append(current_ssim)
        entropy_list.append(current_entropy)

        # Bitrate calculation in kbps
        avg_bitrate = round(encoded_bits / 1000)  # Convert to kbps
        bitrate_list.append(avg_bitrate)

        print(f"I-Frame {i_frame}: Encoded Bits={encoded_bits}")
        print(f"Frame {len(psnr_list)}: PSNR={current_psnr}, SSIM={current_ssim}, Entropy={current_entropy}")

    # Huffman encode all motion vectors after frame processing
    all_motion_vectors = np.array(all_motion_vectors).astype(np.uint8)
    encoded_motion_vectors = huffman.compress(all_motion_vectors)
    encoded_bits_mv = len(encoded_motion_vectors)
    total_bits += encoded_bits_mv

    # Calculate Shannon entropy for motion vectors
    entropy_mv = calculate_entropy(all_motion_vectors)
    mv_entropy_list.append(entropy_mv)
    print(f"Huffman Encoded Motion Vectors: Encoded Bits={encoded_bits_mv}, Entropy={entropy_mv}")

    return bitrate_list, psnr_list, ssim_list, entropy_list, mv_entropy_list

def process_all_settings(iframes_directory, motion_vectors_directory, output_directory):
    all_metrics = []

    foreground_bits_list = [1, 2, 3, 4, 5, 6, 7, 8]
    background_bits_list = [1, 2, 3, 4, 5, 6, 7, 8]

    for bg_bits in background_bits_list:
        for fg_bits in foreground_bits_list:
            print(f"Processing with FG bits: {fg_bits}, BG bits: {bg_bits}")
            bitrate_list, psnr_list, ssim_list, entropy_list, mv_entropy_list = process_iframes(iframes_directory, motion_vectors_directory, fg_bits, bg_bits)
            for bitrate, psnr, ssim_val, entropy, mv_entropy in zip(bitrate_list, psnr_list, ssim_list, entropy_list, mv_entropy_list):
                all_metrics.append((bitrate, psnr, ssim_val, entropy, mv_entropy, fg_bits, bg_bits))

    plot_results(all_metrics)

def plot_results(metrics):
    plt.figure(figsize=(21, 5))

    plt.subplot(1, 4, 1)
    for bg_bits in sorted(set(metric[6] for metric in metrics)):
        bitrates = [metric[0] for metric in metrics if metric[6] == bg_bits]
        psnrs = [metric[1] for metric in metrics if metric[6] == bg_bits]
        plt.plot(bitrates, psnrs, marker='o', linestyle='-', label=f'BG {bg_bits} bits')
    plt.title('PSNR vs. Bitrate')
    plt.xlabel('Bitrate (kbps)')
    plt.ylabel('PSNR (dB)')
    plt.legend()

    plt.subplot(1, 4, 2)
    for bg_bits in sorted(set(metric[6] for metric in metrics)):
        bitrates = [metric[0] for metric in metrics if metric[6] == bg_bits]
        ssims = [metric[2] for metric in metrics if metric[6] == bg_bits]
        plt.plot(bitrates, ssims, marker='o', linestyle='-', label=f'BG {bg_bits} bits')
    plt.title('SSIM vs. Bitrate')
    plt.xlabel('Bitrate (kbps)')
    plt.ylabel('SSIM')
    plt.legend()

    plt.subplot(1, 4, 3)
    for bg_bits in sorted(set(metric[6] for metric in metrics)):
        bitrates = [metric[0] for metric in metrics if metric[6] == bg_bits]
        entropies = [metric[3] for metric in metrics if metric[6] == bg_bits]
        plt.plot(bitrates, entropies, marker='o', linestyle='-', label=f'BG {bg_bits} bits')
    plt.title('Entropy vs. Bitrate')
    plt.xlabel('Bitrate (kbps)')
    plt.ylabel('Entropy')
    plt.legend()

    plt.subplot(1, 4, 4)
    for bg_bits in sorted(set(metric[6] for metric in metrics)):
        bitrates = [metric[0] for metric in metrics if metric[6] == bg_bits]
        mv_entropies = [metric[4] for metric in metrics if metric[6] == bg_bits]
        plt.plot(bitrates, mv_entropies, marker='o', linestyle='-', label=f'BG {bg_bits} bits')
    plt.title('Motion Vectors Entropy vs. Bitrate')
    plt.xlabel('Bitrate (kbps)')
    plt.ylabel('Motion Vectors Entropy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{output_directory}/psnr_ssim_entropy_combined.png')
    plt.show()

# Directory containing I-frame images and pre-extracted motion vectors
iframes_directory = './iframes_output/'
process_all_settings(iframes_directory, motion_vectors_directory, output_directory)


# In[ ]:





# In[22]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pywt
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
import heapq
from collections import Counter

# Directory to save the output videos and Huffman encoded files
output_directory = './output_videos/'
huffman_directory = './huffman_encoded/'
motion_vectors_directory = './motion_vectors_output/'
os.makedirs(output_directory, exist_ok=True)
os.makedirs(huffman_directory, exist_ok=True)
os.makedirs(motion_vectors_directory, exist_ok=True)

class HuffmanCoding:
    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    class HeapNode:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None

        def __lt__(self, other):
            return self.freq < other.freq

    def make_frequency_dict(self, data):
        return Counter(data)

    def build_heap(self, frequency):
        for key in frequency:
            node = HuffmanCoding.HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = HuffmanCoding.HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, node, current_code):
        if node is None:
            return

        if node.char is not None:
            self.codes[node.char] = current_code
            self.reverse_mapping[current_code] = node.char

        self.make_codes_helper(node.left, current_code + "0")
        self.make_codes_helper(node.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        self.make_codes_helper(root, "")

    def get_encoded_data(self, data):
        encoded_text = ""
        for character in data:
            encoded_text += self.codes[character]
        return encoded_text

    def compress(self, data):
        frequency = self.make_frequency_dict(data)
        self.build_heap(frequency)
        self.merge_nodes()
        self.make_codes()
        encoded_data = self.get_encoded_data(data)
        return encoded_data

def apply_wavelet_transform(image, wavelet='db1', level=1):
    coeffs = pywt.dwt2(image, wavelet)
    return coeffs

def apply_inverse_wavelet_transform(coeffs):
    return pywt.idwt2(coeffs, 'db1')

def lloyd_max_quantization(subband, n_levels):
    flat_subband = subband.ravel()
    min_val, max_val = flat_subband.min(), flat_subband.max()
    quant_levels = np.linspace(min_val, max_val, n_levels)
    decision_boundaries = (quant_levels[:-1] + quant_levels[1:]) / 2
    indices = np.digitize(flat_subband, decision_boundaries, right=True)
    quantized_subband = quant_levels[indices - 1]
    return quantized_subband.reshape(subband.shape)

def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return round(psnr_value, 2)

def calculate_entropy(data):
    flat_data = data.flatten()
    frequency_dict = Counter(flat_data)
    total_count = sum(frequency_dict.values())
    entropy = -sum((count / total_count) * np.log2(count / total_count) for count in frequency_dict.values())
    return entropy

def process_video_with_settings(video_path, output_directory, motion_vectors_directory, fg_bits, bg_bits, i_frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    model = YOLO('yolov8m-seg.pt')
    
    # Define the codec and create VideoWriter object for .avi
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_filename = f'{output_directory}/reconstructed_bg{bg_bits}_fg{fg_bits}.avi'
    out = cv2.VideoWriter(video_filename, fourcc, frame_rate, (width, height))

    psnr_sum = 0
    ssim_sum = 0
    entropy_sum = 0
    total_bits = 0
    frame_count = 0

    huffman = HuffmanCoding()

    # Prepare for motion estimation
    prev_gray = None

    # Load pre-extracted motion vectors
    motion_vectors_files = [os.path.join(motion_vectors_directory, f) for f in os.listdir(motion_vectors_directory) if f.endswith('.npy')]
    all_motion_vectors = []
    for mv_file in motion_vectors_files:
        motion_vectors = np.load(mv_file)
        all_motion_vectors.extend(motion_vectors.flatten())

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray_image.shape

        results = model.predict(frame, show=False, save=False)
        
        highest_score = 0
        best_mask = None

        foreground_mask = np.zeros((height, width), dtype=np.uint8)
        background_mask = np.ones((height, width), dtype=np.uint8) * 255

        for r in results:
            for idx, score in enumerate(r.boxes.conf):
                if score > highest_score:
                    highest_score = score
                    best_mask = r.masks[idx]

        if best_mask:
            for segment in best_mask.xy:
                segment = np.array(segment, dtype=np.int32)
                cv2.fillPoly(foreground_mask, [segment], 255)
                cv2.fillPoly(background_mask, [segment], 0)

            foreground = cv2.bitwise_and(gray_image, gray_image, mask=foreground_mask)
            background = cv2.bitwise_and(gray_image, gray_image, mask=background_mask)

            coeffs_foreground = apply_wavelet_transform(foreground)
            cA, (cH, cV, cD) = coeffs_foreground

            quantized_cA = lloyd_max_quantization(cA, 2 ** fg_bits)
            quantized_cH = lloyd_max_quantization(cH, 2 ** fg_bits)
            quantized_cV = lloyd_max_quantization(cV, 2 ** fg_bits)
            quantized_cD = lloyd_max_quantization(cD, 2 ** fg_bits)

            quantized_coeffs_foreground = (quantized_cA, (quantized_cH, quantized_cV, quantized_cD))
            reconstructed_foreground = apply_inverse_wavelet_transform(quantized_coeffs_foreground)
            normalized_reconstructed_foreground = np.clip(reconstructed_foreground, 0, 255)

            coeffs_background = apply_wavelet_transform(background)
            cA_back, (cH_back, cV_back, cD_back) = coeffs_background

            quantized_cA_back = lloyd_max_quantization(cA_back, 2 ** bg_bits)
            quantized_cH_back = lloyd_max_quantization(cH_back, 2 ** bg_bits)
            quantized_cV_back = lloyd_max_quantization(cV_back, 2 ** bg_bits)
            quantized_cD_back = lloyd_max_quantization(cD_back, 2 ** bg_bits)

            quantized_coeffs_background = (quantized_cA_back, (quantized_cH_back, quantized_cV_back, quantized_cD_back))
            reconstructed_background = apply_inverse_wavelet_transform(quantized_coeffs_background)
            normalized_reconstructed_background = np.clip(reconstructed_background, 0, 255)

            reconstructed_image = np.where(foreground_mask > 0, normalized_reconstructed_foreground, normalized_reconstructed_background)

            if frame_count % i_frame_interval == 0:
                # Huffman encode I-frame
                flattened_reconstructed_image = reconstructed_image.flatten().astype(np.uint8)
                encoded_data = huffman.compress(flattened_reconstructed_image)
                encoded_bits = len(encoded_data)  # Length of the binary string in bits
                total_bits += encoded_bits
                # Debug print for Huffman encoded I-frame
                print(f"I-Frame {frame_count}: Encoded Bits={encoded_bits}")

            # Calculate metrics directly using the quantized and reconstructed image
            current_psnr = calculate_psnr(gray_image, reconstructed_image)
            current_ssim = ssim(gray_image, reconstructed_image, data_range=255)
            current_entropy = calculate_entropy(reconstructed_image)

            psnr_sum += current_psnr
            ssim_sum += current_ssim
            entropy_sum += current_entropy
            frame_count += 1

            # Write the reconstructed frame to the video file
            out.write(cv2.cvtColor(reconstructed_image.astype(np.uint8), cv2.COLOR_GRAY2BGR))

            # Debug print for each frame
            print(f"Frame {frame_count}: PSNR={current_psnr}, SSIM={current_ssim}, Entropy={current_entropy}")

    cap.release()
    out.release()

    # Huffman encode all motion vectors after frame processing
    all_motion_vectors = np.array(all_motion_vectors).astype(np.uint8)
    encoded_motion_vectors = huffman.compress(all_motion_vectors)
    encoded_bits_mv = len(encoded_motion_vectors)
    total_bits += encoded_bits_mv

    # Calculate Shannon entropy for motion vectors
    entropy_mv = calculate_entropy(all_motion_vectors)

    # Debug print for motion vectors
    print(f"Huffman Encoded Motion Vectors: Encoded Bits={encoded_bits_mv}, Entropy={entropy_mv}")

    # Calculate average metrics
    avg_psnr = round(psnr_sum / frame_count, 2)
    avg_ssim = round(ssim_sum / frame_count, 4)
    avg_entropy = round(entropy_sum / frame_count, 4)

    # Calculate the total duration of the video in seconds
    duration_seconds = total_frames / frame_rate

    # Debug prints to verify the calculation of duration
    print(f"Total Frames: {total_frames}")
    print(f"Frame Rate: {frame_rate}")
    print(f"Duration (seconds): {duration_seconds}")

    # Bitrate calculation in kbps (right after motion vectors encoding)
    avg_bitrate = round((total_bits / duration_seconds) / 1000)  # Convert to kbps

    print(f'Processed FG {fg_bits} bits, BG {bg_bits} bits: Bitrate={avg_bitrate} kbps, PSNR={avg_psnr} dB, SSIM={avg_ssim}, Entropy={avg_entropy}, MV Entropy={entropy_mv}')
    return avg_bitrate, avg_psnr, avg_ssim, avg_entropy, entropy_mv

def process_all_settings(video_path, output_directory, motion_vectors_directory):
    metrics = []

    foreground_bits_list = [3, 4, 5]
    background_bits_list = [3, 4, 5]

    for bg_bits in background_bits_list:
        for fg_bits in foreground_bits_list:
            print(f"Processing with FG bits: {fg_bits}, BG bits: {bg_bits}")
            avg_bitrate, avg_psnr, avg_ssim, avg_entropy, entropy_mv = process_video_with_settings(video_path, output_directory, motion_vectors_directory, fg_bits, bg_bits)
            metrics.append((avg_bitrate, avg_psnr, avg_ssim, avg_entropy, entropy_mv, fg_bits, bg_bits))

    plot_results(metrics)

def plot_results(metrics):
    plt.figure(figsize=(21, 5))

    plt.subplot(1, 4, 1)
    for bg_bits in sorted(set(metric[5] for metric in metrics)):
        bitrates = [metric[0] for metric in metrics if metric[5] == bg_bits]
        psnrs = [metric[1] for metric in metrics if metric[5] == bg_bits]
        plt.plot(bitrates, psnrs, marker='o', linestyle='-', label=f'BG {bg_bits} bits')
    plt.title('PSNR vs. Bitrate')
    plt.xlabel('Bitrate (kbps)')
    plt.ylabel('PSNR (dB)')
    plt.legend()

    plt.subplot(1, 4, 2)
    for bg_bits in sorted(set(metric[5] for metric in metrics)):
        bitrates = [metric[0] for metric in metrics if metric[5] == bg_bits]
        ssims = [metric[2] for metric in metrics if metric[5] == bg_bits]
        plt.plot(bitrates, ssims, marker='o', linestyle='-', label=f'BG {bg_bits} bits')
    plt.title('SSIM vs. Bitrate')
    plt.xlabel('Bitrate (kbps)')
    plt.ylabel('SSIM')
    plt.legend()

    plt.subplot(1, 4, 3)
    for bg_bits in sorted(set(metric[5] for metric in metrics)):
        bitrates = [metric[0] for metric in metrics if metric[5] == bg_bits]
        entropies = [metric[3] for metric in metrics if metric[5] == bg_bits]
        plt.plot(bitrates, entropies, marker='o', linestyle='-', label=f'BG {bg_bits} bits')
    plt.title('Entropy vs. Bitrate')
    plt.xlabel('Bitrate (kbps)')
    plt.ylabel('Entropy')
    plt.legend()

    plt.subplot(1, 4, 4)
    for bg_bits in sorted(set(metric[5] for metric in metrics)):
        bitrates = [metric[0] for metric in metrics if metric[5] == bg_bits]
        mv_entropies = [metric[4] for metric in metrics if metric[5] == bg_bits]
        plt.plot(bitrates, mv_entropies, marker='o', linestyle='-', label=f'BG {bg_bits} bits')
    plt.title('Motion Vectors Entropy vs. Bitrate')
    plt.xlabel('Bitrate (kbps)')
    plt.ylabel('Motion Vectors Entropy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{output_directory}/psnr_ssim_entropy_combined.png')
    plt.show()

# Path to the input video
video_path = '/Users/maria/Downloads/akiyo_cif.avi'
process_all_settings(video_path, output_directory, motion_vectors_directory)


# Calculating bitrate from entropy

# In[33]:


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pywt
from skimage.metrics import structural_similarity as ssim
from ultralytics import YOLO
import heapq
from collections import Counter

# Directory to save the output videos and Huffman encoded files
output_directory = './output_videos/'
huffman_directory = './huffman_encoded/'
motion_vectors_directory = './motion_vectors_output/'
os.makedirs(output_directory, exist_ok=True)
os.makedirs(huffman_directory, exist_ok=True)
os.makedirs(motion_vectors_directory, exist_ok=True)

class HuffmanCoding:
    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    class HeapNode:
        def __init__(self, char, freq):
            self.char = char
            self.freq = freq
            self.left = None
            self.right = None

        def __lt__(self, other):
            return self.freq < other.freq

    def make_frequency_dict(self, data):
        return Counter(data)

    def build_heap(self, frequency):
        for key in frequency:
            node = HuffmanCoding.HeapNode(key, frequency[key])
            heapq.heappush(self.heap, node)

    def merge_nodes(self):
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = HuffmanCoding.HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, node, current_code):
        if node is None:
            return

        if node.char is not None:
            self.codes[node.char] = current_code
            self.reverse_mapping[current_code] = node.char

        self.make_codes_helper(node.left, current_code + "0")
        self.make_codes_helper(node.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        self.make_codes_helper(root, "")

    def get_encoded_data(self, data):
        encoded_text = ""
        for character in data:
            encoded_text += self.codes[character]
        return encoded_text

    def compress(self, data):
        frequency = self.make_frequency_dict(data)
        self.build_heap(frequency)
        self.merge_nodes()
        self.make_codes()
        encoded_data = self.get_encoded_data(data)
        return encoded_data

def apply_wavelet_transform(image, wavelet='db1', level=1):
    coeffs = pywt.dwt2(image, wavelet)
    return coeffs

def apply_inverse_wavelet_transform(coeffs):
    return pywt.idwt2(coeffs, 'db1')

def lloyd_max_quantization(subband, n_levels):
    flat_subband = subband.ravel()
    min_val, max_val = flat_subband.min(), flat_subband.max()
    quant_levels = np.linspace(min_val, max_val, n_levels)
    decision_boundaries = (quant_levels[:-1] + quant_levels[1:]) / 2
    indices = np.digitize(flat_subband, decision_boundaries, right=True)
    quantized_subband = quant_levels[indices - 1]
    return quantized_subband.reshape(subband.shape)

def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return round(psnr_value, 2)

def calculate_entropy(data):
    flat_data = data.flatten()
    frequency_dict = Counter(flat_data)
    total_count = sum(frequency_dict.values())
    entropy = -sum((count / total_count) * np.log2(count / total_count) for count in frequency_dict.values())
    return entropy

def process_video_with_settings(video_path, output_directory, motion_vectors_directory, fg_bits, bg_bits, i_frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    model = YOLO('yolov8m-seg.pt')
    
    # Define the codec and create VideoWriter object for .avi
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_filename = f'{output_directory}/reconstructed_bg{bg_bits}_fg{fg_bits}.avi'
    out = cv2.VideoWriter(video_filename, fourcc, frame_rate, (width, height))

    psnr_sum = 0
    ssim_sum = 0
    entropy_sum = 0
    frame_count = 0

    huffman = HuffmanCoding()

    # Prepare for motion estimation
    prev_gray = None

    # Load pre-extracted motion vectors
    motion_vectors_files = [os.path.join(motion_vectors_directory, f) for f in os.listdir(motion_vectors_directory) if f.endswith('.npy')]
    all_motion_vectors = []
    for mv_file in motion_vectors_files:
        motion_vectors = np.load(mv_file)
        all_motion_vectors.extend(motion_vectors.flatten())

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray_image.shape

        results = model.predict(frame, show=False, save=False)
        
        highest_score = 0
        best_mask = None

        foreground_mask = np.zeros((height, width), dtype=np.uint8)
        background_mask = np.ones((height, width), dtype=np.uint8) * 255

        for r in results:
            for idx, score in enumerate(r.boxes.conf):
                if score > highest_score:
                    highest_score = score
                    best_mask = r.masks[idx]

        if best_mask:
            for segment in best_mask.xy:
                segment = np.array(segment, dtype=np.int32)
                cv2.fillPoly(foreground_mask, [segment], 255)
                cv2.fillPoly(background_mask, [segment], 0)

            foreground = cv2.bitwise_and(gray_image, gray_image, mask=foreground_mask)
            background = cv2.bitwise_and(gray_image, gray_image, mask=background_mask)

            coeffs_foreground = apply_wavelet_transform(foreground)
            cA, (cH, cV, cD) = coeffs_foreground

            quantized_cA = lloyd_max_quantization(cA, 2 ** fg_bits)
            quantized_cH = lloyd_max_quantization(cH, 2 ** fg_bits)
            quantized_cV = lloyd_max_quantization(cV, 2 ** fg_bits)
            quantized_cD = lloyd_max_quantization(cD, 2 ** fg_bits)

            quantized_coeffs_foreground = (quantized_cA, (quantized_cH, quantized_cV, quantized_cD))
            reconstructed_foreground = apply_inverse_wavelet_transform(quantized_coeffs_foreground)
            normalized_reconstructed_foreground = np.clip(reconstructed_foreground, 0, 255)

            coeffs_background = apply_wavelet_transform(background)
            cA_back, (cH_back, cV_back, cD_back) = coeffs_background

            quantized_cA_back = lloyd_max_quantization(cA_back, 2 ** bg_bits)
            quantized_cH_back = lloyd_max_quantization(cH_back, 2 ** bg_bits)
            quantized_cV_back = lloyd_max_quantization(cV_back, 2 ** bg_bits)
            quantized_cD_back = lloyd_max_quantization(cD_back, 2 ** bg_bits)

            quantized_coeffs_background = (quantized_cA_back, (quantized_cH_back, quantized_cV_back, quantized_cD_back))
            reconstructed_background = apply_inverse_wavelet_transform(quantized_coeffs_background)
            normalized_reconstructed_background = np.clip(reconstructed_background, 0, 255)

            reconstructed_image = np.where(foreground_mask > 0, normalized_reconstructed_foreground, normalized_reconstructed_background)

            if frame_count % i_frame_interval == 0:
                # Huffman encode I-frame
                flattened_reconstructed_image = reconstructed_image.flatten().astype(np.uint8)
                encoded_data = huffman.compress(flattened_reconstructed_image)
                encoded_bits = len(encoded_data)  # Length of the binary string in bits
                # Debug print for Huffman encoded I-frame
                print(f"I-Frame {frame_count}: Encoded Bits={encoded_bits}")

            # Calculate metrics directly using the quantized and reconstructed image
            current_psnr = calculate_psnr(gray_image, reconstructed_image)
            current_ssim = ssim(gray_image, reconstructed_image, data_range=255)
            current_entropy = calculate_entropy(reconstructed_image)

            psnr_sum += current_psnr
            ssim_sum += current_ssim
            entropy_sum += current_entropy
            frame_count += 1

            # Write the reconstructed frame to the video file
            out.write(cv2.cvtColor(reconstructed_image.astype(np.uint8), cv2.COLOR_GRAY2BGR))

            # Debug print for each frame
            print(f"Frame {frame_count}: PSNR={current_psnr}, SSIM={current_ssim}, Entropy={current_entropy}")

    cap.release()
    out.release()

    # Huffman encode all motion vectors after frame processing
    all_motion_vectors = np.array(all_motion_vectors).astype(np.uint8)
    encoded_motion_vectors = huffman.compress(all_motion_vectors)
    encoded_bits_mv = len(encoded_motion_vectors)
    total_bits = encoded_bits_mv  # Only consider motion vectors' encoded bits

    # Calculate Shannon entropy for motion vectors
    entropy_mv = calculate_entropy(all_motion_vectors)

    # Debug print for motion vectors
    print(f"Huffman Encoded Motion Vectors: Encoded Bits={encoded_bits_mv}, Entropy={entropy_mv}")

    # Calculate average metrics
    avg_psnr = round(psnr_sum / frame_count, 2)
    avg_ssim = round(ssim_sum / frame_count, 4)
    avg_entropy = round(entropy_sum / frame_count, 4)

    # Calculate the total duration of the video in seconds
    duration_seconds = total_frames / frame_rate

    # Debug prints to verify the calculation of duration
    print(f"Total Frames: {total_frames}")
    print(f"Frame Rate: {frame_rate}")
    print(f"Duration (seconds): {duration_seconds}")

    # Bitrate calculation in kbps (considering only motion vectors' encoded bits)
    avg_bitrate = round((total_bits / duration_seconds) / 1000)  # Convert to kbps

    print(f'Processed FG {fg_bits} bits, BG {bg_bits} bits: Bitrate={avg_bitrate} kbps, PSNR={avg_psnr} dB, SSIM={avg_ssim}, Entropy={avg_entropy}, MV Entropy={entropy_mv}')
    return avg_bitrate, avg_psnr, avg_ssim, avg_entropy, entropy_mv

def process_all_settings(video_path, output_directory, motion_vectors_directory):
    metrics = []

    foreground_bits_list = [3, 4, 5]
    background_bits_list = [3, 4, 5]

    for bg_bits in background_bits_list:
        for fg_bits in foreground_bits_list:
            print(f"Processing with FG bits: {fg_bits}, BG bits: {bg_bits}")
            avg_bitrate, avg_psnr, avg_ssim, avg_entropy, entropy_mv = process_video_with_settings(video_path, output_directory, motion_vectors_directory, fg_bits, bg_bits)
            metrics.append((avg_bitrate, avg_psnr, avg_ssim, avg_entropy, entropy_mv, fg_bits, bg_bits))

    plot_results(metrics)

def plot_results(metrics):
    plt.figure(figsize=(21, 5))

    plt.subplot(1, 4, 1)
    for bg_bits in sorted(set(metric[5] for metric in metrics)):
        bitrates = [metric[0] for metric in metrics if metric[5] == bg_bits]
        psnrs = [metric[1] for metric in metrics if metric[5] == bg_bits]
        plt.plot(bitrates, psnrs, marker='o', linestyle='-', label=f'BG {bg_bits} bits')
    plt.title('PSNR vs. Bitrate')
    plt.xlabel('Bitrate (kbps)')
    plt.ylabel('PSNR (dB)')
    plt.legend()

    plt.subplot(1, 4, 2)
    for bg_bits in sorted(set(metric[5] for metric in metrics)):
        bitrates = [metric[0] for metric in metrics if metric[5] == bg_bits]
        ssims = [metric[2] for metric in metrics if metric[5] == bg_bits]
        plt.plot(bitrates, ssims, marker='o', linestyle='-', label=f'BG {bg_bits} bits')
    plt.title('SSIM vs. Bitrate')
    plt.xlabel('Bitrate (kbps)')
    plt.ylabel('SSIM')
    plt.legend()

    plt.subplot(1, 4, 3)
    for bg_bits in sorted(set(metric[5] for metric in metrics)):
        bitrates = [metric[0] for metric in metrics if metric[5] == bg_bits]
        entropies = [metric[3] for metric in metrics if metric[5] == bg_bits]
        plt.plot(bitrates, entropies, marker='o', linestyle='-', label=f'BG {bg_bits} bits')
    plt.title('Entropy vs. Bitrate')
    plt.xlabel('Bitrate (kbps)')
    plt.ylabel('Entropy')
    plt.legend()

    plt.subplot(1, 4, 4)
    for bg_bits in sorted(set(metric[5] for metric in metrics)):
        bitrates = [metric[0] for metric in metrics if metric[5] == bg_bits]
        mv_entropies = [metric[4] for metric in metrics if metric[5] == bg_bits]
        plt.plot(bitrates, mv_entropies, marker='o', linestyle='-', label=f'BG {bg_bits} bits')
    plt.title('Motion Vectors Entropy vs. Bitrate')
    plt.xlabel('Bitrate (kbps)')
    plt.ylabel('Motion Vectors Entropy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{output_directory}/psnr_ssim_entropy_combined.png')
    plt.show()

# Path to the input video
video_path = '/Users/maria/Downloads/akiyo_cif.avi'
process_all_settings(video_path, output_directory, motion_vectors_directory)


# In[ ]:





# In[ ]:




