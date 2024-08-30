#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
from skimage.draw import polygon
from skimage.metrics import mean_squared_error, structural_similarity as ssim
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
import os
import pywt


# In[2]:


from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n-seg.pt')


# In[3]:


results = model('/Users/maria/Lloyd-max quantization/March/21 March/cars.png', save = True, show_labels = False, show_conf = False, show_boxes = False, save_crop = True)  # predict on image


# In[4]:


# View results
for r in results:
    print(r.masks)  # print the Masks object containing the detected instance masks


# In[5]:


from PIL import Image

# Run inference
results = model('')  # results list

# Show the results
for r in results:
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    im.show()  # show image
    im.save('results.jpg')  # save image


# In[6]:


from PIL import Image

def get_image_dimensions(image_path):
    """
    Prints the dimensions of an image in pixels (width x height).

    """
    try:
        # Open the image file
        with Image.open(image_path) as img:
            # Get image dimensions
            width, height = img.size
            
            # Print the dimensions
            print(f"Dimensions of '{image_path}': {width} x {height} pixels")
            
    except FileNotFoundError:
        print(f"File '{image_path}' not found. Please check the path and try again.")
    except IOError:
        print(f"Error opening file '{image_path}'. It may not be an image file.")


image_path = '/Users/maria/Lloyd-max quantization/March/21 March/cars.png' 
get_image_dimensions(image_path)


# In[7]:


# Run batched inference on a list of images
results = model.predict('/Users/maria/Lloyd-max quantization/March/21 March/cars.png', save = True, save_txt = True, show_conf=False, show_labels=False, show_boxes = False)  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs


# In[8]:


# Display the segments' coords

masks.xy


# In[9]:


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

def create_mask_from_polygon(image_shape, polygon_coords):
    # Create a mask for a segment of the image based on polygon coordinates
    rr, cc = polygon(polygon_coords[:, 1], polygon_coords[:, 0], image_shape)
    mask = np.zeros(image_shape, dtype=np.uint8)
    mask[rr, cc] = 1
    return mask


# In[10]:


# Load the original image and convert to grayscale
original_image_path = '/Users/maria/Lloyd-max quantization/March/21 March/cars.png'
original_image = Image.open(original_image_path).convert('L')
original_image.save('cars_grayscale.png')
original_image = np.array(original_image)


# In[11]:


import cv2
import numpy as np

# Load the image
image_path = '/Users/maria/Lloyd-max quantization/March/21 March/cars.png'
image = cv2.imread(image_path)

# Determine the bit depth based on the dtype of the image array
if image.dtype == np.uint8:
    bit_depth = 8
elif image.dtype == np.uint16:
    bit_depth = 16
elif image.dtype == np.float32:
    bit_depth = 32  

print(f"The image bit depth is: {bit_depth}")


# Wavelets applied

# In[12]:


import cv2
import numpy as np


# Load your image
image_path = '/Users/maria/Lloyd-max quantization/March/21 March/cars.png'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height, width = gray_image.shape

# Initialize the masks
foreground_mask = np.zeros((height, width), dtype=np.uint8)
background_mask = np.ones((height, width), dtype=np.uint8) * 255

# Fill in the foreground mask based on the segment coordinates
for segment in masks.xy:
    # Ensure the segment forms a closed shape
    segment = np.array(segment, dtype=np.int32)
    cv2.fillPoly(foreground_mask, [segment], 255)
    
    # Subtract these segments from the background mask:
    cv2.fillPoly(background_mask, [segment], 0)

    
# Create the foreground by applying the mask on the grayscale image
foreground = cv2.bitwise_and(gray_image, gray_image, mask=foreground_mask)

# Create the background by applying the inverse mask on the grayscale image
background = cv2.bitwise_and(gray_image, gray_image, mask=background_mask)

# Save or show the results
cv2.imwrite('foreground.jpg', foreground)
cv2.imwrite('background.jpg', background)


# In[13]:


# Apply wavelet transform to both segments
foreground_segment = cv2.bitwise_and(gray_image, gray_image, mask=foreground)
coeffs_foreground = apply_wavelet_transform(foreground_segment)
cA, (cH, cV, cD) = coeffs_foreground
background_segment = cv2.bitwise_and(gray_image, gray_image, mask=background)
coeffs_background = apply_wavelet_transform(background_segment)
cA_back, (cH_back, cV_back, cD_back) = coeffs_background


# In[14]:


#image_path = '/Users/maria/Lloyd-max quantization/March/21 March/cars.png'
#gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  


# In[15]:


import numpy as np
from PIL import Image
import os 

# Lists of bits for foreground and background
foreground_bits_list = [1, 2, 3, 4, 5, 6, 7, 8]
background_bits_list = [1, 2, 3, 4, 5, 6, 7, 8]

psnr_values = []
mse_values = []
ssim_values = []
bitrates = []  
shannon_foreground = []
shannon_background = []
total_shannon = []
max_bit_depths = []
bpp_values = []  
image_sizes_bits = [] 


data_range = 255 
reconstructed_image = np.copy(gray_image)

for background_bits in background_bits_list:
    for foreground_bits in foreground_bits_list:
        # Calculate the quantization levels
        foreground_quantization_levels = 2 ** foreground_bits
        background_quantization_levels = 2 ** background_bits
        
        shannon_foreground_per_iteration = []
        shannon_background_per_iteration = []
        
        
        # Quantize foreground coefficients
        quantized_cA = lloyd_max_quantization(cA.flatten(), foreground_quantization_levels).reshape(cA.shape)
        quantized_cH = lloyd_max_quantization(cH.flatten(), foreground_quantization_levels).reshape(cH.shape)
        quantized_cV = lloyd_max_quantization(cV.flatten(), foreground_quantization_levels).reshape(cV.shape)
        quantized_cD = lloyd_max_quantization(cD.flatten(), foreground_quantization_levels).reshape(cD.shape)
        quantized_coeffs_foreground = (quantized_cA, (quantized_cH, quantized_cV, quantized_cD))

        # Calculate Shannon entropy for foreground
        shannon_fore = np.mean([shannon_entropy(coeff) for coeff in [quantized_cA, quantized_cH, quantized_cV, quantized_cD]])
        shannon_foreground_per_iteration.append(shannon_fore)

        reconstructed_foreground = apply_inverse_wavelet_transform(quantized_coeffs_foreground)
        normalized_reconstructed_foreground = np.clip(reconstructed_foreground, 0, 255)
        reconstructed_image = np.where(foreground_mask == 255, normalized_reconstructed_foreground, reconstructed_image)
        
        # Quantize background coefficients
        quantized_cA_back = lloyd_max_quantization(cA_back.flatten(), background_quantization_levels).reshape(cA.shape)
        quantized_cH_back = lloyd_max_quantization(cH_back.flatten(), background_quantization_levels).reshape(cH.shape)
        quantized_cV_back = lloyd_max_quantization(cV_back.flatten(), background_quantization_levels).reshape(cV.shape)
        quantized_cD_back = lloyd_max_quantization(cD_back.flatten(), background_quantization_levels).reshape(cD.shape)
        quantized_coeffs_background = (quantized_cA_back, (quantized_cH_back, quantized_cV_back, quantized_cD_back))

        # Calculate Shannon entropy for background
        shannon_back = np.mean([shannon_entropy(coeff) for coeff in [quantized_cA_back, quantized_cH_back, quantized_cV_back, quantized_cD_back]])
        shannon_background_per_iteration.append(shannon_back)

        reconstructed_background = apply_inverse_wavelet_transform(quantized_coeffs_background)
        normalized_reconstructed_background = np.clip(reconstructed_background, 0, 255)
        reconstructed_image = np.where(background_mask == 255, normalized_reconstructed_background, reconstructed_image)

        # Sum the Shannon entropies for all segments in the foreground and background
        total_shannon_foreground = sum(shannon_foreground_per_iteration)
        total_shannon_background = sum(shannon_background_per_iteration)

        # Compute the total Shannon entropy for the iteration
        total_shannon_iteration = (total_shannon_foreground + total_shannon_background) / 2
        total_shannon.append(total_shannon_iteration)

        # Compute the maximum bit depth used in this iteration
        max_bit_depth = max(foreground_bits, background_bits)
        max_bit_depths.append(max_bit_depth)

        # Calculate and store metrics
        current_psnr = calculate_psnr(gray_image, reconstructed_image)
        psnr_values.append(current_psnr)
        current_mse = np.mean((gray_image - reconstructed_image) ** 2)
        mse_values.append(current_mse)
        current_ssim = ssim(gray_image, reconstructed_image, data_range=data_range)
        ssim_values.append(current_ssim)

        # Save the final reconstructed image after each iteration with a unique name
        final_image = Image.fromarray(reconstructed_image.astype(np.uint8))
        final_image_filename = f'cars_image__background{background_bits}_foreground{foreground_bits}.png'
        final_image.save(final_image_filename)


        # Calculate the image file size in bits
        image_size_bytes = os.path.getsize(final_image_filename)
        image_size_bits = image_size_bytes * 8
        image_sizes_bits.append(image_size_bits)

       # Calculate BPP based on this final image
        total_pixels = gray_image.shape[0] * gray_image.shape[1]
        exact_bpp = image_size_bits / total_pixels
        bpp_values.append(exact_bpp)  # Append BPP for this combination 

# Print all exact bpp values stored
print("Exact BPP values for all images:", bpp_values)


# In[16]:


psnr_values


# In[17]:


ssim_values


# In[18]:


len(psnr_values)


# In[19]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

num_backgrounds = 8 
num_foregrounds = 8 

# Prepare the plot
plt.figure(figsize=(10, 5))

# Generate colors from a colormap
colors = cm.rainbow(np.linspace(0, 1, num_backgrounds))

# Calculate and plot PSNR vs BPP for each background bit depth
for i in range(num_backgrounds):
    start_idx = i * num_foregrounds
    end_idx = start_idx + num_foregrounds
    plt.plot(bpp_values[start_idx:end_idx], ssim_values[start_idx:end_idx], marker='o', linestyle='-', color=colors[i], label=f'Background {i + 1}')

plt.title('Wavelets_SSIM vs BPP for Different Backgrounds')
plt.xlabel('Bits Per Pixel (BPP)')
plt.ylabel('Structural Similarity Index (SSIM)')
plt.grid(True)
plt.legend() 

plt.show()


# In[20]:


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

num_backgrounds = 8 
num_foregrounds = 8 

# Prepare the plot
plt.figure(figsize=(10, 5))

# Generate colors from a colormap
colors = cm.rainbow(np.linspace(0, 1, num_backgrounds))

# Calculate and plot PSNR vs BPP for each background bit depth
for i in range(num_backgrounds):
    start_idx = i * num_foregrounds
    end_idx = start_idx + num_foregrounds
    plt.plot(bpp_values[start_idx:end_idx], psnr_values[start_idx:end_idx], marker='o', linestyle='-', color=colors[i], label=f'Background {i + 1}')

plt.title('Wavelets_PSNR vs BPP for Different Backgrounds')
plt.xlabel('Bits Per Pixel (BPP)')
plt.ylabel('Peak Signal-to-Noise Ratio (PSNR)')
plt.grid(True)
plt.legend() 

plt.show()


# In[21]:


total_shannon


# In[22]:


len(total_shannon)


# In[23]:


shannon_fore


# In[24]:


for background_bits in background_bits_list:
    for foreground_bits in foreground_bits_list:

        # Print Shannon entropy values for the current iteration
        print(f"Iteration: Background Bits {background_bits}, Foreground Bits {foreground_bits}")
        print(f"Shannon Entropy for Foreground: {shannon_fore}")
        print(f"Shannon Entropy for Background: {shannon_back}")
        print("")  # Blank line for readability between iterations

        
print(f"Total Shannon Entropy  {total_shannon}")
        


# In[34]:


from PIL import Image
import os
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np

def compress_and_evaluate(image_path, output_folder):
    """
    Compresses an image at different qualities, saves them, and evaluates BPP, PSNR, and SSIM.

    Args:
    - image_path: Path to the original image file.
    - output_folder: Folder where the compressed images will be saved.
    - qualities: List of quality levels for JPEG compression.
    """
    # Load the original image
    original_img = Image.open(image_path)
    original_img_array = np.array(original_img)
    
    # Convert image to RGB if not already (JPEG doesn't support alpha channel)
    if original_img.mode in ("RGBA", "P"):
        original_img = original_img.convert("RGB")
    
    original_size = original_img.size
    total_pixels = original_size[0] * original_size[1]
    
    # Iterate through each quality level, compress, and evaluate
    for quality in range(1, 101):
        output_path = os.path.join(output_folder, f"Quality_{quality}.jpg")
        original_img.save(output_path, 'JPEG', quality=quality)
        
        # Calculate BPP
        jpeg_image_size_bytes = os.path.getsize(output_path)
        jpeg_image_size_bits = image_size_bytes * 8
        jpeg_bpp = image_size_bits / total_pixels
        
        # Calculate PSNR and SSIM
        jpeg_compressed_img = Image.open(output_path)
        jpeg_compressed_img_array = np.array(jpeg_compressed_img)
        
        # Ensure the original image is in the same color space as the compressed one
        if original_img.mode != jpeg_compressed_img.mode:
            original_img = original_img.convert(jpeg_compressed_img.mode)
            original_img_array = np.array(original_img)
        
        jpeg_psnr_value = psnr(original_img_array, jpeg_compressed_img_array)
        jpeg_ssim_value = ssim(original_img_array, jpeg_compressed_img_array, data_range=jpeg_compressed_img_array.max() - jpeg_compressed_img_array.min())
        
        print(f"Quality {quality}: BPP = {jpeg_bpp}, PSNR = {jpeg_psnr_value}, SSIM = {jpeg_ssim_value}")


image_path = '/Users/maria/Lloyd-max quantization/March/21 March/cars_grayscale.png'  
output_folder = '/Users/maria/Lloyd-max quantization/March/21 March/JPEG Images'  
compress_and_evaluate(image_path, output_folder)


# In[ ]:




