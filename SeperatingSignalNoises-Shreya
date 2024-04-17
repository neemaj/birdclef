1. Use Thresholding to Separate Signal from Noise
- First must denoise signal by takeing the forward double-density DWT over four scales
- Then we can approach to soft thresholding method sets coefficients with values less 
  than the threshold T to 0, then subtracts T from the non-zero coefficients
- After we take the inverse wavelet transform of the new wavelet coefficients
-> example code:
function y = double_S1D(x,T)

% x: noise signal
% T: threshold

[af, sf] = filters1; 
J = 4;
w = double_f1D(x,J,af);
% loop through scales
for j = 1:J
    % loop through subbands
    for s = 1:2
       w{j}{s} = soft(w{j}{s},T);
    end
end
y = double_i1D(w,J,sf);

- RMS Error vs. Threshold Point plot

A certain threshold can be applied to a spectagaram 

Adaptive Thresholding: Instead of using a fixed threshold level, adaptive methods adjust the threshold 
  based on the statistics of the surrounding data. This can be more effective in diverse recording environments where noise levels vary.
Median Filtering: This involves using the median values within a neighborhood to decide if a point 
  in the spectrogram should be considered part of the signal or noise. Points significantly higher than the median are 
  likely part of the signal.

BINARY EROSION
The main goal of binary erosion is to reduce or eliminate small, isolated points of noise in binary images.
Erosion uses a structuring element (or kernel), which is a small matrix used to probe and reduce the shapes 
  within an image. The kernel has a defined shape and size, such as a square or circle.
  The kernel is moved over the image (like a sliding window), and for each position, it 
  considers the neighborhood defined by the kernel shape. If all pixels in this neighborhood are 1s, 
the pixel in the center is left as 1; otherwise, it is set to 0. 
src: The image which is to be eroded
kernel: The kernel matrix
iterations : (Optional) The number of iterations that specify how many times the operation will be performed. The default value is 1r
ex:kernel = np.ones((7,7),np.uint8) 

    eroded_image = cv2.erode(masked_image,kernel,iterations = 3)

BINARY DILATION
used to expand the areas of foreground pixels. 
  It is typically used after erosion to restore the size of an eroded signal and to fill in small holes and gaps in the data.
dilation also uses a structuring element to probe the image. 
  The shape and size of the kernel can vary based on the specific requirements of the application. 
   The kernel is again slid over the image, and for each position, if at least one pixel under the kernel is 1, 
  the pixel in the center of the kernel area in the output image is set to 1. This operation increases the size of 
  objects and fills in gaps and holes.
In detail, the grayscale morphological 2-D dilation is the max-sum correlation (for consistency with conv2d, we use unmirrored filters):


output[b, y, x, c] =
   max_{dy, dx} input[b,
                      strides[1] * y + rates[1] * dy,
                      strides[2] * x + rates[2] * dx,
                      c] +
                filters[dy, dx, c]

Application:
choice of the kernel size and shape is crucial. 
  larger kernel makes the operations more aggressive, which might be necessary in noisy 
  conditions but can also lead to loss of important signal details. 
LIbaries 
