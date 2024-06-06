import cv2
import numpy as np
from scipy.fft import fft2, ifft2
import matplotlib.pyplot as plt


#.................................. thresholding using optimal threshold ...................................................
def optimal_thresholding(gray_image):

     # Maximum number of rows and cols for image
    max_x = gray_image.shape[1] - 1
    max_y = gray_image.shape[0] - 1
    first_corner = (gray_image[0, 0]).astype(int)
    second_corner = (gray_image[0, max_x]).astype(int)
    third_corner = (gray_image[max_y, 0]).astype(int)
    forth_corner= (gray_image[max_x, max_y]).astype(int)

    # Mean Value of Background Intensity, Calculated From The Four Corner Pixels
    background_mean = ( first_corner + second_corner + third_corner + forth_corner ) / 4
    Sum = 0
    Length = 0

    # Loop To Calculate Mean Value of object Intensity
    for i in range(0, gray_image.shape[1]):
        for j in range(0, gray_image.shape[0]):
            # Skip The Four Corner Pixels
            if not ((i == 0 and j == 0) or (i == max_x and j == 0) or (i == 0 and j == max_y) or (i == max_x and j == max_y)):
                Sum += gray_image[j, i]
                Length += 1
    object_mean = Sum / Length

    OldThreshold = (background_mean + object_mean) / 2
    NewThreshold = new_threshold(gray_image, OldThreshold)

    # Iterate untill the old and new threshold is equal
    while OldThreshold.any() != NewThreshold.any():
        OldThreshold = NewThreshold
        NewThreshold = new_threshold(gray_image, OldThreshold)
    thresh_img = np.zeros(gray_image.shape)
    thresh_img = np.uint8(np.where(gray_image >= NewThreshold, 255, 0))
    return thresh_img
def new_threshold(gray_image, Threshold):

    # Get Background Array, Consisting of All Pixels With Intensity Lower Than The Given Threshold
    new_background = gray_image[np.where(gray_image < Threshold)]
    # Get object Array, Consisting of All Pixels With Intensity Higher Than The Given Threshold
    new_object = gray_image[np.where(gray_image > Threshold)]

    new_background_mean = np.mean(new_background)
    new_object_mean = np.mean(new_object)
    # Calculate Optimal Threshold
    OptimalThreshold = (new_background_mean + new_object_mean) / 2
    return OptimalThreshold


#.................................. thresholding using spectral threshold ...................................................
def spectral_thresholding(gray_image):
    blur = cv2.GaussianBlur(gray_image,(5,5),0)
    hist = cv2.calcHist([blur],[0],None,[256],[0,256]) 
    hist /= float(np.sum(hist)) 
    ClassVarsList = np.zeros((256, 256))
    for bar1 in range(len(hist)):

        for bar2 in range(bar1, len(hist)):
            ForegroundLevels = []
            BackgroundLevels = []
            MidgroundLevels = []
            ForegroundHist = []
            BackgroundHist = []
            MidgroundHist = []
            for level, value in enumerate(hist):
                if level < bar1:
                    BackgroundLevels.append(level)
                    BackgroundHist.append(value)
                elif level > bar1 and level < bar2:
                    MidgroundLevels.append(level)
                    MidgroundHist.append(value)
                else:
                    ForegroundLevels.append(level)
                    ForegroundHist.append(value)
            
            FWeights = np.sum(ForegroundHist) / float(np.sum(hist))
            BWeights = np.sum(BackgroundHist) / float(np.sum(hist))
            MWeights = np.sum(MidgroundHist) / float(np.sum(hist))
            FMean = np.sum(np.multiply(ForegroundHist, ForegroundLevels)) / float(np.sum(ForegroundHist))
            BMean = np.sum(np.multiply(BackgroundHist, BackgroundLevels)) / float(np.sum(BackgroundHist))
            MMean = np.sum(np.multiply(MidgroundHist, MidgroundLevels)) / float(np.sum(MidgroundHist))
            BetClsVar = FWeights * BWeights * np.square(BMean - FMean) + \
                                                FWeights * MWeights * np.square(FMean - MMean) + \
                                                    BWeights * MWeights * np.square(BMean - MMean)
            ClassVarsList[bar1, bar2] = BetClsVar
        max_value = np.nanmax(ClassVarsList)
    threshold = np.where(ClassVarsList == max_value)[0][0]
    output_image = np.zeros_like(gray_image)
    output_image[gray_image > threshold] = 255
    return output_image


def otsu_thresholding(gray_image):
    HistValues = plt.hist(gray_image.ravel(), 256)[0]
    # print(hist)
    background, foreground = np.split(HistValues,[1])

    within_variance = []
    between_variance = []
    d = 0 
    for i in range(len(HistValues)):
        background, foreground = np.split(HistValues,[i])
        c1 = np.sum(background)/(gray_image.shape[0]* gray_image.shape[1])
        c2 = np.sum(foreground)/(gray_image.shape[0]*gray_image.shape[1])

        background_mean = np.sum([ intensity*frequency for intensity,frequency in enumerate(background)])/np.sum(background)
        background_mean = np.nan_to_num(background_mean)
        foreground_mean = np.sum([ (intensity + d)*(frequency) for intensity,frequency in enumerate(foreground)])/np.sum(foreground)
        
        background_variance = np.sum([(intensity - background_mean)**2*frequency for intensity,frequency in enumerate(background)])/np.sum(background)
        background_variance = np.nan_to_num(background_variance)
        foreground_variance = np.sum([(((intensity + d - foreground_mean)*(intensity + d - foreground_mean))*frequency) for intensity,frequency in enumerate(foreground)])/np.sum(foreground)

        d = d +1
        within_variance.append((c1*background_variance) + (c2*foreground_variance))
        between_variance.append(c1*c2*(background_mean-foreground_mean)*(background_mean-foreground_mean))

    min =np.argmin(within_variance)
    max=np.argmax(background_variance)

    thresh_img = np.uint8(np.where(gray_image >min, 255, 0))
    return thresh_img

def Local_threshold(gray_image, win_size ):
    # Create an empty binary image
    binary_img = np.zeros_like(gray_image)

# Loop through each pixel in the image
    for i in range(gray_image.shape[0]):
        for j in range(gray_image.shape[1]):
            # Calculate the local threshold using the Otsu method
            i_min = max(0, i - win_size // 2)
            i_max = min(gray_image.shape[0] - 1, i + win_size // 2)
            j_min = max(0, j - win_size // 2)
            j_max = min(gray_image.shape[1] - 1, j + win_size // 2)
            local_img = gray_image[i_min:i_max+1, j_min:j_max+1]
            threshold = np.mean(local_img) + 0.5 * (np.std(local_img) / 128 - 1)
        
            # Binarize the pixel based on the local threshold
            if gray_image[i, j] > threshold:
                binary_img[i, j] = 255
    return binary_img