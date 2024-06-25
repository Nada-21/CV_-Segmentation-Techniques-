# CV_-Segmentation-Techniques-

## Description:
A small web application based app developed with python and streamlit, to apply 
different image processing techniques.  

## Requirements:
• Python 3.  
• Streamlit 1.13.0  
• Numpy 1.23.4  
• Matplotlib 3.6.2  

## Running command:
Streamlit run server.py      
-The UI contains two main tabs Thresholding, Segmentation  
# Tab1:
• Optimal Thresholding  
• Otsu Thresholding  
• Spectral Thresholding  
• Local Thresholding  

### Optimal thresholding:
Minimize number of misclassified pixels if we have some prior 
knowledge about distribution of the gray levels values that make 
up the object and the background.
Is the one who divide histogram into two parts given that 
distribution of values at the same segment has minimum variance.  
![Screenshot (1476)](https://github.com/MayarFayez/CV_-Segmentation-Techniques-/assets/93496610/b8c38f77-4191-4ef1-a398-833c9a6aeb97)  
#### Basic steps for Optimal thresholding:
1. First approximation that the four corners of the image contain background pixels 
only and the reminder contains object pixels.  
2. At every step “t” compute mean of background and object gray level.  
3. Determine threshold value “T” at every step as the mid-point between the two 
means.  
4. Update the value of threshold until T(t+1) =T(t)
#### Results:
![Screenshot (1477)](https://github.com/MayarFayez/CV_-Segmentation-Techniques-/assets/93496610/c4fe0235-f851-4eba-95c0-405dba8d4d3a)  
### Otsu thresholding:
The algorithm iteratively searches for the threshold that minimizes the withinclass variance, defined as a weighted sum of variances of the two classes 
(background and foreground). The colors in grayscale are usually between 0-255 
(0-1 in case of float). So, if we choose a threshold of 100, then all the pixels with 
values less than 100 becomes the background and all pixels with values greater 
than or equal to 100 becomes the foreground of the image.  
The formula for finding the within-class variance at any threshold t is given by:  
𝝈2(𝒕) = 𝝎𝒃𝒈(𝒕)𝝈2𝒃𝒈(𝒕) + 𝝎𝒇𝒈(𝒕)𝝈2𝒇𝒈(𝒕) (1)  
![Screenshot (1478)](https://github.com/MayarFayez/CV_-Segmentation-Techniques-/assets/93496610/e51e13b1-03ae-4234-84cb-4dd7fe28bc39)  
#### Results:
![Screenshot (1479)](https://github.com/MayarFayez/CV_-Segmentation-Techniques-/assets/93496610/24b00c7f-fd2f-41ad-8148-f097b08fea30)  
### Spectral thresholding:
It is proposed for segmentation an image into multiple levels using the mean and 
variance starting from the extreme pixel values at both ends of histogram plot.  
![Screenshot (1480)](https://github.com/MayarFayez/CV_-Segmentation-Techniques-/assets/93496610/a56ccfb0-bbd4-4b1d-87b3-ab97c408c99d)  
#### Basic steps for Spectral thresholding:
1. Smooth image using gaussian filter.  
2. Calculate histogram and normalize it by dividing each bin by the total number 
of pixels in the image.  
3. The nested loop to iterates over all possible threshold values from 0 to 255.  
4. Divide the image pixels to 3 groups: background, foreground and midground.  
5. For each group calculate weights and mean intensity values then use these 
values to calculate variance for the current threshold value.  
#### Results:
![Screenshot (1481)](https://github.com/MayarFayez/CV_-Segmentation-Techniques-/assets/93496610/b558e8ad-9896-4488-bb0c-1cc32b2e200a)  
###  Local thresholding:
Local thresholding algorithms can enable local image regions such as brightness, 
contrast, and texture to have corresponding optimal thresholds. Common local 
threshold segmentation algorithms include the mean value of the local 
neighborhood blocks and the Gaussian weighted sum of the local neighborhood 
blocks.  
![Screenshot (1483)](https://github.com/MayarFayez/CV_-Segmentation-Techniques-/assets/93496610/ad3ce533-b092-4b06-be20-80a7db43c6f9)  
#### Parameters that user can enter them:
• Window size (it must be an odd number).  
#### Results:
![Screenshot (1484)](https://github.com/MayarFayez/CV_-Segmentation-Techniques-/assets/93496610/30eb87cb-e107-45e7-87ee-236fda8aebdc)  
# Tab2:
• map RGB to LUV  
• K-Means  
• Region Growing  
• Agglomerative  
• Mean Shift Methods  
### map RGB to LUV:
1. Convert RGB To XYZ using the following formulas.
![Screenshot (1485)](https://github.com/MayarFayez/CV_-Segmentation-Techniques-/assets/93496610/52592b4c-653f-45c0-97ee-e5e017dd1e27)
2. Convert XYZ to LUV as follows.  
 un = 0.19793943  
 vn = 0.46831096  
 u` = 4*x / (x + 15*y + 3*z)  
 v` = 9*y / (x + 15*y + 3*z)  
 l = 116*(y**(1/3))-16 for l> 0.008856  
OR l= 903.3*y for l<= 0.008856  
 u = 13 * l * (u` - un)  
 v = 13 * l * (v`- vn)  
3. Scaling is performed as follows.  
8U data type:  
Input image Algorithm output  
L = l * FW_MAX_8U / 100  
U = (u + 134) * FW_MAX_8U / 354  
V = (v + 140) * FW_MAX_8U / 256
#### Basic steps for map RGB to LUV:
1. Convert RGB to XYZ   
2. Convert XYZ to LUV  
3. Scale L, U, and V
#### Results: 
![Screenshot (1486)](https://github.com/MayarFayez/CV_-Segmentation-Techniques-/assets/93496610/54d8e9d5-5cfe-4156-89c9-1b0050de5b19)  
###  Segmentation Using K-Means:
K means clustering Initially assumes random 
cluster centers in feature space. Data are 
clustered to these centers according to the 
distance between them and centers. Now we 
can update the value of the center for each 
cluster, it is the mean of its points. Process is 
repeated and data are re-clustered for each 
iteration, new mean is calculated till 
convergence. Finally we have our centers and its related data points.  
![Screenshot (1487)](https://github.com/MayarFayez/CV_-Segmentation-Techniques-/assets/93496610/621ce1b5-1eec-4338-b1fc-adb7fcd5943e)  
#### Basic steps for k means segmentation:
1. Pick n data points that will act as the initial centroids.  
2. Calculate the Euclidean distance of each data point from each of the centroid points 
selected in step 1.  
3. Form data clusters by assigning every data point to whichever centroid it has the 
smallest distance from.  
4. Take the average of each formed cluster. The mean points are our new centroids.  
• Repeat steps 2 through 4 until there is no longer a change in centroids.
#### Parameters that user can enter them:
• K-Means Number  
• Initial Points for initial tests  
#### Results
![Screenshot (1488)](https://github.com/MayarFayez/CV_-Segmentation-Techniques-/assets/93496610/c595224f-b10f-4c92-8d57-198b20d7682e)  
###  Segmentation Using Region Growing:
Region growing method. this method is based on choosing seeds, and then for each 
seeds, finding his "similar" neighbors, and creating a region of those similar pixels, 
the tricky part of the method is which pixels to choose as seeds, and how many of 
those is necessary to choose in order to get a robust segmentation in this algorithm 
we use fully automated seeds pick - can be chosen using the mathemtical features of 
a image. i.e finding local minimas of an image. After picking the seeds, all is to find 
spread around the seed and to find similar pixels. We defined similiar pixels by 
euclidean distance of the RGB pixels between a seed and his neighbors, to wrap the 
algorithm up, to return the optimal segmentation, we assigned automated procedure 
on picking number of local minimas and threshold on the euclidean distance. by 
iterating, it finds the best yielded segmentation ( best measured by highest F-score).  
#### Basic steps for Region Growing segmentation:
1. Find the 30th (or less) minimal local minimas of the image. (30 is a chosen 
threshold by trials and errors)  
2. For each seed, apply BFS(Breadth-First Search) using Queue data structure, to 
find nearest similar neighbors, and add them to region:  
• q = Queue()  
• q.Enqueue(seed)  
• While q is not empty  
o p = q.Dequeuq()  
o For each neighbor Ƥ of p:  
if Ƥ and p similar(Euclidean RGB dist < Threshold) and Ƥ   
not already labeled then:  
q.Enqueue(Ƥ) , label(Ƥ)  
3. Start with initial threshold = 3, Apply region growing iterativly, with different 
threshold to find optimal threshold that yields robust segmentation  
#### Parameters that user can enter them:  
• Number of local minimas     
• Threshold on the euclidean distance  
#### Results
![Screenshot (1489)](https://github.com/MayarFayez/CV_-Segmentation-Techniques-/assets/93496610/2a368018-aa86-496b-8318-2f688fae0a88)  
###  Segmentation Using Agglomerative Methods:
Hierarchical clustering uses two different approaches to create clustersone of them is 
Agglomerative which is a bottom-up approach in which the algorithm starts with 
taking all data points as single clusters and merging them until one cluster is left.  
![Screenshot (1490)](https://github.com/MayarFayez/CV_-Segmentation-Techniques-/assets/93496610/205a627c-c9bd-4210-ba88-bc6dc6108771)  
Agglomerative starts joining data points of the dataset that are the closest to each 
other and repeats until it merges all of the data points into a single cluster containing 
the entire dataset.  
#### Basic steps for Agglomerative segmentation:
1. Take every data point as a separate cluster. If there are N data points, the 
number of clusters will be N  
2. Take the two closest data points or clusters and merge them to form a 
bigger cluster. The total number of clusters becomes N-1.  
3. Subsequent algorithm iterations will continue merging the nearest two 
clusters until only one cluster is left.  
4. Once the algorithm combines all the data points into a single cluster, it can 
build the dendrogram describing the clusters’ hierarchy  
5. Measuring distance bewteen two clusters using Euclidean distance  
6. Get the Dendrogram plot, where the x-axis shows all data points, and the y-axis shows the distance between them.   
7. Once we have the dendrogram for the clusters, we can set a threshold (a 
red horizontal dashed line) to visually see the number of output classes 
from the dataset after algorithm execution.  
#### Parameters that user can enter them:
• Number of clusters For Agglomerative  
#### Results:
![Screenshot (1491)](https://github.com/MayarFayez/CV_-Segmentation-Techniques-/assets/93496610/2c26f34a-414d-414e-9d16-19ee0ff00031)   
###  Segmentation Using Mean Shift Methods:
Mean Shift is also known as the mode-seeking 
algorithm that assigns the data points to the 
clusters in a way by shifting the data points 
towards the high-density region. The highest 
density of data points is termed as the model in the region.  
![Screenshot (1492)](https://github.com/MayarFayez/CV_-Segmentation-Techniques-/assets/93496610/e33cdf5b-c3f4-4547-bac6-828611fbf9cd)
![Screenshot (1493)](https://github.com/MayarFayez/CV_-Segmentation-Techniques-/assets/93496610/6ff9205d-b99f-4b70-adda-493221168cfa)  
#### Basic steps for mean-shift segmentation:
1. Getting the feature space of the input-image.  
2. Selecting a random row from the feature space and assigning it as the current 
mean.  
3. Finding the eucledian distance of the randomly selected row i.e. current mean 
with all the other rows.  
4. Checking if the distance calculated is within the window. If yes taking those rows 
and adding them to a list below_threshold_arr.  
5. Calculate Euclidean distance between mean color/position and current mean.  
6. When found, color all the rows in below_threshold_arr with the color of the row 
in below_threshold_arr that has i,j nearest to mean_i and mean_j.  
7. Remove below-threshold pixels from feature space.  
#### Parameters that user can enter them:
• Window Size  
• Threshold  
#### Results:
![Screenshot (1495)](https://github.com/MayarFayez/CV_-Segmentation-Techniques-/assets/93496610/939c96f7-2de2-49d9-a110-38cd8a8daeb5)



<table>
    <tbody>
    <tr>
        <td colspan="6" style="text-align: center;"><b> Team Members </b></td>
    </tr>
    <tr>
        <td align="center" valign="top" width="20%">
            <a href="https://github.com/Naira06">
                <img alt="Naira Youssef" src="https://avatars.githubusercontent.com/Naira06" width="100px;">
                <br/>
                <sub><b>Naira Youssef</b></sub>
            </a>
            <br/>
        </td>
        <td align="center" valign="top" width="20%">
            <a href="https://github.com/Nada-21">
                <img alt="Nada Ahmed" src="https://avatars.githubusercontent.com/Nada-21" width="100px;">
                <br/>
                <sub><b>Nada Ahmed</b></sub>
            </a>
            <br/>
        </td>
        <td align="center" valign="top" width="20%">
            <a href="https://github.com/Karemanyasser">
                <img alt="Kareman Yasser" src="https://avatars.githubusercontent.com/Karemanyasser" width="100px;">
                <br/>
                <sub><b>Kareman Yasser</b></sub>
            </a>
            <br/>
        </td>
        <td align="center" valign="top" width="20%">
            <a href="https://github.com/MayarFayez">
                <img alt="Mayar Fayez" src="https://avatars.githubusercontent.com/MayarFayez" width="100px;">
                <br/>
                <sub><b>Mayar Fayez</b></sub>
            </a>
            <br/>
        </td>
        <td align="center" valign="top" width="20%">
            <a href="https://github.com/GhofranMohamed">
                <img alt="Ghofran Mohamed" src="https://avatars.githubusercontent.com/GhofranMohamed" width="100px;">
                <br/>
                <sub><b>Ghofran Mohamed</b></sub>
            </a>
            <br/>
        </td>
    </tr>
    </table>
   





 







