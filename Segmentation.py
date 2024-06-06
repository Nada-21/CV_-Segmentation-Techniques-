import pandas as pd
import numpy as np
import cv2
# from tqdm import tqdm
import time 
from scipy.linalg import eigh
np.set_printoptions(suppress=True) 
from skimage.feature import peak_local_max
from tqdm.notebook import tqdm
import random

#......................... Segmentation Using K-Means ..........................................
def kmeans(image, k, means):
    start = time.time()
    input_image = image/255  #Normalise the image
    input_image = input_image.astype(np.float32)
    #convert the MxNx3 image to a Kx3 image where k = MxN
    vectorised = input_image.reshape((-1,3))
    #Convert the array to a dataframe
    input_image_df = pd.DataFrame(vectorised)
    input_image_df.rename(columns={0:'R', 1:'G', 2: 'B'}, inplace =True)
    #taking random centroids for initial tests
    centroids = input_image_df.sample(means)

    X = input_image_df
    diff = 1
    j=0
    while(abs(diff)>0.05):
        XD=X
        i=1
        #iterate over each centroid point 
        for index1,row_c in centroids.iterrows():
            ED=[]
            #iterate over each data point
            print("Calculating distance")
            for index2,row_d in tqdm(XD.iterrows()):
                #calculate distance between current point and centroid
                d1=(row_c["R"]-row_d["R"])**2
                d2=(row_c["G"]-row_d["G"])**2
                d3=(row_c["B"]-row_d["B"])**2
                d=np.sqrt(d1+d2+d3)
                ED.append(d) #append disstance in a list 'ED'
            X[i]=ED  #append distace for a centroid in original data frame
            i=i+1

        C=[]
        print("Getting Centroid")
        for index,row in tqdm(X.iterrows()):
            min_dist=row[1]  #get distance from centroid of current data point
            pos=1
            #loop to locate the closest centroid to current point
            for i in range(k):
                if row[i+1] < min_dist:  #if current distance is greater than that of other centroids
                    #the smaller distanc becomes the minimum distance 
                    min_dist = row[i+1]
                    pos=i+1
            C.append(pos)
        #assigning the closest cluster to each data point
        X["Cluster"]=C
        #grouping each cluster by their mean value to create new centroids
        centroids_new = X.groupby(["Cluster"]).mean()[["R","G", "B"]]
        if j == 0:
            diff=1
            j=j+1
        else:
            #check if there is a difference between old and new centroids
            diff = (centroids_new['R'] - centroids['R']).sum() + (centroids_new['G'] - centroids['G']).sum() + (centroids_new['B'] - centroids['B']).sum()
            print(diff.sum())
        centroids = X.groupby(["Cluster"]).mean()[["R","G","B"]]

    centroids = centroids.to_numpy()
    labels = X["Cluster"].to_numpy()
    #overwritting the pixels values
    segmented_image = centroids[labels-1]
    segmented_image = segmented_image.reshape(input_image.shape)

    exe_time = str(time.time() - start)  # Ouput the execution time

    return segmented_image, exe_time

#.................................. Segmentation Using region growing ...................................................

class Queue:
      def __init__(self):
        self.items = []

      def isEmpty(self):
        return self.items == []

      def enqueue(self, item):
        self.items.insert(0,item)

      def dequeue(self):
        return self.items.pop()

      def size(self):
        return len(self.items)

def f_evaluataion(img, GT): # enter two thresholded gray images
    img[img > 0] = 1  # binary map of image
    GT[GT > 0] = 1  # binary map of GT
    e_gt = np.sum(img + GT == 2)
    e = np.sum(img)
    gt = np.sum(GT)
    if e == 0:
        return 0
    p = round(e_gt / e, 3)
    r = round(e_gt / gt, 3)
    f = round(2 * p * r / (p + r), 3)
#     return p, r, f
    if np.isnan(f):
        return 0
    return f

def img_Resize(path,size): #accepts path! not image
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, size, interpolation = cv2.INTER_AREA)
    return im

def generate_seeds(img_samp): #input: a gray image
    img_samp = cv2.GaussianBlur(img_samp ,(3,3),cv2.BORDER_DEFAULT) #apply some smoothing
    dist = 10
    flag = False
    #get local minimas of image intensity and use them as seeds
    while flag is False:
        seeds = peak_local_max(img_samp.max() - img_samp, min_distance=dist, indices=True) 
        #iterate until the seeds filtered to equal or less to 30 seeds (found this value by trials and errors)
        if seeds.shape[0]<= 30:
            flag = True
        dist+=10
    return seeds #return seeds which is list of coordinates


def region_growing_BFS(img_samp,img_samp_color,T,connectivity=8): #USING BFS and Queue DataStruct to find the growing region of a seed
    seeds = generate_seeds(img_samp) 
     #img_samp IS GrayScale IMAGE!!! 
    #img_samp_color IS RGB IMAGE!!! 
    rg_img = np.zeros(img_samp_color[:, :, 0].shape)
    ngbrs = [(0,-1),(0,1),(-1,0),(1,0),(-1, -1), (-1, 1), (1, -1), (1, 1)] # if connectivity is 4 than reads only first 4 items, else reads all 8
    height, width = img_samp_color[:, :, 0].shape
    for i in range(seeds.shape[0]):#
        seed = seeds[i]
        q = Queue()
        q.enqueue(seed)
        ###APPLY BFS:
        count2 =0
        while q.size() > 0:
            p = q.dequeue()
            rg_img[p[0], p[1]] = 255
            for j in range(connectivity): # For each neighbor of the pixel, connectivity = 4 || 8
                # Compute the neighbor pixel position
                x_new = p[0] + ngbrs[j][0]
                y_new = p[1] + ngbrs[j][1]
                # Boundary Condition - check if the coordinates are inside the image
                check_inside = (x_new >= 0) & (y_new >= 0) & (x_new < height) & (y_new < width)
                # Check neighbor if inside boundries and not already "labeled" (inside rg_img)
                if check_inside:
                    if rg_img[x_new, y_new] == 0:
                        #change the threshold to RGB EUCLIDEAN DISTANCE
                        R = int(img_samp_color[p[0], p[1], 0])
                        G = int(img_samp_color[p[0], p[1], 1])
                        B = int(img_samp_color[p[0], p[1], 2])
                        RR = int(img_samp_color[x_new, y_new, 0])
                        GG = int(img_samp_color[x_new, y_new, 1])
                        BB = int(img_samp_color[x_new, y_new, 2])
                        dist = ((RR - R) ** 2 + (GG - G) ** 2 + (BB - B) ** 2) ** 0.5 #euclidean distance between the RGB colors of a pixel and his neighbor
                        if dist < T: # if the pixels similar by defined Threshold, then label the pixel
                            q.enqueue((x_new, y_new))
                            rg_img[x_new, y_new] = 255
    return rg_img,seeds

def region_growing_optimal(img_samp,img_samp_color,gt,T=3,connectivity=4):
    t = T
    f = 0
    segmat,seeds = region_growing_BFS(img_samp,img_samp_color,t,connectivity)
    f_next = f_evaluataion(segmat, gt)
    while f_next > f:
        f = f_next
        t+=1
        best_segmat = segmat
        best_seeds = seeds
        segmat,seeds = region_growing_BFS(img_samp,img_samp_color,t,connectivity)
        f_next = f_evaluataion(segmat, gt)
    t+=1
    f_new = f_next
    segmat,seeds = region_growing_BFS(img_samp,img_samp_color,t,connectivity)
    f_next = f_evaluataion(segmat, gt)
    if f_next < f:
        return f,best_segmat,best_seeds
    
    while f_next > f_new:
        f_new = f_next
        t+=1
        best_segmat = segmat
        best_seeds = seeds
        segmat,seeds = region_growing_BFS(img_samp,img_samp_color,t,connectivity)
        f_next = f_evaluataion(segmat, gt)
        return f_new,best_segmat,best_seeds

#.................................. Segmentation Using Agglomerative ...................................................

def euclidean_distance(point1, point2):
   
    # Computes euclidean distance of point1 and point2.
    
   
    return np.linalg.norm(np.array(point1) - np.array(point2))

def clusters_distance(cluster1, cluster2):
  
    # Computes distance between two clusters.
    
    
    return max([euclidean_distance(point1, point2) for point1 in cluster1 for point2 in cluster2])
  
def clusters_distance_2(cluster1, cluster2):
    
    # Computes distance between two centroids of the two clusters
    
    cluster1_center = np.average(cluster1, axis=0)
    cluster2_center = np.average(cluster2, axis=0)
    return euclidean_distance(cluster1_center, cluster2_center)


class AgglomerativeClustering:
    
    def __init__(self, k=2, initial_k=25):
        self.k = k
        self.initial_k = initial_k
        
    def initial_clusters(self, points):
       
        # partition pixels into self.initial_k groups based on color similarity
        
        groups = {}
        d = int(256 / (self.initial_k))
        for i in range(self.initial_k):
            j = i * d
            groups[(j, j, j)] = []
        for i, p in enumerate(points):
            if i%100000 == 0:
                print('processing pixel:', i)
            go = min(groups.keys(), key=lambda c: euclidean_distance(p, c))  
            groups[go].append(p)
        return [g for g in groups.values() if len(g) > 0]
        
    def fit(self, points):

        # initially, assign each point to a distinct cluster
        print('Computing initial clusters ...')
        self.clusters_list = self.initial_clusters(points)
        print('number of initial clusters:', len(self.clusters_list))
        print('merging clusters ...')

        while len(self.clusters_list) > self.k:

            # Find the closest (most similar) pair of clusters
            cluster1, cluster2 = min([(c1, c2) for i, c1 in enumerate(self.clusters_list) for c2 in self.clusters_list[:i]],
                 key=lambda c: clusters_distance_2(c[0], c[1]))

            # Remove the two clusters from the clusters list
            self.clusters_list = [c for c in self.clusters_list if c != cluster1 and c != cluster2]

            # Merge the two clusters
            merged_cluster = cluster1 + cluster2

            # Add the merged cluster to the clusters list
            self.clusters_list.append(merged_cluster)

            print('number of clusters:', len(self.clusters_list))
        
        print('assigning cluster num to each point ...')
        self.cluster = {}
        for cl_num, cl in enumerate(self.clusters_list):
            for point in cl:
                self.cluster[tuple(point)] = cl_num
                
        print('Computing cluster centers ...')
        self.centers = {}
        for cl_num, cl in enumerate(self.clusters_list):
            self.centers[cl_num] = np.average(cl, axis=0)
                    


    def predict_cluster(self, point):
        """
        Find cluster number of point
        """
        # assuming point belongs to clusters that were computed by fit functions
        return self.cluster[tuple(point)]

    def predict_center(self, point):
        """
        Find center of the cluster that point belongs to
        """
        point_cluster_num = self.predict_cluster(point)
        center = self.centers[point_cluster_num]
        return center

#............................Segmentation Using Mean Shift Method..............................
def mean_shift(img,window=70,threshold=1.0):
    t1=time.time()

    def euclidean_distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2) ** 2))
    
    row, col, _ = img.shape
    segmented_image = np.zeros((row,col,3), dtype= np.uint8)
    feature_space   = np.zeros((row * col,5))
    counter=0 
    current_mean_random = True
    current_mean_arr = np.zeros((1,5))

    for i in range(0,row):
        for j in range(0,col):      
            feature_space[counter]=[img[i][j][0],img[i][j][1],img[i][j][2],i,j]
            counter+=1

    while(len(feature_space) > 0):
        print (len(feature_space))
        #selecting a random row from the feature space and assigning it as the current mean    
        if current_mean_random:
            current_mean_index = random.randint(0, feature_space.shape[0] - 1)
            current_mean_arr[0] = feature_space[current_mean_index]
        below_threshold_arr=[]

        distances = np.zeros(feature_space.shape[0])
        for i in range(0,len(feature_space)):
            distance = 0
            #Finding the eucledian distance of the randomly selected row i.e. current mean with all the other rows
            for j in range(0,5):
                distance += ((current_mean_arr[0][j] - feature_space[i][j])**2)
                    
            distances[i] = distance**0.5

            #Checking if the distance calculated is within the window. If yes taking those rows and adding 
            #them to a list below_threshold_arr
        below_threshold_arr = np.where(distances < window)[0]
        
        mean_color = np.mean(feature_space[below_threshold_arr, :3], axis=0)
        mean_pos = np.mean(feature_space[below_threshold_arr, 3:], axis=0)
        # Calculate Euclidean distance between mean color/position and current mean
        mean_color_distance = euclidean_distance(mean_color, current_mean_arr[0][:3])
        mean_pos_distance = euclidean_distance(mean_pos, current_mean_arr[0][3:])
        mean_e_distance = mean_color_distance + mean_pos_distance

        if(mean_e_distance < threshold):                
            new_arr = np.zeros((1,3))
            new_arr[0] = mean_color
            # When found, color all the rows in below_threshold_arr with 
            #the color of the row in below_threshold_arr that has i,j nearest to mean_i and mean_j
            current_mean_random = True
            segmented_image[feature_space[below_threshold_arr, 3].astype(int), feature_space[below_threshold_arr, 4].astype(int)] = new_arr
            # Remove below-threshold pixels from feature space
            feature_space[below_threshold_arr, :] = -1
            feature_space = feature_space[feature_space[:, 0] != -1]
            
        else:
            current_mean_random = False
            current_mean_arr[0, :3] = mean_color
            current_mean_arr[0, 3:] = mean_pos

    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    exe_time = str(time.time() - t1)  # Ouput the execution time

    return segmented_image, exe_time
