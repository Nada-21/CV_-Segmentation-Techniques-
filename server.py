import streamlit as st
import cv2
from thresholding import*
from Segmentation import*
from skimage.filters import threshold_local

#...........................Convert RGB To LUV................................
def rgb_to_luv(rgb_img):
    # Convert RGB to XYZ
    rgb_normalized = rgb_img.astype(np.float32) / 255.0
    r, g, b = np.split(rgb_normalized, 3, axis=2)
    x = 0.412453 * r + 0.35758 * g + 0.180423 * b
    y = 0.212671 * r + 0.71516 * g + 0.072169 * b
    z = 0.019334 * r + 0.119193 * g + 0.950227 * b
    xyz_img = np.concatenate([x, y, z], axis=2)
    # Convert XYZ to LUV
    un = 0.19793943
    vn = 0.46831096
    x, y, z = np.split(xyz_img, 3, axis=2)
    u = 4*x / (x + 15*y + 3*z)
    v = 9*y / (x + 15*y + 3*z)
    delta = 0.008856
    k = 903.3
    y_linear = np.where(y > delta,116*(y**(1/3))-16, (k*y))
    l =  y_linear 
    u_ = 13 * l * (u - un)
    v_ = 13 * l * (v - vn)
    # Scale L, U, and V
    l_scaled = np.clip(l * 255 / 100, 0, 255).astype(np.uint8)
    u_scaled = np.clip((u_ + 134) * 255 / 354, 0, 255).astype(np.uint8)
    v_scaled = np.clip((v_ + 140) * 255 / 256, 0, 255).astype(np.uint8)
    # Stack LUV channels and return
    luv_img = np.concatenate([l_scaled, u_scaled, v_scaled], axis=2)
    return luv_img
#.........................................................................................

# def on_mouse(event, x, y, flags, params):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         clicks.append((y,x))

def cluster_and_plot_agglo(n_clusters,pixels,input_img):
    agglo = AgglomerativeClustering(k=n_clusters, initial_k=25)
    agglo.fit(pixels)

    new_img = [[agglo.predict_center(pixel) for pixel in row] for row in input_img]
    new_img = np.array(new_img, np.uint8)
    return new_img

st.set_page_config(page_title=" Image Processing", page_icon="📸", layout="wide",initial_sidebar_state="collapsed")

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_st_style, unsafe_allow_html=True)

with open("style.css") as source_des:
    st.markdown(f"""<style>{source_des.read()}</style>""", unsafe_allow_html=True)

side = st.sidebar
uploaded_img =side.file_uploader("Upload Image",type={"png", "jpg", "jfif" , "jpeg"})
col1,col2 =side.columns(2)
Knum = col2.number_input("K Number",min_value=1,max_value=20,value=3)
means = col1.number_input("Initial Points For K-Means",min_value=1,max_value=20,value=5)
window_size = side.number_input("Window Size", min_value= 20 , max_value= 100, value= 21)
Number_of_local_minimas = side.number_input("Number of local minimas For Region Growing",min_value=1,max_value=200,value=3)
Threshold_on_the_euclidean_distance =side.number_input("Threshold on the euclidean distance For Region Growing",min_value=1,max_value=20,value=4)
n_clusters = side.number_input("Number of clusters For Agglomerative",min_value=1,max_value=20,value=2)
threshold = side.number_input("Threshold For Mean Shift Segmentation",min_value=1,max_value=20,value=1)


tab1, tab2  = st.tabs(["Thresholding", "Segmentation"])

with tab1:
    uploadimg,result = st.columns(2)
    select = result.selectbox("Select Thresholding Method",("",'Optimal Thresholding','Otsu Thresholding',"Spectral Thresholding","Local Thresholding"))
   
    if uploaded_img is not None:
        file_path = 'Images/'  +str(uploaded_img.name)
        input_img = cv2.imread(file_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img = cv2.resize(input_img,(300,300))
        uploadimg.image(input_img)

        if select == "Optimal Thresholding":
            output_image=optimal_thresholding(input_img)
            result.image(output_image)
        if select == "Spectral Thresholding":
            output_image=spectral_thresholding(input_img)
            result.image(output_image )

        if select=="Otsu Thresholding":
            output_image = otsu_thresholding(input_img)
            # open cv just for comparison 
            # ret, output_image = cv2.threshold(input_img, 120, 255, cv2.THRESH_BINARY + 
            #                                 cv2.THRESH_OTSU)
            result.image(output_image)

        if select == "Local Thresholding":
            output_image = Local_threshold(input_img, window_size)
            result.image(output_image)   


with tab2:
    uploadimg,result = st.columns(2)
    select = result.selectbox("Select Segmentation Method",("",'K-Means','Region Growing',"Agglomerative","Mean Shift"))
    
    if uploaded_img is not None:
        file_path = 'Images/'  +str(uploaded_img.name)
        input_img = cv2.imread(file_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        luv_image = rgb_to_luv(input_img)
        uploadimg.image(input_img)
        
        uploadimg.image(luv_image)

        if select == "K-Means":
            output_image, exe_time = kmeans(input_img, Knum, means)
            result.image(output_image)
            result.text("Total time elapsed (s): " + exe_time )

        if select == "Region Growing":
            img_samp = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            gt = img_Resize(file_path,(img_samp.shape[1],img_samp.shape[0]))
            f, output_image,markers=region_growing_optimal(img_samp,input_img,gt,T=Number_of_local_minimas,connectivity=Threshold_on_the_euclidean_distance)
            result.image(output_image)

        if select == "Agglomerative":
            pixels = input_img.reshape((-1,3))
            output_image=cluster_and_plot_agglo(n_clusters,pixels,input_img)
            result.image(output_image)

        if select == "Mean Shift":
            output_image, exe_time = mean_shift(input_img, window_size, threshold)
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
            result.image(output_image)
            result.text("Total time elapsed (s): " + exe_time )
