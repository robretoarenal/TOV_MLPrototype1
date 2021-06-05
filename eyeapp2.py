import streamlit as st
import numpy as np

import pandas as pd
from PIL import Image
from keras.models import load_model
import cv2
import time


#import keras model
eye_model_1 = load_model('sigmoid_final_cataract-90.h5')
eye_model_2 = load_model('accuracy-92size100x100_Threshold_0.56.h5')

st.set_page_config(page_title="AI EYE DIAGNOSIS", page_icon=None)

#Below are HTML code variables
    
#used on main header
    
main_header = """
    <div style="background:#ffffff";padding:1px">
    <h2 style="color:black;text-align:center;">EyeSpark by inno*spark</h2>
    </div>
    """

cataract_header = """
    <div style="background:black";padding:10px">
    <h2 style="color:white;text-align:center;">Cataract Diagnosis</h2>
    </div>
    """

dr_header = """
    <div style="background:black";padding:10px">
    <h2 style="color:white;text-align:center;">Diabetic Retinopathy Diagnosis</h2>
    </div>
    """
  
dr_msg = """
    <div style="background:#fffffff">
    <h3 style="color:#000000;text-align:center;">Diabetic Retinopathy Diagnosis is not available in this version of EyeSpark.</h3>
    </div>
    """   
#variable used on side bar
side_header = """
    <div style="background:#fffffff">
    <h5 style="color:#000000;text-align:center;">Select Disease to Diagnose </h5>
    </div>
    """
# variable to display has disease message
has_disease_html = """
    <div style="background:#ed7834 ;padding:1px">
    <h3 style="color:white;text-align:center;"> Patient has Eye Disease</h3>
    </div>
    """
    
#variable to display no disease message
no_disease_html = """
    <div style="background:#5c92bf ;padding:1px">
    <h3 style="color:white;text-align:center;"> Patient does not have Eye Disease</h3>
    </div>
    """

                   
#get uploaded image from user input

def eye_user_input():
    
    uploaded_eye_img = st.sidebar.file_uploader("UPLOAD EYE IMAGE ...", type=["jpeg", "jpg"])
    
    if uploaded_eye_img is not None:
         #Load Image
         eye_image = Image.open(uploaded_eye_img)
         st.image(eye_image, caption='Eye Image', use_column_width=False, width=200)
         #add the predict button after the image is uploaded
        
         if st.button("Predict"): 
            predict_eye_disease_cataract_1(eye_image)
            
#function below is called when user selects cataract from main menu
def eye_user_input_cataract(): 
    
    st.sidebar.image(['997_right.jpg','eyeimage.jpeg'], width= 100, caption=['Retinal Image','Anterior Image'])
    image_types= ['Retinal Image', 'Anterior Image']
    selection = st.sidebar.radio("What type of image ar you uploading?", image_types)
    
    if selection == 'Retinal Image':
        eye_user_input_cataract_1()
    if selection == 'Anterior Image':
        eye_user_input_cataract_2()
               
    #st.sidebar.image(['997_right.jpg','rob.jpeg'], width=75, caption=['Retinal Image','Anterior Image'])
    #if st.sidebar.button('Upload Retinal Image'):  
               
               #eye_user_input_cataract_1()
               
                
    #st.sidebar.image('rob.jpeg', width=75, caption='Anterior Image')
    #if st.sidebar.button('Upload Anterior Image'):                 
               #eye_user_input_cataract_2()

           
def eye_user_input_cataract_1():
    
       
       uploaded_eye_img = st.file_uploader("UPLOAD RETINAL IMAGE", type=["jpeg", "jpg"])
    
       if uploaded_eye_img is not None:
             #Load Image
             eye_image = Image.open(uploaded_eye_img)
             st.image(eye_image, caption='Retinal Image Diagnosis', use_column_width=False, width=200)
             #add the predict button after the image is uploaded
        
             if st.button("Diagnose"): 
                 predict_eye_disease_cataract_1(eye_image)

def eye_user_input_cataract_2():
    
       
       uploaded_eye_img = st.file_uploader("UPLOAD ANTERIOR IMAGE", type=["jpeg", "jpg"])
    
       if uploaded_eye_img is not None:
             #Load Image
             eye_image = Image.open(uploaded_eye_img)
             st.image(eye_image, caption='Eye Image', use_column_width=False, width=200)
             #add the predict button after the image is uploaded
        
             if st.button("Diagnose"): 
                 predict_eye_disease_cataract_2(eye_image)
           
           
#predict using eye model 1                
def predict_eye_disease_cataract_1(img):
    
         image = np.asarray(img)      
         resized_img= cv2.resize(image[:,:,::-1], (64, 64))
         reshaped_img = resized_img.reshape(1 ,64 , 64 , -1)
              
         cataract_pred= eye_model_1.predict(reshaped_img)
         st.write("")
  
        #show progress bar
         my_bar = st.progress(0)

         for percent_complete in range(100):
             time.sleep(0.01)
             my_bar.progress(percent_complete + 1)
                
         per = float(cataract_pred) *100 

    
         st.write('The possibility of patient having cataract is: {:.2f}%'.format(per))
        
        
#predict using eye model 1                
def predict_eye_disease_cataract_2(img):
    
         image = np.asarray(img)      
         resized_img= cv2.resize(image[:,:,::-1], (100, 100))
         reshaped_img = resized_img.reshape(1 ,100 , 100 , -1)
              
         cataract_pred= eye_model_2.predict(reshaped_img)
         st.write("")
  
        #show progress bar
         my_bar = st.progress(0)

         for percent_complete in range(100):
             time.sleep(0.01)
             my_bar.progress(percent_complete + 1)
                
         per = float(cataract_pred) *100 

    
         st.write('The possibility of patient having cataract is: {:.2f}%'.format(per))
        
 #function below is called when user selects cataract from main menu
def dr_user_input_cataract():
     
        st.markdown(dr_header, unsafe_allow_html = True)  

def main():


    
    # Main Header -- uses main header variable to display main header
    st.markdown(main_header, unsafe_allow_html = True)

    # Side bar Header -- code below needed to properly format the side bar logo and text
    title_container = st.beta_container()
    col1, col2 = st.beta_columns([1, 20])
    image = Image.open('eyesparklogo.jpeg')
    with title_container:
         with col1:
                st.sidebar.image(image, width=100)
         with col2:
                st.sidebar.markdown('<h2 style="color: black;">EyeSpark Diagnostic Tool</h2>',
                            unsafe_allow_html=True)
                
    #side bar header to describe menu/ to ask user to select disease to diagnose
    st.sidebar.markdown(side_header, unsafe_allow_html = True)
        
    #menu item selcction
    
    menu_items= [" ", "Cataract","Diabetic Retinopathy"]
    
    choice = st.sidebar.selectbox(" ", menu_items)   
    
    if choice == "Diabetic Retinopathy":
        
      
        # PATIENT DATA
        dr_user_input_cataract()
        st.markdown(dr_msg, unsafe_allow_html = True)
        
 
    elif choice == "Cataract":
        
        # PATIENT DATA
        st.markdown(cataract_header, unsafe_allow_html = True) 
        #eye_user_input_cataract_1()
        eye_user_input_cataract()
        
    
             

    
if __name__=='__main__':
    main()
