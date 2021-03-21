import streamlit as st
import numpy as np

import pandas as pd
from PIL import Image
from keras.models import load_model
import cv2
import time


#import keras model
eye_model = load_model('sigmoid_final_cataract-90.h5')

st.set_page_config(page_title="AI EYE DIAGNOSIS", page_icon=None)
                   
#function for getting user input through streamlit app


#Eye Disease Detection. User input
def eye_user_input():
    
    uploaded_eye_img = st.sidebar.file_uploader("UPLOAD EYE IMAGE ...", type=["jpeg", "jpg"])
    
    if uploaded_eye_img is not None:
         #Load Image
         eye_image = Image.open(uploaded_eye_img)
         st.image(eye_image, caption='Eye Image', use_column_width=False, width=200)
         #add the predict button after the image is uploaded
        
         if st.button("Predict"): 
            predict_eye_disease(eye_image)
            

       
def predict_eye_disease(img):
    
         image = np.asarray(img)
         #resized_img = cv2.resize(image, (64, 64))
         #change to BGR(blue, green, red)
         resized_img= cv2.resize(image[:,:,::-1], (64, 64))
         reshaped_img = resized_img.reshape(1 ,64 , 64 , -1)
         
       
         cataract_pred= eye_model.predict(reshaped_img)
         st.write("")
       

         my_bar = st.progress(0)

         for percent_complete in range(100):
             time.sleep(0.01)
             my_bar.progress(percent_complete + 1)
                
         per = float(cataract_pred) *100 

    
         st.write('The possibility of patient having cataract is: {:.2f}%'.format(per))
        
     
         #st.write(per)
            
            


def main():

    #HTML code variables
    html_main_header = """
    <div style="background:#000000;padding:10px">
    <h2 style="color:white;text-align:center;">  Eye Diagnostic Tool </h2>
    </div>
    """
    # sub header: Eye Dissease Preciction ML App
    has_disease_html = """
    <div style="background:#ed7834 ;padding:1px">
    <h3 style="color:white;text-align:center;"> Patient has Eye Disease</h3>
    </div>
    """
    
    not_diabetic_html = """
    <div style="background:#5c92bf ;padding:1px">
    <h3 style="color:white;text-align:center;"> Patient does not have Eye Disease</h3>
    </div>
    """
    
    # Main Header 

    st.markdown(html_main_header, unsafe_allow_html = True)

    # Side Bar Header
    st.sidebar.image('https://assets-news-bcdn.dailyhunt.in/cmd/resize/400x400_80/fetchdata16/images/35/43/ca/3543ca4a8d2f8ef1f0440cbd9d4aa6641a7ebf92fbd5afec814c8864979ed085.png', use_column_width= True)
        
    #menu item selcction
    
    menu_items= [" ", "Cataract", "Diabetic Retinopathy"]
    
    choice = st.sidebar.selectbox("MENU", menu_items)   
    
    if choice == "Cataract Predictor":
        
      
        # PATIENT DATA
        st.subheader('Diabetic Retinopathy')
        
 
    elif choice == "Cataract":
        
        # PATIENT DATA
        st.subheader('Cataract')   
        eye_user_input()
        

    
if __name__=='__main__':
    main()

