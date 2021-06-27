import requests 

import streamlit as st
import numpy as np

import pandas as pd
from PIL import Image
from keras.models import load_model
import cv2
import time
from io import BytesIO
from base64 import decodebytes, encodebytes
from datetime import timedelta, datetime
import pathlib

#import keras model-- the retinal model is not implementing with api so uploading here:
eye_model_1 = load_model('sigmoid_final_cataract-90.h5')
#eye_model_2 = load_model('accuracy-92size100x100_Threshold_0.56.h5')

API_URL = 'http://tov-m-LoadB-RUXY2HD2AGFL-242d67356c99bf24.elb.us-east-1.amazonaws.com:80/eyesDiagnosis'

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
    <h3 style="color:#000000;text-align:center;">Diabetic Retinopathy Diagnosis is not available in this version of app.</h3>
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

def write_reponse_image(image_64_encode):
    
    #Parsed images directory
    curr_dir_path = pathlib.Path(__file__).parent.absolute()
    # parsed_img_dir = '/parsed_images/'  
    parsed_img_dir_full = str(curr_dir_path)
    #st.write('directory:', parsed_img_dir_full)
    #Decoding image_64_encode
    #image_64_decode = decodestring(image_64_encode)
    image_64_decode = decodebytes(image_64_encode.encode("ascii"))
    image_result_name_tmp = datetime.now().strftime("%Y%m%d%H%M%S%f") + '.jpg'
    tmp_img_name = parsed_img_dir_full+ "/" + image_result_name_tmp
    image_result = open(tmp_img_name, 'wb') # create a writable image and write the decoding result
    image_result.write(image_64_decode)
    
    #st.write('tmp_img_name', tmp_img_name)
    
    return tmp_img_name

#get uploaded image from user input
            
#function below is called when user selects cataract from main menu
def eye_user_input_cataract(): 
    
    st.sidebar.write('Below are examples of Retinal and Anterior Images:')
    st.sidebar.image('997_right.jpg', width= 100, caption='Retinal Image')
    st.sidebar.image('rob.jpeg', width= 70, caption='Anterior Image')
    image_types= ['Retinal Image', 'Anterior Image']
    selection = st.radio("What type of image are you uploading?", image_types)    
    
    if selection == 'Retinal Image':
        uploaded_img = st.file_uploader("UPLOAD RETINAL IMAGE", type=["jpeg", "jpg"])
        eye_user_input_cataract_1(uploaded_img)
        
    if selection == 'Anterior Image':
        uploaded_img = st.file_uploader("UPLOAD RETINAL IMAGE", type=["jpeg", "jpg"])
        eye_user_input_cataract_2(uploaded_img)
        
    
                  
       
def eye_user_input_cataract_1(uploaded_eye_img):
   
       if uploaded_eye_img is not None:
             #Load Image
             eye_image = Image.open(uploaded_eye_img)
             st.image(eye_image, caption='Retinal Image Diagnosis', use_column_width=False, width=100)
             #add the predict button after the image is uploaded
        
             if st.button("Diagnose"):              
                 predict_eye_disease_cataract_1(eye_image)
                                

def eye_user_input_cataract_2(anterior_img):
           
   
       if anterior_img is not None:
             #Load Image           
             eye_image = Image.open(anterior_img)
             st.image(eye_image, caption='Anterior Image', use_column_width=False, width=100)
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

         st.write('The probability of patient having cataract is: {:.2f}%'.format(per))

#print message

def print_message(key, value):
    
     if key == 'message':
                    
                    if value == 'No face detected':
                        st.write('No Face Detected. Please Upload a Face!')
                        #message = 'No Face Detected. Please Upload a Face!'
                        
                    #if value == 'success':
                        #st.write(message,'Face Detected! Processing...')
                        #message = 'Face Detected! Processing...'
                        
                    if value == 'Multiple faces detected':
                        st.write('Multiple Faces Detected. Please Upload One Face!' )                   
                        #message = 'Multiple Faces Detected. Please Upload One Face!' 
        
#predict using  API. API URL declared above  

def predict_eye_disease_cataract_2(img):
        
        imagefile = BytesIO()
        img.save(imagefile, format='PNG')
        imagedata = imagefile.getvalue()
        
        cropped_left = ''
        cropped_right= ''
        
        diagnosis_left = ''
        diagnosis_right =''
        
        
        desc_left = ''
        desc_right=''
      
        message = ''
      
        test_photo = {'file': imagedata} #Single Face
       
        
        resp = requests.post(url=API_URL, files=test_photo)
        
        st.write('Server Status:',resp)
       
        
        if resp.status_code != 200:
           # This means something went wrong and we want to know what happened
           st.write("Something went wrong on Server, try again!")
        
        else:
            json_pairs = resp.json().items()
            #st.write('json',json_pairs)
              
            for key, value in json_pairs:
                #st.write('key', key)
                #st.write('value', value)
                print_message(key, value)
                                   
               
                #if key == 'message':
                    
                    #if value == 'No face detected':
                        #st.write('No Face Detected. Please Upload a Face!')
                        #message = 'No Face Detected. Please Upload a Face!'
                        
                    #if value == 'success':
                        #st.write(message,'Face Detected! Processing...')
                        #message = 'Face Detected! Processing...'
                        
                    #if value == 'Multiple faces detected':
                        #st.write('Multiple Faces Detected. Please Upload One Face!' )                   
                        #message = 'Multiple Faces Detected. Please Upload One Face!' 
                
                if key == 'data'and value !=[]:
              
                    st.write(message,'Face Detected! Processing...')
                    
                    dic_data = value[0]
                    for data_key in dic_data:
                        if data_key == 'left_eye_im':
                            cropped_left= write_reponse_image(dic_data[data_key])
                            
                        if data_key == 'right_eye_im':
                            cropped_right = write_reponse_image(dic_data[data_key])
                            
                        if data_key == 'left_eye_im_diagnosis':
                            diagnosis_left= dic_data[data_key]
                            
                        if data_key == 'right_eye_im_diagnosis':
                            diagnosis_right= dic_data[data_key]
                            
                        if data_key == 'left_eye_im_desc':
                            desc_left =dic_data[data_key]
                            
                        if data_key == 'right_eye_im_desc':
                            desc_right =dic_data[data_key]
                    
                    
                    
                    #show progress bar
                    my_bar = st.progress(0)
                    
                    for percent_complete in range(100):
                        time.sleep(0.01)
                        my_bar.progress(percent_complete + 1)
                        
                    
                    st.write(desc_left)
                    st.write('The probability of left eye cataract is:',diagnosis_left)
                    st.image(cropped_left)
       
                    #st.write('---------------------------------------------')
                    st.write(desc_right)
                    st.write('The probability of right eye cataract is',diagnosis_right)
                    st.image(cropped_right)
       
                
 #function below is called when user selects DR from main menu
def dr_user_input_cataract():
     
        st.markdown(dr_header, unsafe_allow_html = True)  

def main():
  # Main Header -- uses main header variable to display main header
    st.markdown(main_header, unsafe_allow_html = True)

    # Side bar Header -- code below needed to properly format the side bar logo and text
  
    image = Image.open('eyesparklogo.jpeg')
    
    # Side bar Header -- code below needed to properly format the side bar logo and text
    cols = st.sidebar.beta_columns(3)
    space1 = cols[0].image(image, width=100)
    space2 = cols[1].text("   ")
    space3 = cols[0].text("   ")
    
    st.sidebar.markdown('<h2 style="color: black;">EyeSpark Diagnostic Tool</h2>',unsafe_allow_html=True)
                
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
