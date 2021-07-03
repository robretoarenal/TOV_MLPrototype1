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

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from requests.packages.urllib3.exceptions import MaxRetryError

#import keras model-- the retinal model is not implementing with api so uploading here:
eye_model_1 = load_model('sigmoid_final_cataract-90.h5')
#eye_model_2 = load_model('accuracy-92size100x100_Threshold_0.56.h5')

#API_URL = 'http://tov-m-LoadB-RUXY2HD2AGFL-242d67356c99bf24.elb.us-east-1.amazonaws.com:80/eyesDiagnosis'
API_URL = 'http://tov-m-LoadB-RXZXM237C20Y-611ee7192043f9fb.elb.us-east-1.amazonaws.com:80/eyesDiagnosis'

st.set_page_config(page_title="AI EYE DIAGNOSIS", page_icon=None)

#Below are HTML code variables
    

cataract_header = """
    <div style="background:white";padding:0px">
    <h3 style="color:black;text-align:left;">Cataract Diagnosis</h3>    
    <h5 text-align:center;> What type of image are you uploading?</h5>
    </div>
    """

dr_header = """
    <div style="background:white";padding:0px">
    <h3 style="color:black;text-align:left;">Diabetic Retinopathy Diagnosis</h3>
     <h5 text-align:center;> Diabetic Retinopathy diagnosis is not available in this app version.</h5>
    </div>
    """
   
#variable used on side bar
side_header = """
    <div style="background:#fffffff">
    <h5 style="color:#000000;">Select a Disease to Diagnose </h5>
    </div>
    """
# variable to display has disease message
has_disease_html = """
    <div style="background:#ed7834 ;padding:1px">
    <h3 style="color:white;text-align:center;"> Patient has Eye Disease</h3>
    </div>
    """
#multiple faces detected 
multiple_faces_html = """
    <div style="background:#ed7834;padding:1px">
    <h3 style="color:white;text-align:center;"> Multiple Faces Detected! Please Upload One Face!</h3>
    </div>
    """
#no face detected 
noface_html = """
    <div style="background:#ed7834;padding:1px">
    <h3 style="color:white;text-align:center;"> No face detected! Please Upload a Human Face!</h3>
    </div>
    """
#variable to display  message
no_cataract_right_html = """
    <div style="background:#5c92bf ;padding:1px">
    <h3 style="color:white;text-align:center;"> Right Eye : No Cataract Detected</h3>
    </div>
    """
#variable to display no disease message
cataract_right_html = """
    <div style="background:#ed7834 ;padding:1px">
    <h3 style="color:white;text-align:center;"> Right Eye : Cataract Detected</h3>
    </div>
    """

# Cataract Preciction message
no_cataract_left_html = """
    <div style="background:#5c92bf ;padding:1px">
    <h3 style="color:white;text-align:center;"> Left Eye : No Cataract Detected</h3>
    </div>
    """
    
cataract_left_html = """
    <div style="background:#ed7834 ;padding:1px">
    <h3 style="color:white;text-align:center;"> Left Eye : Cataract Detected</h3>
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
    
    st.sidebar.write('Below are examples of retinal and anterior images:')
    #st.sidebar.image('997_right.jpg', width= 80, caption='Retinal Image')
    #st.sidebar.image('rob.jpeg', width= 50, caption='Anterior Image')
    
    p_cols = st.sidebar.beta_columns(2)
    p_cols[0].image('997_right.jpg', width= 85, caption='Retinal Image')
    p_cols[1].image('rob.jpeg', width= 55, caption='Anterior Image')
   
    image_types= ['Retinal Image', 'Anterior Image']
    selection = st.radio("   ", image_types)    
    
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
                        #st.write('No Face Detected. Please Upload a Face!')
                        st.markdown(noface_html,unsafe_allow_html = True)
                        #message = 'No Face Detected. Please Upload a Face!'
                        
                    #if value == 'success':
                        #st.write(message,'Face Detected! Processing...')
                        #message = 'Face Detected! Processing...'
                        
                    if value == 'Multiple faces detected':
                        #st.write('Multiple Faces Detected. Please Upload One Face!' )
                        st.markdown(multiple_faces_html,unsafe_allow_html = True)
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
       
        
        #resp = requests.post(url=API_URL, files=test_photo)
        t0 = time.time()
        try:
            resp = requests.models.Response()
            resp.status_code = 400

             #Controlling http Session
            session_ = requests.Session()
            allowed_methods_ = frozenset({'HEAD', 'GET', 'TRACE', 'POST'})
            status_forcelist_ = frozenset({502, 503, 504})
            retries_ = Retry(total=5, backoff_factor=1,allowed_methods = allowed_methods_, status_forcelist=status_forcelist_)
            session_.mount('http://', HTTPAdapter(max_retries=retries_))


            resp = session_.post(API_URL, files=test_photo, headers={'User-Agent': 'Mozilla/5.0'})

        except MaxRetryError as e:
            st.write(f"Failed due to: {e.reason}")    
        except Exception as e:
            if hasattr(e, 'message'):
                st.write(f"Failed due to {e.message}")
            else:
                st.write(f"Failed due to: {e}")  
    
        finally:
            t1 = time.time()
            #st.write(f"Took, {t1-t0}, seconds")
            #print(f"Retries info:{retries_.__dict__}")
            session_.close()

                       
           
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
                        
                    st.write(f"It took {round(t1-t0)} seconds to diagnose")    
                    
                    per_left = float(diagnosis_left) *100 
                    per_right = float(diagnosis_right) *100 

                    if desc_left == 'Left Eye : No Cataract detected':
                        st.markdown(no_cataract_left_html, unsafe_allow_html = True)
                    else:
                        st.markdown(cataract_left_html, unsafe_allow_html = True)
                    
                    st.write('The probability of left eye cataract is: {:.2f}%'.format(per_left))
                    st.image(cropped_left)
       
                    #st.write('---------------------------------------------')
                    #st.write(desc_right)
                    if desc_right == 'Right Eye : No Cataract detected':
                        st.markdown(no_cataract_right_html, unsafe_allow_html = True)
                    else:
                        st.markdown(cataract_right_html, unsafe_allow_html = True)
                        
                    st.write('The probability of right eye cataract is: {:.2f}%'.format(per_right))
                    st.image(cropped_right)
       
                
 #function below is called when user selects DR from main menu
def dr_user_input_cataract():
     
        st.markdown(dr_header, unsafe_allow_html = True)  

def main():
  # Main Header -- uses main header variable to display main header
    im_main = Image.open('eyesparklogo.jpeg')
    #st.markdown(main_header, unsafe_allow_html = True)
    #st.image(im_main, width=150)
    m_cols = st.beta_columns(3)
    m_cols[0].text("        ")
    m_cols[1].image(im_main, width=125) 
    m_cols[1].text("        ")
   
 
    # Side bar Header -- code below needed to properly format the side bar logo and text
  
    #image = Image.open('eyesparklogo.jpeg')
    image = Image.open('innoRect.jpg')
    
    # Side bar Header -- code below needed to properly format the side bar logo and text
    cols = st.sidebar.beta_columns(2)
  
    cols[0].image(image, width=250)
    cols[1].text(" ")
    
  
  
    
    
    
    
    st.sidebar.markdown('<h2 style="color: black;">EyeSpark Diagnostic Tool</h2>',unsafe_allow_html=True)
                
    #side bar header to describe menu/ to ask user to select disease to diagnose
    st.sidebar.markdown(side_header, unsafe_allow_html = True)
        
    #menu item selcction
    
    menu_items= [" ", "Cataract","Diabetic Retinopathy"]
    
    choice = st.sidebar.selectbox(" ", menu_items)   
    
    if choice == "Diabetic Retinopathy":
     
        # PATIENT DATA
        dr_user_input_cataract()
        #st.markdown(dr_msg, unsafe_allow_html = True)
        
 
    elif choice == "Cataract":
        
        # PATIENT DATA
        st.markdown(cataract_header, unsafe_allow_html = True) 
        #eye_user_input_cataract_1()
        eye_user_input_cataract()
 
    
if __name__=='__main__':
    main()
