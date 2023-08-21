import streamlit as st
import av
import csv
import pandas as pd
import numpy as np 
import pickle 
import mediapipe as mp
from landmarks import landmarks
import time
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
import pickle

current_stage = ''
counter = 0 
bodylang_prob = np.array([0,0]) 
bodylang_class = '' 
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)
init = False
model_name = './data/finalise_model.sav' #sports name can be changed with s_option
coords = './data/coords.csv'
num_coords = 33


if "pose" not in st.session_state:
	st.session_state.pose = ""
else:
	pose_name = st.session_state.pose

def calculate_angle(a,b,c): #calculation of the angles of the joints
	a = np.array(a) # First
	b = np.array(b) # Mid
	c = np.array(c) # End
	
	radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
	angle = np.abs(radians*180.0/np.pi)
	
	if angle >180.0:
		angle = 360-angle
		
	return angle


# countdown timer 
def countdown(t):
	while t: # while t > 0 for clarity 
		mins = t // 60
		secs = t % 60
		timer = '{:02d}:{:02d}'.format(mins, secs)
		print(timer, end="\r") # overwrite previous line 
		time.sleep(1)
		t -= 1

def reset_counter(): 
	global counter
	counter = 0 

with open('deadlift.pkl', 'rb') as f: 
	model = pickle.load(f)

def callback(frame):
    global current_stage
    global counter
    global bodylang_class
    global bodylang_prob

    image = frame.to_ndarray(format="bgr24")

    results = pose.process(image)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius = 2),
                                  mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius = 2))

        try:
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row], columns = landmarks)
            bodylang_prob = model.predict_proba(X)[0]
            bodylang_class = model.predict(X)[0]
            if bodylang_class == "down" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
                current_stage = "down"
            elif current_stage == "down" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
                current_stage = "up"
                counter += 1
            text = bodylang_class + ": " + str(bodylang_prob[bodylang_prob.argmax()])
            # Convert the image to RGB for PIL
            image_pil = Image.fromarray(image[..., ::-1])
            draw = ImageDraw.Draw(image_pil)
            font = ImageFont.load_default()  # Provide the path to your font file if not in default location
            draw.text((0, 0), text, font=font, fill=(255, 255, 255))
            # Convert the image back to BGR
            image = np.array(image_pil)[:, :, ::-1]

        except Exception as e:
            print(e)
    else:
        print("No landmarks found")

    img = image[:, :500, :]

    return av.VideoFrame.from_ndarray(img, format="bgr24")


def analysis_callback(frame, model):
    image = frame.to_ndarray(format="bgr24")
    results = pose.process(image)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius = 2),
                                  mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius = 2))

        try:
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row], columns=landmarks)
            bodylang_prob = model.predict_proba(X)[0]
            bodylang_class = model.predict(X)[0]
            text = bodylang_class + ": " + str(bodylang_prob[bodylang_prob.argmax()])
            # Convert the image to RGB for PIL
            image_pil = Image.fromarray(image[..., ::-1])
            draw = ImageDraw.Draw(image_pil)
            font = ImageFont.load_default()
            draw.text((0, 0), text, font=font, fill=(255, 255, 255))
            # Convert the image back to BGR
            image = np.array(image_pil)[:, :, ::-1]

        except Exception as e:
            print(e)
    else:
        print("No landmarks found")

    img = image[:, :500, :]

    return av.VideoFrame.from_ndarray(img, format="bgr24")

def photo_callback(image, model):
    results = pose.process(image)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius = 2),
                                  mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius = 2))

        try:
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            X = pd.DataFrame([row], columns=landmarks)
            bodylang_prob = model.predict_proba(X)[0]
            bodylang_class = model.predict(X)[0]
            text = bodylang_class + ": " + str(bodylang_prob[bodylang_prob.argmax()])
            # Convert the image to RGB for PIL
            image_pil = Image.fromarray(image[..., ::-1])
            draw = ImageDraw.Draw(image_pil)
            font = ImageFont.load_default()
            draw.text((0, 0), text, font=font, fill=(255, 255, 255))
            # Convert the image back to BGR
            image = np.array(image_pil)[:, :, ::-1]

        except Exception as e:
            print(e)
    else:
        print("No landmarks found")

    img = image[:, :500, :]
    return img  # , bodylang_class, bodylang_prob


def train_callback(frame, pose_name):
    image = frame.to_ndarray(format="bgr24")
    results = pose.process(image)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius = 2),
                                  mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius = 2))

        try:
            row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
            with open('./data/coords.csv', mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                row.insert(0, pose_name)
                csv_writer.writerow(row)

        except Exception as e:
            print(e)
    else:
        print("No landmarks found")
    img = image[:, :500, :]
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def initialise_file():
	num_coords = 33
	landmarks = ['class']
	for val in range(1, num_coords+1):
		landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
	with open('./data/coords.csv', mode='w', newline='') as f:
		csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		csv_writer.writerow(landmarks)


def train_data_model():
	dft = pd.read_csv('./data/coords.csv')
	X = dft.drop('class', axis = 1)
	y = dft['class']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
	pipelines = {#'lr':make_pipeline(StandardScaler(), LogisticRegression()), #Prediction Probability
				#'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
				'rf':make_pipeline(StandardScaler(), RandomForestClassifier()), #Prediction Probability
				#'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
				} #Isotonic Regression is another training model with prediction probability

	fit_models = {}
	for algo, pipeline in pipelines.items():
		model = pipeline.fit(X_train, y_train)
		fit_models[algo] = model
	st.write("Training Completed")    
	pickle.dump(fit_models['rf'], open(model_name, 'wb'))   
	st.write("save model_name")
