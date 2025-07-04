######## Webcam Object Detection Using Tensorflow-trained Classifier #########
#V 17 from macbook testing tailscale 
#Code based from Evan Juras Turotior
#Other fearutes are by Josue Tristan 
# Date: 10/27/19 created by author
# Being used by the CCC robot ece410
# Description: 
# This program uses a TensorFlow Lite model to perform object detection on a live webcam
# feed. It draws boxes and scores around the objects of interest in each frame from the
# webcam. To improve FPS, the webcam object runs in a separate thread from the main program.
# This script will work with either a Picamera or regular USB webcam.
#
# This code is based off the TensorFlow Lite image classification example at:
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py
#
# I added my own method of drawing boxes and labels using OpenCV.
# this code was based off a 3 part tutorial, more information in the report, and is being used for our goal
# Import packages
import os
import argparse
import threading
import cv2
import numpy as np
import sys
import time
from time import sleep
from threading import Thread
import threading
import importlib.util
import right_motora
import left_motora
import str_motora
from Rosmaster_Lib import Rosmaster
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import DistanceSensor

factory = PiGPIOFactory()
sensor = DistanceSensor(echo=24, trigger=23, pin_factory=factory)


#creates the bot object
bot = Rosmaster()
bot.create_receive_threading()
pid_values = bot.get_motion_pid()
print(f"Current PID values: KP={pid_values[0]} KI={pid_values[1]} KD={pid_values[2]}")
#it shounld be kp = 0.2 ki = 0.0 kd = 4.0 by defult 
#but on the video its 0.5 0.1 0.3 
print("Current PID:", bot.get_motion_pid())



#Set initial PID parameters
bot.set_pid_param(kp=0.8, ki=0.01, kd=0.7, forever=False) #tune these later
print(f"Current PID values: KP={pid_values[0]} KI={pid_values[1]} KD={pid_values[2]}")
print("Current PID:", bot.get_motion_pid())

#sensor = DistanceSensor(echo=24,trigger=23)

#sets the the webcam window to 896 x 504
IM_WIDTH = 896
IM_HEIGHT = 504
#sets the TL: top left and the BR: bottom right
TL_inside = (int(IM_WIDTH*.4),int(IM_HEIGHT*.70))
BR_inside = (int(IM_WIDTH*.6),int(IM_HEIGHT*.9))
#path for stright
TL_path = (int(IM_WIDTH*.45),int(IM_HEIGHT*0)) #move the LEFT line of path
BR_path = (int(IM_WIDTH*.47),int(IM_HEIGHT*1)) #move the RIGHT line of path
#TL_inside = (int(IM_WIDTH*0.1),int(IM_HEIGHT*0.35))
#BR_inside = (int(IM_WIDTH*0.45),int(IM_HEIGHT-5))
font = cv2.FONT_HERSHEY_SIMPLEX
T = True
is_strafing = False
# Define VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
# shared state for threading
current_x = None
frame_center = (TL_path[0] + BR_path[0]) // 2

print("motion going")

def car_motion(V_x, V_y, V_z):
    speed_x= V_x / 10.0
    speed_y = V_y / 10.0
    speed_z = V_z / 10.0
    bot.set_car_motion(speed_x, speed_y, speed_z)
    return speed_x, speed_y, speed_z

# def strafe_left():
#     print("Strafing left...")
#     bot.set_motor(35, -35, -35, 35)
#     #bot.set_car_motion(0,-0.07,0)
#     while not stop_strafe_event.is_set():
#         vx, vy, vz = bot.get_motion_data()
#         print("Actual motion:", vx, vy, vz)
#         #m1, m2, m3, m4 = bot.get_motor_endcoder()
# 	#print(f"[Encoders] M1={m1}, M2={m2}, M3={m3}, M4={m4}")
#         sleep(0.1)
#     bot.set_motor(0, 0, 0, 0)
#     #bot.set_car_motion(0,0,0)
#     print("Stopped strafing left.")

def strafe_left():
    global current_x, frame_center
    print("Strafing left (dynamic)…")
    while not stop_strafe_event.is_set():
        if current_x is None:
            # haven’t seen a cone yet
            sleep(0.1)
            continue

        # compute how far right-of-center we are
        error = current_x - frame_center  
        abs_err = abs(error)

        # pick power level
        if abs_err > 100:
            power = 40     # full speed
        elif abs_err > 50:
            power = 35     # medium
        else:
            power = 31     # fine adjust

        # strafe left: (power, -power, -power, power)
        bot.set_motor(power, -power, -power, power)

        # debug info
        vx, vy, vz = bot.get_motion_data()
        print(f"[Strafe L] error={error}  power={power}  motion={vx:.1f},{vy:.1f},{vz:.1f}")

        sleep(0.1)

    # once told to stop:
    bot.set_motor(0, 0, 0, 0)
    print("Stopped strafing left.")

def strafe_right():
    print("Strafing right...")
    bot.set_motor(-35, 35, 35, -35)
    #bot.set_car_motion(0,0.07,0)
    while not stop_strafe_event.is_set():
	#m1, m2, m3, m4 = bot.get_motor_endcoder()
	#print(f"[Encoders] M1={m1}, M2={m2}, M3={m3}, M4={m4}")
        sleep(0.1)
    #bot.set_motor(0, 0, 0, 0)
    bot.set_car_motion(0,0,0)
    print("Stopped strafing right.")

def move_forward():
    print('MMMM Going forward SLOW')
    bot.set_motor(20,20,20,20) #this make it fo slower

    #bot.set_car_motion(0.5, 0, 0)
    while not (stop_event.is_set() or quit_event.is_set()):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            bot.set_motor(0,0,0,0)
            final_dist = sensor.distance * 100
            stop_event.set()
            stop_strafe_event.set()
            break
        sleep(0.1)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    print('Manual stop triggered.')
        #    stop_event.set()
        #    break
    bot.set_car_motion(0, 0, 0)
    print('Movement stopped (obstacle or manual).')
			
def read_sensor():
    global flag
    flag = 1
    print('enter sensor')
    while flag == 1:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("q pressed stop everything")
            stop_event.set()
            stop_strafe_event.set()
            flag = 0
            break
        
        dis = sensor.distance * 100
        print('distance: {:.2f} cm'.format(dis))
        sleep(0.1)
        if dis < 8:
            print("Obstacle detected! Distance: {:.2f} cm".format(dis))
            stop_event.set()  # signal to stop the bot
            flag = 0
            break
        sleep(0.1)
    print('read sensor exiting')

stop_event = threading.Event()
stop_strafe_event = threading.Event() #if enters the PATH send single event
quit_event = threading.Event()
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='896x504')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(.99)#(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

# give the camera a moment to auto-expose and adjust
#warmup_secs = 3
#detect_after = time.time() + warmup_secs
#print(f"Camera buffering… detection will start in {warmup_secs}s")

# after starting the stream…
#for _ in range(10):
#    _ = videostream.read()
#    time.sleep(0.1)   # small pause between reads
#print("Camera warmed up, starting main loop.")

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while T:
    flag = 0
    centered = False
    strafe_thread = None
    cone_detected = False

    # Start timer (for calculating frame rate)
    while not centered: #added this <-------------------------------------------------------
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        frame = videostream.read()

        # Acquire frame and resize to expected shape [1xHxWx3]
        frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        
        #line divides the frame and draws boxes path and touch zone
        cv2.rectangle(frame,TL_path,BR_path,(225,0,255),3)
        cv2.putText(frame," ",(TL_inside[0]+10,TL_inside[1]-10),font,1,(225,0,255),3,cv2.LINE_AA)
        #touch zone the frame cones needs to be to get arm down
        cv2.rectangle(frame,TL_inside,BR_inside,(20,20,255),3)
        cv2.putText(frame,"touch zone",(TL_inside[0]+10,TL_inside[1]-10),font,1,(20,255,255),3,cv2.LINE_AA)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects
        
        cone_detected = False
        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(1): #each frame it will perform these
        #for i in range(len(scores)):#find all the matching objects more than one
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, 
                # need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                #width = int((min(imW,(boxes[i][3] * imW))) - (max(1,(boxes[i][1] * imW))))
                

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                if object_name == "cone": #only do this if cone is found
                    cone_detected = True
                    if ymin < 250: #this is placeholder for the closest cone(perform pick-up )
                        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (255, 10, 0), 2)
                        label = '%s: %d%% % d' % (object_name, int(scores[i]*100),ymin) # Example: 'person: 72%'
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text 
                    else: #placeholder to the rest behind (do nothing)
                        cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2) #this is the squares around the cones
                        label = '%s: %d%% % d' % (object_name, int(scores[i]*100),ymax - ymin) # Example: 'person: 72%'
                        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                        label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text   
        
                #PERFORM PICK-UP, this part will drive up to closest clone and pick-up
                #PERFORM PICK-UP, this part will drive up to closest clone and pick-up
                    print('i see something!')
                # i think it wounld be better to make a caluctue on how far it is from the middle of PATH
                # half the dis and that how long to run the motors then update with
                # updated frame untill its in the middle of the PATH, in each if statement
                # i think it wounld be better to make a caluctue on how far it is from the middle of PATH
                # half the dis and that how long to run the motors then update with
                # updated frame untill its in the middle of the PATH, in each if statement
                #print(ymin,' ',xmin,' ',ymax,' ',xmax)
                    x = int(((xmin+xmax)/2))
                    current_x = x

                    y = int(((ymin+ymax)/2))
                    cv2.circle(frame,(x,y), 5, (75,13,180), -1)

                    left_diff = int(TL_path[0] - x) 
                    print(x,' ',y)
                    if (x < TL_path[0]):
                        print('LLLLLL turning right wheels')
                        left_diff = int(TL_path[0] - x) 
                        print('left diff: ',left_diff)
                        #if strafe_thread is None or not strafe_thread.is_alive():
                            #stop_strafe_event.clear()
                            #strafe_thread = threading.Thread(target=strafe_right)
                            #print('starting strafe_thread')
                            #strafe_thread.start()
                        if not is_strafing:
                            stop_strafe_event.clear()
                            strafe_thread = threading.Thread(target=strafe_right)  # or strafe_left
                            is_strafing = True
                            print("starting strafe_thread")
                            strafe_thread.start()

                        #right wheels trun function
                        #left_motora.lmotor()
                        ##bot.set_motor(-40,40,40,-40)# Y axis positive (left wheels turn)
                        #bot.set_car_motion(0,1,0)
                        ##sleep(0.5)
                        ##bot.set_car_motion(0,0,0)
                    elif (x > BR_path[0]):
                        print('RRR turning left wheels')
                        right_diff = int(x - BR_path[0])
                        print('right diff: ',right_diff)
                        if strafe_thread is None or not strafe_thread.is_alive():
                            stop_strafe_event.clear()
                            strafe_thread = threading.Thread(target=strafe_left)
                            print('starting strafe_thread')
                            strafe_thread.start()
                        #left wheels trun function
                        #right_motora.rmotor()
                        ##bot.set_motor(40,-40,-40,40)# Y axis positive (left wheels turn)
                        #bot.set_car_motion(0,-1,0)
                        ##sleep(0.5)
                        ##bot.set_car_motion(0,0,0)
                #this portion is for having a red square "arm-drop" zone, if you want to use it
                #elif ((x > TL_inside[0]) and (x < BR_inside[0]) and (y > TL_inside[1]) and (y < BR_inside[1])):
                    #print('touch down, drop arm ready')
                    #print(BR_inside,' ',TL_inside)
                #this is to test each motor with a duration, its for debugging 
                    #time1 = input('enter time1:\n')
                    #time2 = input('enter time2:\n')
                    #tt_motora.motor(time1,time2)
                    elif (TL_path[0] <= x <= BR_path[0]):
        #we may have to do a threding and have moving forward and sensor run at the same time
                        stop_strafe_event.set()
                        if strafe_thread is not None:
                            strafe_thread.join()
                            centered = True
        #_________________________________________________________________
                        print('Cone lined up! Moving forward...')
            # Start threads
                        stop_event.clear()  # Make sure event is cleared before starting
                        move_thread = threading.Thread(target=move_forward)
                        sensor_thread = threading.Thread(target=read_sensor)
                        move_thread.start()
                        sensor_thread.start()
                        move_thread.join()
                        sensor_thread.join() #to exit end threding and do the PICK-UP
                        print("robot stopped. Deploying arm...")
                        break
        

            else:
                print('looking......')
                bot.set_car_motion(0,0,0)
                if not cone_detected:
                    stop_strafe_event.set()
                    if strafe_thread is not None and strafe_thread.is_alive():
                        strafe_thread.join()
                centered = False  # Go back to hunting


        print('imshow here')
        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

            # Calculate framerate
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1
        Key = cv2.waitKey(1) & 0xFF
        # Press 'q' to quit
        if Key == ord('q'):
            print("exiting....")
            stop_event.set()
            quit_event.set()
            stop_strafe_event.set()
            T = False
            break #this breaks the while true: exits while loop

# Clean up
print("Shutting down all threads...")

stop_event.set()
stop_strafe_event.set()
#i think i should add a stop_event_ultrasenor just to kill it in-case
#I have for forward and left and right and cam 

try:
    move_thread.join(timeout=1)
except:
    pass

try:
    sensor_thread.join(timeout=1)
except:
    pass

try:
    strafe_thread.join(timeout=1)
except:
    pass
cv2.destroyAllWindows()
videostream.stop()
del bot
