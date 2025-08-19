######## Webcam Object Detection Using Tensorflow-trained Classifier #########
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
import ctypes
#from gpiozero import DistanceSensor
from Rosmaster_Lib import Rosmaster
#from picamera2 import Picamera2

from gpiozero import DistanceSensor
from gpiozero.pins.pigpio import PiGPIOFactory

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



try:
    lib_path = ctypes.util.find_library('edgetpu')
    print(f"Loading libedgetpu from: {lib_path}")
    edgetpu = ctypes.CDLL(lib_path)
except Exception as e:
    print(f"Error loading libedgetpu: {e}")
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
    global current_x, frame_center
    print("Strafing right (dynamic)…")
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
        bot.set_motor(-power, power, power, -power)

        # debug info
        vx, vy, vz = bot.get_motion_data()
        print(f"[Strafe R] error={error}  power={power}  motion={vx:.1f},{vy:.1f},{vz:.1f}")

        sleep(0.1)





#def strafe_right():
 #   print("Strafing right...")
 #  bot.set_motor(-35, 35, 35, -35)
    #bot.set_car_motion(0,0.07,0)
    #while not stop_strafe_event.is_set():
	#m1, m2, m3, m4 = bot.get_motor_endcoder()
	#print(f"[Encoders] M1={m1}, M2={m2}, M3={m3}, M4={m4}")
#        sleep(0.1)
    #bot.set_motor(0, 0, 0, 0)
#    bot.set_car_motion(0,0,0)
#    print("Stopped strafing right.")

def move_forward():
    print(' |^| Going Forward SLOW |^|')
    bot.set_motor(20, 20, 20, 20)  # slow forward motion

    while not (stop_event.is_set() or quit_event.is_set()):
        sleep(0.1)

    bot.set_motor(0, 0, 0, 0)
    print('Movement stopped (obstacle or manual).')

    #print('MMMM Going forward SLOW')
    #bot.set_motor(20,20,20,20) #this make it fo slower

    ##bot.set_car_motion(0.5, 0, 0)
    #while not (stop_event.is_set() or quit_event.is_set()):
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #bot.set_motor(0,0,0,0)
            #final_dist = sensor.distance * 100
            #stop_event.set()
            #stop_strafe_event.set()
            #break
        #sleep(0.1)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #print('Manual stop triggered.')
            #stop_event.set()
            #break
    #bot.set_car_motion(0, 0, 0)
    #print('Movement stopped (obstacle or manual).')
			
def read_sensor():
    global flag
    global arm_deployed
    flag = 1
    print('entering sensor')

    while flag == 1 and not quit_event.is_set():
        print("[THREAD] Reading distance...")

        dis = sensor.distance * 100
        print("[THREAD] Distance:", dis)

        print('distance: {:.2f} cm'.format(dis))

        if dis < 8:
            print("Obstacle detected! Distance: {:.2f} cm".format(dis))
            arm_deployed = True
            #if needed ill add a function for robot to wait
            stop_event.set()  # signal to stop the bot
            flag = 0
            break

        sleep(0.1)

    print('read sensor exiting')
    print('arm_deployed = ', arm_deployed)
    #global flag
    #flag = 1
    #print('enter sensor')
    #while flag == 1:
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #print("q pressed stop everything")
            #stop_event.set()
            #stop_strafe_event.set()
            #flag = 0
            #break
        
        #dis = sensor.distance * 100
        #print('distance: {:.2f} cm'.format(dis))
        #sleep(0.1)
        #if dis < 8:
            #print("Obstacle detected! Distance: {:.2f} cm".format(dis))
            #stop_event.set()  # signal to stop the bot
            #flag = 0
            #break
        #sleep(0.1)
    #print('read sensor exiting')

stop_event = threading.Event()
stop_strafe_event = threading.Event() #if enters the PATH send single event
quit_event = threading.Event()

class VideoStream:
    """Camera object that controls video streaming."""

    def __init__(self, resolution=(640, 480), framerate=30):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.stopped = False
        self.frame = None
        self._read_once()

    def _read_once(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame = frame

    def start(self):
        from threading import Thread
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

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
min_conf_threshold = float(.9)#(args.threshold)
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
    print("[MODEL] Loading model...")
    print("[DEBUG] Model path:", PATH_TO_CKPT)
    try:
        print("[DEBUG] Attempting to load Edge TPU delegate...")
        #interpreter = Interpreter(model_path=PATH_TO_CKPT)
        interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1')])
        print("[DEBUG] Interpreter successfully created.")

        print(PATH_TO_CKPT)
    except Exception as e:
        print("[ERROR] Interpreter creation failed:", e)

else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
print("[MODEL] Allocating tensors...")

interpreter.allocate_tensors()
print("[MODEL] Model ready.")

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
#set arm up right
#starting point and reset
#bot.set_uart_servo_torque(0)
#print("enable = 0")
bot.set_uart_servo_angle( 1, 90, run_time = 1500)
time.sleep(1)
bot.set_uart_servo_angle( 2, 170, run_time = 1500)
time.sleep(1)
bot.set_uart_servo_angle( 4, 0, run_time = 1500)
time.sleep(1)
bot.set_uart_servo_angle( 3, 0, run_time = 1500)
time.sleep(1)
bot.set_uart_servo_angle( 6, 170, run_time = 1500)
print("arm reset")
arm_deployed = False #this is to make sure the arm is not deployed at the start

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while T:
    flag = 0
    centered = False
    strafe_thread = None
    cone_detected = False
  #this is to make sure the arm is not deployed at the start

    # Start timer (for calculating frame rate)
    while not centered: #added this <-------------------------------------------------------
        t1 = cv2.getTickCount()

        # Grab frame from video stream
        #print("[FRAME] Reading frame...")
        frame = videostream.read()
        #print("[FRAME] Frame read.")

        # Acquire frame and resize to expected shape [1xHxWx3]
        #print("[FRAME] Copying frame...")
        frame = frame.copy()
        #print("[FRAME] Copy done.")
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        
        #line divides the frame and draws boxes path and touch zone
        cv2.rectangle(frame,TL_path,BR_path,(225,0,255),3)
        cv2.putText(frame," ",(TL_inside[0]+10,TL_inside[1]-10),font,1,(225,0,255),3,cv2.LINE_AA)
        #touch zone the frame cones needs to be to get arm down
        #cv2.rectangle(frame,TL_inside,BR_inside,(20,20,255),3)
        #cv2.putText(frame,"touch zone",(TL_inside[0]+10,TL_inside[1]-10),font,1,(20,255,255),3,cv2.LINE_AA)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        #print("[INFER] Preparing input...")

        interpreter.set_tensor(input_details[0]['index'],input_data)
        #print("[INFER] Running interpreter...")

        interpreter.invoke()
        #print("[INFER] Inference done.")


        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects
        
        cone_detected = False
        closest_center_x = 0
        closest_center_y = 0
        max_mid_y = -1
        closest_box = None

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(3): #each frame it will perform these
        #for i in range(len(scores)):#find all the matching objects more than one
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, 
                # need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                box_height = ymax - ymin#print('xmin:',xmin,' ymin:',ymin,' xmax:',xmax,' ymax:',ymax)                    
                
                # Shift ymin 10% upward
                ymin = int(max(1, ymin - box_height * 0.2))  # move top of box up

                #mid point of the box
                x_mid = int(((xmin+xmax)/2)- ((xmax - xmin) * 0.09))
                current_x = x_mid

                y_mid = int(((ymin+ymax)/2) * 0.9) #this is the y mid of the box, used to find closest cone
                
                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                if object_name == "cone": #only do this if cone is found
                    cone_detected = True
                    #if ymin < 250: #this is placeholder for the closest cone(perform pick-up )
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 1)
                    label = '%s: %d%% % d' % (object_name, int(scores[i]*100),y_mid) # Example: 'person: 72%'
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text 
                    #else: #placeholder to the rest behind (do nothing)
                    #    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2) #this is the squares around the cones
                    #    label = '%s: %d%% % d' % (object_name, int(scores[i]*100),y_mid) # Example: 'person: 72%'
                    #    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2) # Get font size
                    #    label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                    #    cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                    #    cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text   
        
                    print('i see something3!')
                    cv2.circle(frame,(x_mid,y_mid), 5, (75,13,180), -1)

                    if y_mid > max_mid_y:
                        max_mid_y = y_mid
                        closest_center_x = x_mid
                        closest_center_y = y_mid
                        closest_box = (xmin, ymin, xmax, ymax)
                
        if cone_detected:
            print('FINAL CLOSEST: x_mid:', closest_center_x, ' y_mid:', closest_center_y)


    #--------------------------------------------------------------------------------------------------------------- 
            if (closest_center_x < TL_path[0]):
                print('<-- going left <--')
                print(closest_center_x, ' < ', TL_path[0])
                    #    left_diff = int(TL_path[0] - closest_center_x) 
                    #    print('left diff: ',left_diff)
                if strafe_thread is None or not strafe_thread.is_alive():
                    stop_strafe_event.clear()
                    strafe_thread = threading.Thread(target=strafe_right)
                    print('--> starting strafe_thread -->')
                    strafe_thread.start()
                    
                        #if strafe_thread is None or not strafe_thread.is_alive():
                            #stop_strafe_event.clear()
                            #strafe_thread = threading.Thread(target=strafe_right)
                            #print('starting strafe_thread')
                            #strafe_thread.start()

            elif (closest_center_x > BR_path[0]):
                print('--> going right -->')
                print(closest_center_x, ' > ', BR_path[0])

                    #    right_diff = int(x - BR_path[0])
                    #    print('right diff: ',right_diff)
                if strafe_thread is None or not strafe_thread.is_alive():
                    stop_strafe_event.clear()
                    strafe_thread = threading.Thread(target=strafe_left)
                    print('<-- starting strafe_thread <--')
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
                    
            elif (TL_path[0] <= closest_center_x <= BR_path[0]):
                print('Cone lined up! Moving forward...')
                print(closest_center_x, ' is between ', TL_path[0], ' and ', BR_path[0])

            #stop strafing if it was happening
                stop_strafe_event.set()
                if strafe_thread is not None:
                    strafe_thread.join()
                    centered = True
            # Start threads for moving forward and reading sensor
                    stop_event.clear()  # Make sure event is cleared before starting
                    move_thread = threading.Thread(target=move_forward)
                    sensor_thread = threading.Thread(target=read_sensor)
                    move_thread.start()
                    sensor_thread.start()
                    move_thread.join()
                    sensor_thread.join() #to exit end threding and do the PICK-UP
                    # Give extra time to settle/alignment
                    #print("Waiting for final adjustments...")
                    #time.sleep(0.5)  # half a second pause
                    print('arm_deployed outside the if statement = ', arm_deployed)
                    #bot.set_uart_servo_torque(2)
                    #print("enable = 2")
                    #looks like it goes in here when cone is detected no matter what the position is
                    
                    if arm_deployed == True:
                        print('robot stopped shounld be within 7cm. Deploying arm...')
                        print('arm_deployed inside the if statement = ', arm_deployed)
                     #making sure arm is up right
                        bot.set_uart_servo_angle( 6, 170, run_time = 1500)
                        time.sleep(1)
                        bot.set_uart_servo_angle( 1, 90, run_time = 1500)
                        time.sleep(1)
                        bot.set_uart_servo_angle( 2, 90, run_time = 1500)
                        time.sleep(1)
                        bot.set_uart_servo_angle( 4, 45, run_time = 1500)
                        time.sleep(1)
                        bot.set_uart_servo_angle( 2, 45, run_time = 1500)
                        time.sleep(1)
                        bot.set_uart_servo_angle( 6, 100, run_time = 1500)
                        time.sleep(1)
                        bot.set_uart_servo_angle( 2, 100, run_time = 1500)
                        time.sleep(1)
                        bot.set_uart_servo_angle( 1, 0, run_time = 1500)
                        time.sleep(1)
                        bot.set_uart_servo_angle( 6, 170, run_time = 1500)
                        time.sleep(1)
                        
                        print("arm moved")
                        #bot.set_uart_servo_angle( 4, 0, run_time = 1500)
                        #time.sleep(1)
                    else:    
                        arm_deployed = False
                        print('else: arm_deployed = F', arm_deployed)
                    #add ROBOT_ARM function here
                    #wait till robot arm is done
                    #break
        

        else:
                print('looking......')
                bot.set_car_motion(0,0,0)
                if not cone_detected:
                    stop_strafe_event.set()
                    if strafe_thread is not None and strafe_thread.is_alive():
                        strafe_thread.join()
                centered = False  # Go back to hunting


        #print('imshow here')
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
            bot.set_motor(0, 0, 0, 0)  #stop wheels in case
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
    
# Close the video stream and destroy all OpenCV windows
sensor.close()
cv2.destroyAllWindows()
videostream.stop()
del bot
