
import RPi.GPIO as GPIO
from time import sleep
import sys
import cv2

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

ena,in1,in2 = 2,3,4
enb,in3,in4 = 22,27,17

ena2,in2_1,in2_2 = 24,23,18
enb2,in2_3,in2_4 = 13,5,6

GPIO.setup(ena,GPIO.OUT)
GPIO.setup(in1,GPIO.OUT)
GPIO.setup(in2,GPIO.OUT)

GPIO.setup(enb,GPIO.OUT)
GPIO.setup(in3,GPIO.OUT)
GPIO.setup(in4,GPIO.OUT)
#the other motor driver
GPIO.setup(ena2,GPIO.OUT) 
GPIO.setup(in2_1,GPIO.OUT)
GPIO.setup(in2_2,GPIO.OUT)

GPIO.setup(enb2,GPIO.OUT)
GPIO.setup(in2_3,GPIO.OUT)
GPIO.setup(in2_4,GPIO.OUT)

pwma = GPIO.PWM(ena,100)
pwmb = GPIO.PWM(enb,100)
pwma2 = GPIO.PWM(ena2,100)
pwmb2 = GPIO.PWM(enb2,100)