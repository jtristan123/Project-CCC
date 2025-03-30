import RPi.GPIO as GPIO
from time import sleep
import sys
import cv2
from motor_setup import pwma,pwmb,pwma2,pwmb2

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

#pwma = GPIO.PWM(ena,100)
#pwmb = GPIO.PWM(enb,100)
#pwma2 = GPIO.PWM(ena2,100)
#pwmb2 = GPIO.PWM(enb2,100)

pwma.start(0)
pwmb.start(0)
pwma2.start(0)
pwmb2.start(0)

def lmotor():
    n = 0
    while n < 1:
            #gooo forward
        pwma.ChangeDutyCycle(100) #move at 50% FL
        pwmb.ChangeDutyCycle(100) #move at 50% FR
        pwma2.ChangeDutyCycle(90) #move at 50% BR
        pwmb2.ChangeDutyCycle(70) #move at 50% BL

        #make go 180 to the left
        print('moving left wheels')
        GPIO.output(in1,GPIO.LOW)#left side wheel front RES
        GPIO.output(in2,GPIO.HIGH)#left side wheel front  RES
        GPIO.output(in3,GPIO.HIGH)#right side wheel front FORW
        GPIO.output(in4,GPIO.LOW)#right side wheel front FORW
        
        GPIO.output(in2_4,GPIO.HIGH)#right side wheel back FORW 
        GPIO.output(in2_3,GPIO.LOW)#right side wheel back FORW
        GPIO.output(in2_2,GPIO.LOW)#left side wheel back RES
        GPIO.output(in2_1,GPIO.HIGH)#left side wheel back RES

        #keep going till the dot goes in mid
        sleep(0.2) #more left a little bit
        

        GPIO.output(in1,GPIO.LOW)
        GPIO.output(in2,GPIO.LOW)
        GPIO.output(in3,GPIO.LOW)
        GPIO.output(in4,GPIO.LOW)

        GPIO.output(in2_1,GPIO.LOW)
        GPIO.output(in2_2,GPIO.LOW)
        GPIO.output(in2_3,GPIO.LOW)
        GPIO.output(in2_4,GPIO.LOW) 
        n += 1
        print('moved towards ----->.')
