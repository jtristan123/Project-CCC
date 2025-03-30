
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

pwma.start(0)
pwmb.start(0)
pwma2.start(0)
pwmb2.start(0)

time1 = input('enter time1:\n')
time1 = int(time1)
time2 = input('enter time2:\n')
time2 = int(time2)
n = 0

sleep(25)
while n < 2:
        #gooo forward
    print('going lenght')
    pwma.ChangeDutyCycle(100) #move at 50%
    pwmb.ChangeDutyCycle(100) #move at 50%
    pwma2.ChangeDutyCycle(100) #move at 50%
    pwmb2.ChangeDutyCycle(100) #move at 50%
    
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.HIGH)
    
    GPIO.output(in2_1,GPIO.LOW)
    GPIO.output(in2_2,GPIO.HIGH)
    GPIO.output(in2_3,GPIO.HIGH)
    GPIO.output(in2_4,GPIO.LOW)
    
    sleep(time1)

    #make the 90 deg turn
    print('90 deg')
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.HIGH)
    
    GPIO.output(in2_1,GPIO.LOW)
    GPIO.output(in2_2,GPIO.LOW)
    GPIO.output(in2_3,GPIO.LOW)
    GPIO.output(in2_4,GPIO.LOW)
    sleep(2) #how much time to make a 90
    
    print('going width')
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.HIGH)
    
    GPIO.output(in2_1,GPIO.LOW)
    GPIO.output(in2_2,GPIO.HIGH)
    GPIO.output(in2_3,GPIO.HIGH)
    GPIO.output(in2_4,GPIO.LOW) 
    sleep(time2)

    #make the 90 deg turn
    print('90 deg')
    GPIO.output(in1,GPIO.HIGH)
    GPIO.output(in2,GPIO.LOW)
    GPIO.output(in3,GPIO.LOW)
    GPIO.output(in4,GPIO.HIGH)

    GPIO.output(in2_1,GPIO.LOW)
    GPIO.output(in2_2,GPIO.LOW)
    GPIO.output(in2_3,GPIO.LOW)
    GPIO.output(in2_4,GPIO.LOW) 
    sleep(2) #how much time to make a 90
   
    n += 1
    print(n)
    #stop aka go to the top

GPIO.output(in1,GPIO.LOW)
GPIO.output(in2,GPIO.LOW)
GPIO.output(in3,GPIO.LOW)
GPIO.output(in4,GPIO.LOW)

GPIO.output(in2_1,GPIO.LOW)
GPIO.output(in2_2,GPIO.LOW)
GPIO.output(in2_3,GPIO.LOW)
GPIO.output(in2_4,GPIO.LOW) 

print('loop ended.')
print(time1)
print(time2)