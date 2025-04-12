from gpiozero import DistanceSensor
from time import sleep
import time
from Rosmaster_Lib import Rosmaster
#import str_motora
#import arm_control

bot = Rosmaster()
bot.create_receive_threading()
sensor = DistanceSensor(echo=24,trigger=23)#these are GPIO

#def arm_servo(s_angle):
    #bot.set_uart_servo_angle(servo_id, s_angle, run_time)
#def ultra():
try:
    while True:
        dis = sensor.distance *100
        print('distance: {:.2f} cm'.format(dis))
        sleep(0.3)
        if (dis < 5):
            print("stop calling arm program")

            #bot.set_uart_servo_angle( 6, 170, run_time = 1200)
            #time.sleep(1)
            #bot.set_uart_servo_angle( 1, 85, run_time = 1200)
            #time.sleep(1)
            #bot.set_uart_servo_angle( 3, 40, run_time = 1200)
            #time.sleep(1)
            #bot.set_uart_servo_angle( 4, 30 , run_time = 1200)
            #time.sleep(2)
            #bot.set_uart_servo_angle( 2, 30, run_time = 1500)
            #time.sleep(1)
            #bot.set_uart_servo_angle( 2, 10, run_time = 1200)
            #time.sleep(1)
            #bot.set_uart_servo_angle( 4, 50, run_time = 1200)
            #time.sleep(2)
            #bot.set_uart_servo_angle( 5, 180, run_time = 900)
            #time.sleep(1)
            #CONE IS ON///////////////////////////////////////////////////////////
            #bot.set_uart_servo_angle( 6, 110, run_time = 1200)
            #time.sleep(3)
            #bot.set_uart_servo_angle( 2, 70, run_time = 1200)
            #time.sleep(3)
            #bot.set_uart_servo_angle( 3, 70, run_time = 1200)
            #time.sleep(3)
            #bot.set_uart_servo_angle( 1, 180, run_time = 1200) #making the turn
            #time.sleep(3)
            #bot.set_uart_servo_angle( 4, 10, run_time = 1200)
            #time.sleep(3)
            #bot.set_uart_servo_angle( 3, 25, run_time = 1200)
            #time.sleep(3)
            #bot.set_uart_servo_angle( 6, 170, run_time = 1200)
            #time.sleep(3)
            #cone is dropped
            #bot.set_uart_servo_angle( 1, 85, run_time = 1200)
            #time.sleep(3)
            #bot.set_uart_servo_angle( 3, 40, run_time = 1200)
            #time.sleep(3)
            #bot.set_uart_servo_angle( 4, 30, run_time = 1200)
            #time.sleep(1 )
            print("arm is done")
             #left wheels trun function
            #arm_control.arm()

                
except KeyboardInterrupt:
        
    pass
