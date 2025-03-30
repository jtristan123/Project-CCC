# -*- coding: UTF-8 -*-
import RPi.GPIO as GPIO
import time
import serial

global ser

# Set GPIO port to BCM coding mode
GPIO.setmode(GPIO.BCM)

# Ignore the warning message
GPIO.setwarnings(False)

# Control the servo, index is the ID number of the servo, 
# value is the position of the servo, s_time is the running time of the servo
def Servo_control(index, value, s_time):
    pack1 = 0xff
    pack2 = 0xff
    id = index & 0xff
    len = 0x07
    cmd = 0x03
    addr = 0x2A
    pos1 = (value >> 8) & 0x00ff
    pos2 = value & 0x00ff
    time1 = (s_time >> 8) & 0x00ff
    time2 = s_time & 0x00ff
    checknum = (~(id + len + cmd + addr + pos1 + pos2 + time1 + time2)) & 0xff

    data = [pack1, pack2, id, len, cmd, addr, pos1, pos2, time1, time2, checknum]
    ser.write(bytes(data))

# Set servo ID
def Servo_Set_ID(index):
    if index < 1 or index > 250:
        return None

    pack1 = 0xff
    pack2 = 0xff
    id = 0xfe
    len = 0x04
    cmd = 0x03
    addr = 0x05
    set_id = index & 0xff

    checknum = (~(id + len + cmd + addr + set_id)) & 0xff

    data = [pack1, pack2, id, len, cmd, addr, set_id, checknum]
    ser.write(bytes(data))

# Initialize and set IDs for multiple servos
def initialize_servos(servo_ids):
    for index in servo_ids:
        Servo_Set_ID(index)
        time.sleep(0.01)  # Small delay between setting IDs

# Control multiple servos with specified positions and times
def control_multiple_servos(servo_commands):
    for command in servo_commands:
        index, value, s_time = command
        Servo_control(index, value, s_time)
        time.sleep(0.01)  # Small delay between commands

try:
    ser = serial.Serial("/dev/ttyS0", 115200, timeout=0.001)
    print("serial.isOpen()")

    # List of servo IDs
    servo_ids = [1,2,3,4,5,6]  # Add more IDs as needed

    # Initialize servos
    initialize_servos(servo_ids)
    fixed = 2048
    Servo_control(1,fixed,1500)
    time.sleep(2)
    Servo_control(2,fixed,1500)
    time.sleep(2)
    Servo_control(3,fixed,1500)
    time.sleep(2)
    Servo_control(4,fixed,1500)
    time.sleep(2)
    Servo_control(5,fixed,1500)
    time.sleep(2)
    Servo_control(6,fixed,1500)
    time.sleep(2)
 
    #Servo_control(2,3100,2500)
    #time.sleep(2)


    #while True:
        #Define commands for multiple servos: (servo_id, position, time)
    #    Servo_control(1,fixed,2500)

     #   servo_commands = [
      #      (2, 2048, 1500),
       #     (3, 2048, 1500),
        #    (4, 2048, 1500),
         #   (5, 2048, 1500),
          #  (6, 2048, 1500)

        #]
        #control_multiple_servos(servo_commands)
        #time.sleep(2)

        #servo_commands = [
         #   (2, 2200, 1500),
          #  (3, 2200, 1500),
           # (4, 2200, 1500),
            #(5, 2200, 1500),
            #(6, 2200, 1500)
        #]
        #control_multiple_servos(servo_commands)
        #time.sleep(2)

except KeyboardInterrupt:
    pass

ser.close()
GPIO.cleanup()