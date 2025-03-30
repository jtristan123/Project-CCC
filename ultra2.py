import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)


# Define GPIO pins
GPIO_TRIGGER = 15
GPIO_ECHO = 18
print("distance measurement in progress")
# Set up the GPIO pins
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)

try:
    while True:
        
        GPIO.output(GPIO_TRIGGER, False)
        print("waiting for sensor to settle")
        time.sleep(0.5)

        GPIO.output(GPIO_TRIGGER, True)
        time.sleep(0.00001)
        GPIO.output(GPIO_TRIGGER, False)
        pulse_start = time.time()
        
        while GPIO.input(GPIO_ECHO)==0:
            pulse_start = time.time()
            
        while GPIO.input(GPIO_ECHO)==1:
            pulse_end = time.time()
            
        pulse_duration = pulse_end - pulse_start
        
        distancet = pulse_duration * 34300
        
        distance = distancet / 2
        print("waiting")
        print(distance)

except KeyboardInterrupt:
    print("cleaning upp!")
    GPIO.cleanup()
        