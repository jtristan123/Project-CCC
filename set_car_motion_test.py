import time
from Rosmaster_Lib import Rosmaster

bot = Rosmaster()
bot.create_receive_threading()
time.sleep(0.1)  # Let receive thread start


m1, m2, m3, m4 = bot.get_motor_encoder()
print("encoders:", bot.get_motor_encoder())
  
bot.reset_car_state()
car_type = bot.get_car_type_from_machine()
print("Detected Car Type:", car_type)

while True:
    
    print("start")
    print("Starting slowest strafing (0.02)...")
    print("Before slowest Strafe")
    print("encoders:", bot.get_motor_encoder())
    bot.set_car_motion(0, -0.02, 0)
    time.sleep(10)
    bot.set_car_motion(0, 0, 0)
    print("After slowest Strafe")
    print("encoders:", bot.get_motor_encoder())
    time.sleep(2)

    print("Starting slow strafing (0.05)...")
    print("Before Slow Strafe")
    print("encoders:", bot.get_motor_encoder())
    bot.set_car_motion(0, -0.05, 0)
    time.sleep(10)
    bot.set_car_motion(0, 0, 0)
    print("After Slow Strafe")
    print("encoders:", bot.get_motor_encoder())
    time.sleep(2)

    print("Starting fast strafing (0.7)...")
    print("Before Fast Strafe")
    print("encoders:", bot.get_motor_encoder())
    bot.set_car_motion(0, -0.7, 0)
    time.sleep(10)
    bot.set_car_motion(0, 0, 0)
    print("After Fast Strafe")
    print("encoders:", bot.get_motor_encoder())
    time.sleep(2)

    print("Starting 100% strafing (1)...")
    print("Before 100% Strafe")
    print("encoders:", bot.get_motor_encoder())
    bot.set_car_motion(0, -1, 0)
    time.sleep(10)
    bot.set_car_motion(0, 0, 0)
    print("After 100% Strafe")
    print("encoders:", bot.get_motor_encoder())
    time.sleep(2)
del bot