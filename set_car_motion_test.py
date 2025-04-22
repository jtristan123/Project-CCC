import time
from Rosmaster_Lib import Rosmaster

bot = Rosmaster()
bot.create_receive_threading()
time.sleep(0.1)  # Let receive thread start

#set_pid_param(self, kp, ki, kd, forever=False)
bot.set_pid_param(kp=1.8, ki=0, kd=0.3, forever=False) #tune these later
print("Current PID:", bot.get_motion_pid())
time.sleep(1)

m1, m2, m3, m4 = bot.get_motor_encoder()
print("encoders:", bot.get_motor_encoder())
  
bot.reset_car_state()
car_type = bot.get_car_type_from_machine()
print("Detected Car Type:", car_type)
 
while True:
    print("\nStart test 9")
    print("Starting slowest strafing (0.02)...")

    # Store encoder values BEFORE motion
    m1_start, m2_start, m3_start, m4_start = bot.get_motor_encoder()
    print("Before strafe: ", m1_start, m2_start, m3_start, m4_start)
    bot.set_motor(50, -50, -49, 49)  # strafe left raw PWM

    #bot.set_car_motion(0, 0.35, 0)
    time.sleep(5)

    # Get encoder values AFTER motion
    bot.set_car_motion(0, 0, 0)
    time.sleep(0.2)
    m1_end, m2_end, m3_end, m4_end = bot.get_motor_encoder()
    print("After strafe: ", m1_end, m2_end, m3_end, m4_end)

    # Delta = change in encoder
    print("diff Encoders:")
    print(f"  M1: {m1_end - m1_start}")
    print(f"   M2: {m2_end - m2_start}")
    print(f"   M3: {m3_end - m3_start}")
    print(f"  M4: {m4_end - m4_start}")

    vx, vy, vz = bot.get_motion_data()
    print("Actual motion:", vx, vy, vz)
    print("end")
    time.sleep(10000)

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