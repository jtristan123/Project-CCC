import time
from Rosmaster_Lib import Rosmaster
bot = Rosmaster()

bot.create_receive_threading()
enable = 1


print("motion going")

def car_motion(V_x, V_y, V_z):
    speed_x= V_x / 10.0
    speed_y = V_y / 10.0
    speed_z = V_z / 10.0
    bot.set_car_motion(speed_x, speed_y, speed_z)
    return speed_x, speed_y, speed_z

bot.set_car_motion(0,0,0)

bot.set_uart_servo_torque(enable)
print("enable = 2")
#each function has (servo id: 1 - 6) (angle 0 - 180) and run_time(speed HIGHER IS SLOWER)
bot.set_uart_servo_angle( 6, 90, run_time = 1500)
time.sleep(1)
bot.set_uart_servo_angle( 6, 180, run_time = 1500)
time.sleep(1)




del bot