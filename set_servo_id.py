import time
from Rosmaster_Lib import Rosmaster

bot = Rosmaster()
bot.create_receive_threading()
   
  
enable = 0
print("truning off torque")

def arm_servo(s_angle):
    bot.set_uart_servo_angle(servo_id, s_angle, run_time)

bot.set_uart_servo_torque(enable)
bot.set_uart_servo_angle( 1, 90, run_time = 1500)
time.sleep(1)
bot.set_uart_servo_angle( 2, 60, run_time = 1500)
time.sleep(1)
bot.set_uart_servo_angle( 4, 40, run_time = 1500)
time.sleep(2)
bot.set_uart_servo_angle( 2, 30, run_time = 1500)
time.sleep(1)
bot.set_uart_servo_angle( 3, 60, run_time = 1500)
time.sleep(1)
bot.set_uart_servo_angle( 5, 180, run_time = 1500)
time.sleep(1)
bot.set_uart_servo_angle( 6, 130, run_time = 750)
time.sleep(1)
print("truning off torque")


del bot
 

