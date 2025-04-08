import time
from Rosmaster_Lib import Rosmaster

bot = Rosmaster()
bot.create_receive_threading()
#enable = 2

def arm_servo(s_angle):
    bot.set_uart_servo_angle(servo_id, s_angle, run_time)
#starting point and reset
    bot.set_uart_servo_torque(enable)
    print("enable = 2")
#each function has (servo id: 1 - 6) (angle 0 - 180) and run_time(speed HIGHER IS SLOWER)
bot.set_uart_servo_angle( 6, 180, run_time = 1500)
time.sleep(1)
bot.set_uart_servo_angle( 1, 85, run_time = 1500)
time.sleep(1)
bot.set_uart_servo_angle( 3, 40, run_time = 1500)
time.sleep(1)
bot.set_uart_servo_angle( 4, 30, run_time = 1500)
time.sleep(2)
#bot.set_uart_servo_angle( 2, 30, run_time = 1500)
time.sleep(1)
bot.set_uart_servo_angle( 2, 15, run_time = 1500)
time.sleep(1)
bot.set_uart_servo_angle( 5, 180, run_time = 1500)
time.sleep(1)
#CONE IS ON///////////////////////////////////////////////////////////
bot.set_uart_servo_angle( 6, 110, run_time = 1500)
time.sleep(3)
bot.set_uart_servo_angle( 2, 90, run_time = 1500)
time.sleep(3)
bot.set_uart_servo_angle( 3, 70, run_time = 1500)
time.sleep(3)
bot.set_uart_servo_angle( 1, 180, run_time = 1500)
time.sleep(3)
bot.set_uart_servo_angle( 4, 10, run_time = 1500)
time.sleep(3)
bot.set_uart_servo_angle( 3, 25, run_time = 1500)
time.sleep(3)
bot.set_uart_servo_angle( 6, 170, run_time = 1500)
time.sleep(3)
#cone is dropped
bot.set_uart_servo_angle( 1, 85, run_time = 1500)
time.sleep(3)
bot.set_uart_servo_angle( 3, 10, run_time = 1500)
time.sleep(3)
bot.set_uart_servo_angle( 4, 30, run_time = 1500)
time.sleep(3)
bot.set_uart_servo_angle( 1, 180, run_time = 1500)
time.sleep(3)
bot.set_uart_servo_angle( 2,50, run_time = 1500)
time.sleep(3)

del bot
     

