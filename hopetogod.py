import serial
import time

# Configure the serial port
ser = serial.Serial(
    port='/dev/ttyS0',  # Replace with the correct port if different
    baudrate=115200,      # Baudrate for the bus servos
    timeout=1
)

# Function to calculate the checksum for servo commands
def calculate_checksum(command):
    checksum = 0
    for byte in command:
        checksum ^= byte
    return checksum & 0xFF

# Function to send a command to a servo
def send_servo_command(id, command_type, params):
    command = [0x55, 0x55, id, len(params) + 3, command_type] + params
    checksum = calculate_checksum(command[2:])
    command.append(checksum)
    ser.write(bytearray(command))

# Function to set servo angle
def set_servo_angle(id, angle, time):
    low_byte = angle & 0xFF
    high_byte = (angle >> 8) & 0xFF
    time_low = time & 0xFF
    time_high = (time >> 8) & 0xFF
    params = [low_byte, high_byte, time_low, time_high]
    send_servo_command(id, 0x01, params)

# Initialize all servos to a default position
def initialize_servos():
    for servo_id in range(1, 6):  # Assuming servo IDs are 1 through 6
        set_servo_angle(servo_id, 2048, 1500)  # Set to 512 (neutral position), 1000 ms

# Move each servo to a specific angle
def move_servos(angles, time):
    for servo_id, angle in enumerate(angles, start=1):
        set_servo_angle(servo_id, angle, time)

if __name__ == '__main__':
    try:
        initialize_servos()
        time.sleep(2)  # Wait for the servos to reach the initial position
        
        while True:
            # Example: move servos to new positions
            angles = [2048, 2048, 2048, 2048, 2048, 2048]
            move_servos(angles, 1500)
            time.sleep(2)
            
            angles = [2200, 2200, 2200, 2200, 2200, 2200]
            move_servos(angles, 1500)
            time.sleep(2)
    except KeyboardInterrupt:
        print("Program interrupted")
    finally:
        # Reset servos to neutral position before exiting
        initialize_servos()
        ser.close()
        print("Program terminated and servos reset")
