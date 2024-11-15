#!/usr/bin/env python3

import RPi.GPIO as GPIO
import time
import socket  # Add import for socket communication

SERVO_MIN_PULSE = 500
SERVO_MAX_PULSE = 2500

ServoPin = 18

def map(value, inMin, inMax, outMin, outMax):
    return (outMax - outMin) * (value - inMin) / (inMax - inMin) + outMin

def setup():
    global p
    GPIO.setmode(GPIO.BCM)       # Numbers GPIOs by BCM
    GPIO.setup(ServoPin, GPIO.OUT)   # Set ServoPin's mode is output
    GPIO.output(ServoPin, GPIO.LOW)  # Set ServoPin to low
    p = GPIO.PWM(ServoPin, 50)     # set Frequecy to 50Hz
    p.start(0)                     # Duty Cycle = 0
    
def setAngle(angle):      # make the servo rotate to specific angle (0-180 degrees) 
    angle = max(0, min(180, angle))
    pulse_width = map(angle, 0, 180, SERVO_MIN_PULSE, SERVO_MAX_PULSE)
    pwm = map(pulse_width, 0, 20000, 0, 100)
    p.ChangeDutyCycle(pwm)#map the angle to duty cycle and output it

def servo_control():
    HOST = ''   # Listen on all interfaces
    PORT = 65432  # Port to listen on

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print('Waiting for connection from Mac...')
        while True:  # Add a loop to keep the server running
            conn, addr = s.accept()
            with conn:
                print('Connected by', addr)
                while True:
                    data = conn.recv(1024)
                    if not data:
                        break
                    command = data.decode('utf-8')
                    if command.startswith("MOVE_SERVO"):
                        try:
                            times = int(command.split(":")[1])
                            direction = 1  # 1なら上昇、-1なら下降
                            for _ in range(times):
                                if direction == 1:
                                    for angle in range(0, 181, 5):
                                        setAngle(angle)
                                        time.sleep(0.02)
                                else:
                                    for angle in range(180, -1, -5):
                                        setAngle(angle)
                                        time.sleep(0.02)
                                direction *= -1  # 次回は逆の方向に
                            setAngle(90)  # 動作が終わったら真ん中の位置（90度）に戻す
                            print(f"Servo moved {times} times.")
                        except (IndexError, ValueError):
                            print("Invalid command format.")

def destroy():
    p.stop()
    GPIO.cleanup()

if __name__ == '__main__':     #Program start from here
    setup()
    try:
        servo_control()
    except KeyboardInterrupt:  # When 'Ctrl+C' is pressed, the program destroy() will be executed.
        destroy()