from pyfirmata import Arduino,SERVO
port='COM5'
pin=10

board=Arduino(port)

board.digital[pin].mode=SERVO

def rotateServo(pin,angle):
    board.digital[pin].write(angle)

def door1(val):
    pass