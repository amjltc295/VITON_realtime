'''
from time import sleep
try:
    import serial
    #from pyfirmata import Arduino, util
except:
    import pip
    pip.main(['install', 'serial'])
    import serial
from msvcrt import getch
'''
import serial
##==============================================================================
ser = None
##==============================================================================

##======Write Serial Command to arduino============
def SerialWrite(command):
    ser.write(command)
    ser.flushInput()
##====================================

##======Use this============
def digCtrl(command):
    if command == 1:
        cmd = 'a'.encode('utf-8')
        SerialWrite(cmd)
    elif command == 2:
        cmd = 'b'.encode('utf-8')
        SerialWrite(cmd)

    cmd = 'c'.encode('utf-8')
    SerialWrite(cmd)
##====================================

##=======  Main  ================
def arduinoInit():
    ser = serial.Serial("/dev/ttyACM0", 38400, timeout=2) # Establish the connection on a specific port
    print("Connecting to Arduino.....")
    for i in range (1,10):
        rv=ser.readline()
        print("Loading...")
        #Debug print (rv) # Read the newest output from the Arduino
        print (rv.decode("utf-8"))
        ser.flushInput()
        #sleep(1) # Delay for one tenth of a secon
        Str=rv.decode("utf-8")
        print(Str[0:5])
        if Str[0:5]=="Ready":
            print("Get Arduino Ready !")
            break
    print("==================================")
##------------------------------------------------------

'''
##counter = 65  # "A"
##ser.write(chr(counter).encode('utf-8')) # Convert the decimal number to ASCII then send it to the Arduino
cmd="Key in the Command".encode("utf-8")
SerialWrite(cmd)

DigCtrl(1)
print(1)

#sleep(10)

DigCtrl(2)
print(2)

#sleep(15)

ser.close()
'''
