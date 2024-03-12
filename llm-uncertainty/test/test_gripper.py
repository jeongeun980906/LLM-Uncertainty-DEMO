from pymodbus.client.sync import ModbusTcpClient
from model.gripper import *

def main(): 
    graspclient = ModbusTcpClient('192.168.0.13') 
    closeGrasp(200,100,graspclient)
    # openGrasp(400,1000,graspclient)
main()