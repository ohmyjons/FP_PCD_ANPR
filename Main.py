from FinalANPR import Anpr_Indo
from Camera  import Capture
import cv2 as cv
import mysql.connector


def RecognitionPlat():
    
    img = cv.imread("./test_images/2.jpg")
    print ("recognisi plat berjalan")
    Anpr_Indo(img)
    WaitInput()

def WaitInput():
    while True:
    # print ("asdsa")
        val = input("minta input  =  ")
        print (val)
        # print(val.type)
        if (val == '1'):
            # Capture.GetImage()
            RecognitionPlat()
            
        if (val != '1') :
            print ('input tidak ada')
            WaitInput()
    
def SavePlate():
    # conn db
    db  = mysql.connector.connect(
        host='127.0.0.1',
        user='root',
        passwd='',
        database='data_platnomer'
    )


WaitInput()