import cv2 as cv

class Capture():

    def GetImage(): 
        cam = cv.VideoCapture(1)
        while (True):
            ret, frame = cam.read()
            cv.imshow('frame', frame)
            if cv.waitKey(1) & 0xFF == ord('s'):
                cv.imwrite("Capture/plat2.jpg", frame)
                break
        cam.release()
        cv.destroyAllWindows()