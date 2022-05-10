from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from KNNsklearn import myKNNsklearn, mySVC

def main():

    #svc = mySVC()
    #svc.learn()
    knn = myKNNsklearn(k=9)
    knn.learn()

    # Insperation taken from: Org https://www.codershubb.com/create-the-simple-paint-app-using-python/ For the painting on canvas
    root = Tk()
    root.title("Paint Application")
    root.geometry("280x380")
    paintSize = 15
    # create canvas
    wn=Canvas(root, width=280, height=280, bg='white')
    imgrecreation = np.zeros((28,28))

    def paint(event):
        # get x1, y1, x2, y2 co-ordinates
        x1, y1 = (event.x-paintSize), (event.y-paintSize)
        x2, y2 = (event.x+paintSize), (event.y+paintSize)
        color = "black"
        # display the mouse movement inside canvas
        wn.create_oval(x1, y1, x2, y2, fill=color, outline=color)
        """
        # Paint on my replica that is 10x smaller (there is no way to save canvas so this is good enough)
        imgrecreation[x1//10][y1//10] = 1
        #make the painted area fatter
        for x,y in [(0,1), (1,0), (1,1), (0,-1), (-1,0), (-1,-1)]:
            if ((y1//10)+y > 0) and ((y1//10)+y < 28):
                if imgrecreation[(x1//10)+x][(y1//10)+y] == 0:
                    imgrecreation[(x1//10)+x][(y1//10)+y] = 1  
        """

    def receiveToPredict():
        print("Button pressed")
        #print("SVC predict: ", svc.predict(imgrecreation.T.ravel().reshape(1,-1)))
        print("knn predict: ", knn.predict(imgrecreation.T.ravel().reshape(1,-1)))

    def clear():
        #Clear both canvas and my own matrix
        wn.delete(ALL)
        imgrecreation.fill(0)
    
    predictLable = Label(root, text="Paint your digit")
    predictBtn = Button(root, text ="Predict", command = receiveToPredict)
    clearBtn = Button(root, text ="Clear", command = clear)
    closeBtn = Button(root, text ="Close", command = root.destroy)
    # bind mouse event with canvas(wn)
    wn.bind('<B1-Motion>', paint)
    wn.pack()
    predictLable.pack()
    predictBtn.pack()
    clearBtn.pack()
    closeBtn.pack()
    root.mainloop()

if __name__ == "__main__":
    main()

