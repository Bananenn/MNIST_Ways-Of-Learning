from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
from KNNsklearn import myKNNsklearn, mySVC

def main():

    svc = mySVC()
    svc.learn()
    knn = myKNNsklearn()
    knn.learn()

    # Insperation taken from: Org https://www.codershubb.com/create-the-simple-paint-app-using-python/ For the painting on canvas
    root = Tk()
    root.title("Paint Application")
    root.geometry("280x380")
    paintSize = 10
    # create canvas
    wn=Canvas(root, width=280, height=280, bg='red')
    wn.create_rectangle(40, 40, 240, 240, fill='white', outline='white')
    imgrecreation = np.zeros((280,280))
    predimg28 = np.zeros((28,28))

    def paint(event):
        # get x1, y1, x2, y2 co-ordinates
        x1, y1 = (event.x-paintSize), (event.y-paintSize)
        x2, y2 = (event.x+paintSize), (event.y+paintSize)
        # display the mouse movement inside canvas
        wn.create_oval(x1, y1, x2, y2, fill='black', outline='black')
        #Create replica of the wn canvas since it cannot be saved to np array
        imgrecreation[y1:y2,x1:x2] = 255
        
    
    def resize():
        # Ressize to (28,28)
        toPredict = []
        for i,x in enumerate(range(0,280,10)):
            for j,y in enumerate(range(0,280,10)):
                #Check how much of the square is filled
                x1 = x + 10
                y1 = y +10
                square = imgrecreation[x:x1,y:y1]
                toPredict.append(square.sum()/100)
                #predimg28[i][j] = square.sum()/100
        return toPredict
        

    def receiveToPredict():
        #print("Button pressed")
        pred = np.array(resize()).reshape(1,-1)
        print("knn predict: ", knn.predict(pred))
        print("SVC predict: ", svc.predict(pred))

    def clear():
        #Clear both canvas and my own matrix
        wn.delete(ALL)
        wn.create_rectangle(40, 40, 240, 240, fill='white', outline='white')
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

