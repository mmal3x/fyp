from keras.models import load_model
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog as tkfd
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np


# creating tkinter window
root = Tk()

# loading saved model
model = load_model('final_iter.h5')

# function to obtain image from file directory. The image is pasted on the canvas
# by taking the dimensions of the image.
def create_canvas():

    global pureImage
    global image
    global width
    global height
    global canvas
    global x

    # prompting user to open image from the file system.
    pureImage = Image.open(tkfd.askopenfilename(title="Select an image", filetypes = [("PNG image", "*.png"), ("JPEG image", "*.jpg")]))
    width, height = pureImage.size
    image = ImageTk.PhotoImage(pureImage)

    canvas = Canvas(root, width = 500, height = 500)
    canvas.pack()
    x = canvas.create_image(0,0, image = image, anchor= NW)


# in this function, the image is normalised and converted to
# grey scale format. Its pixels are reshaped to enable the
# classifier to predict the digit. Once the predictions are
# made, the top three are outputted to the screen.

def classify_digit():

    try:
        pureImage is None
    except:
        messagebox.showinfo("Info", "Please insert an image")
        raise

    global firstPredLabel, secondPredLabel, thirdPredLabel
    global predictions
    global img1
    global digits
    global accuracy
    global classAccLabel

    # getting the image
    img1 = pureImage

    # the image should be above 28 by 28 pixels
    if(height >= 28 and width >= 28):

        # resizing the image to 28x28 pixels
        img1 = img1.resize((28,28))

        #converting the image from rgb to grey-scale using the PIL module
        img1 = img1.convert('L')

        # array of image pixels
        img1 = np.array(img1)

        #reshaping the image to match the input shape expected (1 channel for a grayscale image).
        img1 = img1.reshape(1,28,28,1)

        img1 = img1 / 255.0

        # classifying the image
        predictions = model.predict([img1])[0]

        digits = sorted(range(len(predictions)), key = lambda i: predictions[i])[-3:]
        accuracy = [predictions[i] for i in np.argsort(predictions)[-3:]]
        accuracy = np.array(accuracy)
        percentage = accuracy * 100
        percentage = ['%.3f' % p for p in percentage]

        classAccLabel = Label(root, text = 'Top 3 predictions', font = ("Helvitica", 24, "bold"))
        firstPredLabel = Label(root, text = '1.   ' + str(digits[2]) +' , '+ str(percentage[2]) +'%', font =("Helvitica", 22, "bold"))
        secondPredLabel = Label(root, text = '2.   ' + str(digits[1]) +' , '+ str(percentage[1]) +'%', font =("Helvitica", 18, "bold"))
        thirdPredLabel = Label(root, text = '3.   ' + str(digits[0]) +' , '+ str(percentage[0]) +'%', font =("Helvitica", 14, "bold"))

        classAccLabel.place(x = 1100, y = 100)
        firstPredLabel.place(x = 1100, y = 200)
        secondPredLabel.place(x = 1100, y = 300)
        thirdPredLabel.place(x = 1100, y = 400)

    else:
        messagebox.showerror("Image must be at least 28x28 pixels")


def graph():

    try:
        predictions is None
    except:
        messagebox.showinfo("Info", "Please insert an image and classify it")
        raise

    global canvas2
    global plotToCanvas

    # saving the ten predicted classes into lists. [0] = lowest prediction - [9] = highest prediction
    allDigits = sorted(range(len(predictions)), key = lambda i: predictions[i])[-10:] # x-axis labels
    allPreds = [predictions[i] for i in np.argsort(predictions)[-10:]] # y-axis labels]
    allPreds = np.array(allPreds)
    acc = allPreds * 100

    # graph of the predicted probabilities of each class.
    plt.title('SoftMax Probabilites of all classes')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.bar(allDigits, acc, color = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'fuchsia', 'lavender', 'black', 'brown'])
    plt.savefig('plot.png')
    plot = Image.open('plot.png')
    plot2 = plot.resize((432, 288), Image.ANTIALIAS)
    plotToCanvas = ImageTk.PhotoImage(plot2)
    canvas2 = Canvas(root, width = 450, height = 300)
    canvas2.pack()
    canvas2.place(x = 900, y = 450)
    plot_canvas = canvas2.create_image(0,0, image = plotToCanvas, anchor= NW)



def exitWindow():
    root.destroy()

# function call to clear the image, classification labels and graph from the page
def clear_canvas():


    canvas.destroy()
    canvas2.destroy()
    classAccLabel.destroy()
    firstPredLabel.destroy()
    secondPredLabel.destroy()
    thirdPredLabel.destroy()


# title of gui
root.title("Handwritten Digit Classification")

# size of window
root.geometry("1500x1200")

# creating the title of the web app
title = Label(root, text = "DigiFY App!", font =("Helvitica", 24, "bold"))
# outputting it to the screen, keeping the contents compact
title.pack(pady = 20)


# button configuration. Here we create the five buttons needed for the app to run smoothly.
# one of the functions in the command parameters are called once the user clicks on a button
btn_graph = Button(root, text = "Show Accuracies", padx = 40, pady = 20, command = graph)
btn_classify = Button(root, text = "Classify Digit", padx = 40, pady = 20, command = classify_digit)
btn_clear = Button(root, text = "Clear Digit", padx = 40, pady = 20, command = clear_canvas)
btn_insert = Button(root, text = "Insert Image", command = create_canvas)
btn_quit = Button(root, text = "Quit", command = exitWindow)

# placing the button on the screen.
btn_graph.place(x = 100, y = 650)
btn_classify.place(x = 350, y = 650)
btn_clear.place(x = 600, y = 650)
btn_insert.place(x = 50, y = 10)
btn_quit.place(x = 1300, y = 10)

# checking the size of the plot
plotToCanvas = Image.open('plot.png')
width, height = plotToCanvas.size
print(width, height)
# 640 480

# ensures the program continues
root.mainloop()
