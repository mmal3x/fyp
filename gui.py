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
    global canvasImage
    global width
    global height
    global canvas
    global x

    # prompting user to open image from the file system.
    pureImage = Image.open(tkfd.askopenfilename(title="Select an image", filetypes = [("PNG image", "*.png"), ("JPEG image", "*.jpg")]))
    canvasImage = pureImage  # the canvas image

    try:
        # resizing the image to fit the canvas.
        if(canvasImage.size > (500,500)):
            canvasImage = canvasImage.resize((400,400))

    except:
        #  if the image is greter than the 2000x2000 the image is rejected
        if(pureImage.size > (2000,2000)):
            messagebox.showerror("Image is too large")

            pureImage is None

    canvasImage = ImageTk.PhotoImage(canvasImage)
    canvas = Canvas(root, width = 500, height = 500)
    canvas.pack()
    x = canvas.create_image(0,0, image = canvasImage, anchor= NW)


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
    global testImg
    global digits
    global accuracy
    global classAccLabel

    # getting the raw image
    testImg = pureImage
    width, height = testImg.size

# the image should be above 28 by 28 pixels
    if(height >= 28 and width >= 28):

        # resizing the image to 28x28 pixels
        testImg = testImg.resize((28,28))

        #converting the image from rgb to grey-scale using the pillow module
        testImg = testImg.convert('L')

        # the image is converted to an array of image pixels for the convolution process to occur.
        testImg = np.array(testImg)

        #reshaping the image pixels to match the input vector expected for a single channeled grayscale image).
        # each image is a vector of 784 features given the 28x28 image size.
        testImg = testImg.reshape(1,28,28,1)

        # the image pixels are normalised in the range 0-255
        # to
        testImg = testImg / 255.0

        # classifying the image
        predictions = model.predict([testImg])[0]

        digits = sorted(range(len(predictions)), key = lambda i: predictions[i])[-3:]
        accuracy = [predictions[i] for i in np.argsort(predictions)[-3:]]
        accuracy = np.array(accuracy)
        percentage = accuracy * 100
        percentage = ['%.3f' % p for p in percentage]

        classAccLabel = Label(root, text = 'Top 3 predictions', font = ("Helvitica", 24, "bold"))
        firstPredLabel = Label(root, text = '1.   ' + str(digits[2]) +' , '+ str(percentage[2]) +'%', font =("Helvitica", 22, "bold"))
        secondPredLabel = Label(root, text = '2.   ' + str(digits[1]) +' , '+ str(percentage[1]) +'%', font =("Helvitica", 18, "bold"))
        thirdPredLabel = Label(root, text = '3.   ' + str(digits[0]) +' , '+ str(percentage[0]) +'%', font =("Helvitica", 14, "bold"))

        classAccLabel.place(x = 1000, y = 100)
        firstPredLabel.place(x = 1000, y = 200)
        secondPredLabel.place(x = 1000, y = 300)
        thirdPredLabel.place(x = 1000, y = 400)

    else:
        messagebox.showerror("Image must be at least 28x28 pixels")

# graph function takes the prediction values and
# the prediction accuracies as percentages
# and outputs them to the window in a bar plot.
# Each bar has been assigned a different colour
# to help with readability.
def graph():

    try:
        predictions is None
    except:

        messagebox.showinfo("Info", "Please insert an image and/or classify it")
        raise

    global canvas2
    global plotToCanvas

    # saving the ten predicted classes into lists. [0] = lowest prediction - [9] = highest prediction
    allDigits = sorted(range(len(predictions)), key = lambda i: predictions[i])[-10:] # x-axis labels

   # the top 10 prediciton accuracies are likewise assigned to a list
    allPreds = [predictions[i] for i in np.argsort(predictions)[-10:]] # y-axis labels]
    allPreds = np.array(allPreds)
    acc = allPreds * 100

    # graph of the predicted probabilities of each class.
    plt.title('SoftMax Probabilites of all classes')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.bar(allDigits, acc, color = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'fuchsia', 'lavender', 'black', 'brown'])
    plt.xticks(allDigits) # setting the x-scale to display 0-9
    plt.savefig('plot.png')
    plot = Image.open('plot.png')
    plot2 = plot.resize((450, 300), Image.ANTIALIAS)
    plotToCanvas = ImageTk.PhotoImage(plot2)
    canvas2 = Canvas(root, width = 500, height = 350)
    canvas2.pack()
    canvas2.place(x = 900, y = 450)
    plot_canvas = canvas2.create_image(0,0, image = plotToCanvas, anchor= NW)


# tkinter function destroy() function kills the
# window instance.
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
    pureImage, predictions is None


# title of gui
root.title("Handwritten Digit Classification")

# size of window
root.geometry("1500x1200")

# the title of the web app
title = Label(root, text = "DigiFY App!", font =("Helvitica", 24, "bold"))
# outputting it to the screen, keeping the contents compact
title.pack(pady = 20)

# useful information
def help():
    info = Label(root, text = "WELCOME!\n\n\n This app classifies digits from 0-9.\n\n"
                              "1. First draw a digit using a paint tool \n\n "
                              "[https://jspaint.app/#local:41f86bef475158] \n\n"
                              "NOTE:The background preferably should be \n\n"
                              "black or another dark colour\n\n"
                              "2. Save the image as a png or jpeg\n\n"
                              "3. Insert the image using the button on \n\n" 
                              "the top left\n\n"
                              "4. Hit the classify button to test your \n\n"
                              "digit and ouput the top 3 predictions\n\n"
                              "5. For further prediction information, \n\n"
                              "click Show Accuracies\n\n", font = ("Helvitica", 14, "bold"), justify=LEFT)

    info.pack()
    info.place(x = 50, y = 100)

# button configuration. Here we create the five buttons needed for the app to run smoothly.
# one of the functions in the command parameters are called once the user clicks on a button
btn_graph = Button(root, text = "Show Accuracies", padx = 40, pady = 20, command = graph)
btn_classify = Button(root, text = "Classify Digit", padx = 40, pady = 20, command = classify_digit)
btn_clear = Button(root, text = "Clear Digit", padx = 40, pady = 20, command = clear_canvas)
btn_insert = Button(root, text = "Insert Image", command = create_canvas)
btn_help = Button(root, text = "Help", command = help)
btn_quit = Button(root, text = "Quit", command = exitWindow)


# placing the button on the screen.
btn_graph.place(x = 100, y = 650)
btn_classify.place(x = 350, y = 650)
btn_clear.place(x = 600, y = 650)
btn_insert.place(x = 50, y = 20)
btn_help.place(x = 150, y = 20)
btn_quit.place(x = 1300, y = 20)

# checking the size of the plot
plotToCanvas = Image.open('plot.png')
width, height = plotToCanvas.size
print(width, height)
# 640 480

# ensures the program continues
root.mainloop()
