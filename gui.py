from keras.models import load_model
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog as tkfd
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np
import webbrowser as wb

# setting the initial values of the key variables to None as they do not have any data on entry
pureImage, testImg = None, None
predictions, digits, accuracy, percentage, allPreds, allDigits, acc = None, None, None, None, None, None, None
classAccLabel = None
firstPredLabel = None
secondPredLabel = None
thirdPredLabel = None

# saving the original None values of the variables above
pureImage2, testImg2 = pureImage, testImg
predictions2, digits2, accuracy2, percentage2, allPreds2, allDigits2, acc2 = predictions, digits, accuracy, percentage, allPreds, allDigits, acc
classAccLabel2 = classAccLabel
firstPredLabel2 = firstPredLabel
secondPredLabel2 = secondPredLabel
thirdPredLabel2 = thirdPredLabel

# (pureImage, testImg, predictions, digits, accuracy,
#  percentage, allPreds, allDigits, acc, classAccLabel, firstPredLabel, secondPredLabel, thirdPredLabel)

# creating tkinter window instance
root = Tk()

# loading saved model from h5 file
best_model = load_model('best_model.h5')

# function to obtain image from file directory. The image is pasted on the canvas
# by taking the dimensions of the image.
def insert_digit():

    global pureImage
    global canvasImage
    global width
    global height
    global canvas

    if pureImage is not None:
        raise Exception(messagebox.showinfo("Info", "Image is already inserted"))

    # prompting user to open image from the file system.
    pureImage = Image.open(tkfd.askopenfilename(title="Select an image", filetypes = [("PNG image", "*.png"), ("JPEG image", "*.jpg")]))
    canvasImage = pureImage  # the canvas image is one which will be shrunk and outputted to the screen.

    try:
        # resizing the image to fit the canvas.
        if canvasImage.size > (500,500):
            canvasImage = canvasImage.resize((400,400))

    except:
        #  if the image is greater than the 2000x2000 the image is rejected
        if pureImage.size > (2000,2000):
            pureImage = pureImage2
            raise Exception(messagebox.showinfo("Info", "Image is too large"))


    canvasImage = ImageTk.PhotoImage(canvasImage)
    canvas = Canvas(root, width = 500, height = 500)
    canvas.pack()
    canvas.create_image(0,0, image = canvasImage, anchor= NW)


# in this function, the image is normalised and converted to
# grey scale format. Its pixels are reshaped to enable the
# classifier to predict the digit. Once the predictions are
# made, the top three are outputted to the screen.

def classify_digit():

    global firstPredLabel, secondPredLabel, thirdPredLabel
    global predictions
    global testImg
    global digits
    global accuracy
    global percentage
    global classAccLabel



    # should the user click on the classify digit button w/o inserting an image, the exception prompting them to insert
    # an image is raised.
    if pureImage is None and testImg is None and predictions is None and digits is None and percentage is None and accuracy is None and classAccLabel is None and firstPredLabel is None and secondPredLabel is None and thirdPredLabel is None:
        raise Exception(messagebox.showinfo("Info", "Please insert an image"))

    # getting the raw image
    testImg = pureImage
    width, height = testImg.size

# the image should be above 28 by 28 pixels
    if(height >= 28 and width >= 28):

        # resizing the image to 28x28 pixels
        testImg = testImg.resize((28,28))

        #converting the image from rgb to grey-scale using the convert api from the api module
        testImg = testImg.convert('L')

        # the image is converted to an array of image pixels for the convolution process to occur.
        testImg = np.array(testImg)

        #reshaping the image pixels to match the input vector expected for a single channeled grayscale image).
        # each image is a vector of 784 features given the 28x28 image size.
        testImg = testImg.reshape(1,28,28,1)

        # the image pixels are normalised in the range 0-255
        # for the purposes of unit length
        testImg = testImg / 255.0

        # classifying the image using the keras predict API
        predictions = best_model.predict([testImg])[0]

        # sorting list of classification accuracy and prediction score to get the top 3 predictions.
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

    global chartImage
    global plot
    global resizedPlot
    global plotToCanvas
    global allDigits
    global allPreds
    global acc

    # exception block which prompts the user to insert and/or classify an image if the clear digit has been clicked
    if predictions is None and allDigits is None and allPreds is None and acc is None: # and plotToCanvas is None and chartImage is None and resizedPlot is None and plot is None
            raise Exception(messagebox.showinfo("Info", "Please insert an image and/or classify it"))

        # predictions is None or allDigits is None or allPreds is None or acc is Non


    # saving the ten predicted classes into lists. [0] = lowest prediction - [9] = highest prediction
    allDigits = sorted(range(len(predictions)), key = lambda i: predictions[i])[-10:] # x-axis labels

   # the top 10 prediciton accuracies are likewise assigned to a list
    allPreds = [predictions[i] for i in np.argsort(predictions)[-10:]] # y-axis labels]
    allPreds = np.array(allPreds)
    acc = allPreds * 100

    # clearing any saved info from previous plots
    plt.clf()
    plt.cla()
    plt.close()

    # graph of the predicted probabilities of each class.
    plt.title('SoftMax Probabilites of all classes')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.bar(allDigits, acc, color = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'fuchsia', 'lavender', 'black', 'brown'])
    plt.xticks(allDigits) # setting the x-scale to display 0-9
    # os.remove("plot.png") # removing the plot before a new one is saved
    plt.savefig('plot.png')
    plot = Image.open('plot.png')
    resizedPlot = plot.resize((450, 300), Image.ANTIALIAS)
    plotToCanvas = ImageTk.PhotoImage(resizedPlot)
    chartImage = Canvas(root, width = 500, height = 350)
    chartImage.pack()
    chartImage.place(x = 900, y = 450)
    chartImage.create_image(0,0, image = plotToCanvas, anchor= NW)


# tkinter function destroy() function kills the
# window instance.
def exitWindow():
    root.destroy()

# function call to clear the image, classification labels and graph from the page
def clear_canvas():

    # tkinter api destroy() removes all of the widgets from the canvas.
    canvas.destroy()
    chartImage.destroy()
    classAccLabel.destroy()
    firstPredLabel.destroy()
    secondPredLabel.destroy()
    thirdPredLabel.destroy()
    resetVar() # function which resets the variables to None

# function which sets all of the crucial variables to None due to there being no data when
# the user has just
def resetVar():
    global pureImage, testImg
    global predictions, digits, accuracy, percentage, allPreds, allDigits, acc
    global classAccLabel, firstPredLabel, secondPredLabel, thirdPredLabel

    # resetting the the variables back to None once the user has clicked clear digit
    pureImage, testImg = pureImage2, testImg2
    predictions, digits, accuracy, percentage, allPreds, allDigits, acc = predictions2, digits2, accuracy2, percentage2, allPreds2, allDigits2, acc2
    classAccLabel = classAccLabel2
    firstPredLabel = firstPredLabel2
    secondPredLabel = secondPredLabel2
    thirdPredLabel = thirdPredLabel2

# function which opens the paint tool once the url is clicked
def callback(url):

    wb.open_new(url)

# useful information
def help():
    info = Label(root, text = "WELCOME!\n\n\n This app classifies digits from 0-9.\n\n"
                              "1. First draw a digit using a paint tool or by hand. \n\n "
                              "NOTE:The background should preferably be \n\n"
                              "black or another dark colour with the digit\n\n"
                              "in white. Use the paint tools to make the \n\n"
                              "necessary changes.\n\n"
                              "2. Save the image as a png or jpeg.\n\n"
                              "3. Insert the image using the button on \n\n"
                              "the top left.\n\n"
                              "4. Hit the classify button to test your \n\n"
                              "digit and ouput the top 3 predictions.\n\n"
                              "5. For further prediction information, \n\n"
                              "click Show Accuracies\n\n", font = ("Helvitica", 14, "bold"), justify=LEFT)

    info.pack()
    info.place(x = 50, y = 100)

# title of gui
root.title("Handwritten Digit Classification")

# size of window
root.geometry("1500x1200")

# the title of the web app
title = Label(root, text = "DigiFY App!", font =("Helvitica", 24, "bold"))
# outputting it to the screen, keeping the contents compact
title.pack(pady = 20)

# link to the paint tool
btn_paint = Label(root, text = "Click here to draw your digit", fg= "blue", cursor = "hand2")
btn_paint.place(x = 200, y = 22)
btn_paint.bind("<Button-1>", lambda e: callback("https://jspaint.app/#local:41f86bef475158"))

# button configuration. Here we create the five buttons needed for the app to run smoothly.
# one of the functions in the command parameters are called once the user clicks on a button
btn_graph = Button(root, text = "Show Accuracies", padx = 40, pady = 20, command = graph)
btn_classify = Button(root, text = "Classify Digit", padx = 40, pady = 20, command = classify_digit)
btn_clear = Button(root, text = "Clear Digit", padx = 40, pady = 20, command = clear_canvas)
btn_insert = Button(root, text = "Insert Image", command = insert_digit)
btn_help = Button(root, text = "Help", command = help)
btn_quit = Button(root, text = "Quit", command = exitWindow)

# placing the buttons on the screen.
btn_graph.place(x = 100, y = 650)
btn_classify.place(x = 350, y = 650)
btn_clear.place(x = 600, y = 650)
btn_insert.place(x = 50, y = 20)
btn_help.place(x = 150, y = 20)
btn_quit.place(x = 1300, y = 20)

# ensures the program continues
root.mainloop()
