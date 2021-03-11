from pip._vendor.distlib.compat import raw_input

from sys import argv

# https://stackoverflow.com/questions/43650099/a-switch-case-program-in-python

def main():
    print("\nRun individual CNN model, Genetic CNN model or GUI: \n")

    # choice = raw_input("1.Press 1 for CNN model\n2.Press 2 for Genetic CNN model\n3.Press 3 for GUI \n")
    #
    # if (choice == '1'):
    #     mod.trainModel()
    #
    # elif(choice == '2'):
    #     ga
    #
    # elif(choice == '3'):
    #     gui()
    #
    # else:
    #     print("Invalid Selection")

    def runCNNModel():
        import ga

    def runGCNNModel():
        ga()

    def runGUI():
        gui()

    {
        1: runCNNModel(),
        2: runGCNNModel(),
        3: runGUI()
    }[argv[1]](argv[2])

if __name__ == "__main__":
    main()