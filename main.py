import tkinter as tk
from _thread import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import askquestion, showerror, showinfo
from tkinter.ttk import Progressbar
from PIL import Image, ImageTk, ImageOps
import numpy as np

from utils import trainModel, loadModel, saveModel, showFeatures, featuresFile


defaultWidth = 200
defaultHeight = 200

minImageSize = 0
maxImageSize = 5000

selectionSize = 100


class TextureWindow(tk.Toplevel):
    def __init__(self, root):
        super().__init__(root)
        self.geometry("400x300")
        self.frame = tk.Frame(self)
        self.text = tk.StringVar()
        self.label = tk.Label(self.frame, textvariable=self.text,)
        self.label.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)
        self.frame.pack()


class TrainWindow(tk.Toplevel):
    def __init__(self, root):
        super().__init__(root)
        self.geometry("400x300")
        self.frame = tk.Frame(self)
        self.text = tk.StringVar()
        self.label = tk.Label(self.frame, textvariable=self.text,)
        self.label.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)
        self.progressBar = Progressbar(
            self.frame, orient=tk.HORIZONTAL, length=100, mode="determinate")
        self.progressBar.pack(side=tk.TOP, fill=tk.X, expand=tk.YES)
        self.frame.pack()


class SelectionWindow(tk.Toplevel):
    def __init__(self, root, image, classifier):
        super().__init__(root)
        w, h = image.size
        self.geometry(f"{w}x{h}")
        self.rawImage = image
        self.displayedImage = self.rawImage
        self.grayCount = 256
        self.res = 128
        self.quantization = False
        self.equalization = False
        self.classifier = classifier

        self.selectionMenu = tk.Menu(self)
        self.config(menu=self.selectionMenu)

        self.quantizationOptions = tk.Menu(self.selectionMenu)
        self.selectionMenu.add_cascade(
            label="Quantization", menu=self.quantizationOptions)
        self.quantizationOptions.add_command(
            label="2", command=lambda: self.changeQuantization(2))
        self.quantizationOptions.add_command(
            label="4", command=lambda: self.changeQuantization(4))
        self.quantizationOptions.add_command(
            label="8", command=lambda: self.changeQuantization(8))
        self.quantizationOptions.add_command(
            label="16", command=lambda: self.changeQuantization(16))
        self.quantizationOptions.add_command(
            label="32", command=lambda: self.changeQuantization(32))
        self.quantizationOptions.add_command(
            label="64", command=lambda: self.changeQuantization(64))
        self.quantizationOptions.add_command(
            label="128", command=lambda: self.changeQuantization(128))
        self.quantizationOptions.add_command(
            label="256", command=lambda: self.changeQuantization(256))

        self.resolutionOptions = tk.Menu(self.selectionMenu)
        self.selectionMenu.add_cascade(
            label="Resolution", menu=self.resolutionOptions)
        self.resolutionOptions.add_command(
            label="2x2", command=lambda: self.changeRes(2))
        self.resolutionOptions.add_command(
            label="4x4", command=lambda: self.changeRes(4))
        self.resolutionOptions.add_command(
            label="8x8", command=lambda: self.changeRes(8))
        self.resolutionOptions.add_command(
            label="16x16", command=lambda: self.changeRes(16))
        self.resolutionOptions.add_command(
            label="32x32", command=lambda: self.changeRes(32))
        self.resolutionOptions.add_command(
            label="64x64", command=lambda: self.changeRes(64))
        self.resolutionOptions.add_command(
            label="128x128", command=lambda: self.changeRes(128))
        self.resolutionOptions.add_command(
            label="256x256", command=lambda: self.changeRes(256))

        self.equalizationOptions = tk.Menu(self.selectionMenu)
        self.selectionMenu.add_cascade(
            label="Equalization", menu=self.equalizationOptions)
        self.equalizationOptions.add_command(
            label="Equalize", command=self.equalizeImage)

        # TEXTURE MENU
        self.textureOptions = tk.Menu(self.selectionMenu)
        self.selectionMenu.add_cascade(
            label="Texture", menu=self.textureOptions)
        self.textureOptions.add_command(
            label="Compute features", command=self.features)

        self.canvas = tk.Canvas(self, width=w, height=h)
        self.tkImage = ImageTk.PhotoImage(self.displayedImage)
        self.imageCanvas = self.canvas.create_image(
            0, 0, image=self.tkImage, anchor="nw")
        self.canvas.pack()

    def changeRes(self, res):
        self.res = res
        self.reloadScreen()

    def changeQuantization(self, levels):
        self.grayCount = levels
        self.reloadScreen()

    def equalizeImage(self):
        self.equalization = not self.equalization
        self.reloadScreen()

    def reloadScreen(self):
        self.displayedImage = self.rawImage.resize((self.res, self.res))
        self.displayedImage = self.displayedImage.quantize(
            colors=self.grayCount)

        if self.equalization:
            self.displayedImage = ImageOps.equalize(self.displayedImage)
            self.selectionMenu.entryconfig("Texture", state="disabled")
        else:
            self.selectionMenu.entryconfig("Texture", state="normal")

        self.tkImage = ImageTk.PhotoImage(self.displayedImage)
        self.imageCanvas = self.canvas.create_image(
            0, 0, image=self.tkImage, anchor="nw")

        w, h = self.displayedImage.size

        self.geometry(f"{w}x{h}")
        self.canvas.config(width=w, height=h)

    def setImage(self, image):
        self.rawImage = image
        self.reloadScreen()

    def getFeatures(self, screen):
        showFeatures(self.displayedImage, screen)

    def features(self):
        textureWindow = TextureWindow(self)
        start_new_thread(self.getFeatures, (textureWindow,))


class RootWindow:
    def __init__(self):
        self.screen = tk.Tk()
        self.screen.geometry(f"{defaultWidth}x{defaultHeight}")
        self.screen.minsize(defaultWidth, defaultHeight)
        self.screen.eval('tk::PlaceWindow . center')
        self.screen.title("BIRADS - PI")

        self.menubar = tk.Menu(self.screen)
        self.screen.config(menu=self.menubar)

        # IMAGE OPTIONS MENU
        self.imageOptions = tk.Menu(self.menubar)
        self.menubar.add_cascade(
            label="Image options", menu=self.imageOptions)
        self.imageOptions.add_command(
            label="Open Image", command=self.loadImage)
        self.imageOptions.add_command(label="Zoom", command=self.zoom)
        self.imageOptions.add_command(
            label="Reset Zoom", command=self.resetZoom)
        self.imageOptions.add_command(
            label="Select area", command=self.setSelection)

        # TEXTURE MENU
        self.textureOptions = tk.Menu(self.menubar)
        self.menubar.add_cascade(
            label="Texture", menu=self.textureOptions)
        self.textureOptions.add_command(
            label="Compute features", command=self.features)

        # CLASSIFIER MENU

        self.classifierOptions = tk.Menu(self.menubar)
        self.menubar.add_cascade(
            label="Classifier", menu=self.classifierOptions)
        self.classifierOptions.add_command(
            label="Train model", command=self.train)
        self.classifierOptions.add_command(
            label="Save training", command=self.saveTraining)
        self.classifierOptions.add_command(
            label="Load training", command=self.loadTraining)
        self.classifierOptions.add_command(
            label="Classify", command=self.classify)

        # PARAMS
        self.path = ""
        self.rawImage = None
        self.displayedImage = None
        self.tkImage = None
        self.imageCanvas = None

        self.imageScale = 1

        self.selection = False
        self.selectionArea = None
        self.selectionWindow = None

        self.classifier = None

        self.uploadFrame = tk.Frame(self.screen)
        self.uploadFrame.pack(side=tk.BOTTOM, pady=50)

        self.uploadIcon = Image.open("upload.png")
        self.uploadIcon = self.uploadIcon.resize(size=(50, 50))
        self.uploadIconTK = ImageTk.PhotoImage(self.uploadIcon)
        self.uploadButton = tk.Button(
            self.uploadFrame, image=self.uploadIconTK, command=self.loadImage)
        self.uploadButton.pack(side=tk.LEFT)

        self.canvas = tk.Canvas(
            self.screen, width=defaultWidth, height=defaultHeight)
        self.canvas.pack()

        self.canvas.bind('<Button-1>', self.onClick)

        self.screen.mainloop()

    def loadImage(self):
        self.path = askopenfilename(
            filetypes=[("Image FIles", "*.png *.tif *.jpg *.jpeg")])

        # No image found
        if self.path == '':
            return

        # Open image in greyscale
        self.rawImage = (Image.open(self.path)).convert("L")
        self.displayedImage = self.rawImage
        self.tkImage = ImageTk.PhotoImage(self.displayedImage)
        self.imageCanvas = self.canvas.create_image(
            0, 0, image=self.tkImage, anchor='nw')
        w, h = self.displayedImage.size

        self.screen.geometry(f"{w}x{h}")
        self.canvas.config(width=w, height=h)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.uploadFrame.destroy()

    def zoom(self):
        sliderScreen = tk.Tk()
        sliderScreen.title("Zoom")
        slider = tk.Scale(sliderScreen, from_=1, to=5, command=self.setScale,
                          orient=tk.HORIZONTAL, length=200, width=10, sliderlength=15)
        slider.pack()

    def setScale(self, value):
        self.imageScale = int(value)
        # print(self.imageScale)
        self.reloadScreen()

    def resetZoom(self):
        self.imageScale = 1
        self.reloadScreen()

    def reloadScreen(self):
        _w, _h = self.getDisplayDimensions()
        # Display scales under or above size delimiters
        if _w > maxImageSize or _h > maxImageSize or _w <= minImageSize or _h <= minImageSize:
            return

        self.displayedImage = self.rawImage.resize((_w, _h))
        self.tkImage = ImageTk.PhotoImage(self.displayedImage)
        self.canvas.create_image(0, 0, image=self.tkImage, anchor="nw")

        w, h = self.displayedImage.size

        self.screen.geometry(f"{w}x{h}")
        self.canvas.config(width=w, height=h)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def getDisplayDimensions(self):
        w, h = self.rawImage.size

        return int(w * self.imageScale), int(h * self.imageScale)

    def setSelection(self):
        self.selection = not self.selection

    def onClick(self, event):
        x, y = event.x, event.y
        self.canvas.scan_mark(x, y)

        if self.selection:
            if self.selectionArea is not None:
                self.canvas.delete(self.selectionArea)

            self.drawSelection(x, y)
            cropDelimiters = self.cropAreaDelimiters(x, y)
            selectionImage = self.displayedImage.crop(cropDelimiters)

            self.openSelectionWindow(selectionImage)

    def openSelectionWindow(self, image):
        if self.selectionWindow is None or not self.selectionWindow.winfo_exists():
            self.selectionWindow = SelectionWindow(
                self.screen, image, self.classifier)
        else:
            self.selectionWindow.setImage(image)
            self.selectionWindow.classifier = self.classifier

    def cropAreaDelimiters(self, x, y):
        xMin = x - selectionSize
        yMin = y - selectionSize
        xMax = x + selectionSize
        yMax = y + selectionSize
        return xMin, yMin, xMax, yMax

    def selectionDelimiters(self, x, y):
        xCenter, yCenter = self.canvas.canvasx(
            x), self.canvas.canvasy(y)
        xMin, yMin = xCenter - selectionSize, yCenter - selectionSize
        xMax, yMax = xCenter + selectionSize, yCenter + selectionSize
        return xMax, xMin, yMax, yMin

    def drawSelection(self, x, y):
        xMax, xMin, yMax, yMin = self.selectionDelimiters(
            x, y)
        self.selectionArea = self.canvas.create_rectangle(
            xMin, yMin, xMax, yMax, dash=(4, 1), outline="red")

    def getClassifier(self, screen):
        self.classifier = trainModel(screen)

    def train(self):
        trainWindow = TrainWindow(self.screen)
        start_new_thread(self.getClassifier, (trainWindow,))

    def saveTraining(self):
        if self.classifier is not None:
            save = askquestion(
                "BIRADS - PI", message="Do you want to save the model?") == "yes"
            if save:
                saveModel(self.classifier)
        else:
            showerror(
                "Error", "Unable to save - No trained classifier available")

    def loadTraining(self):
        load = askquestion(
            "BIRADS - PI", message="Do you want to load a model?") == "yes"
        if load:
            self.classifier = loadModel()

    def classify(self):
        if self.classifier is None:
            showerror("Error", "No trained classifier")
            return
        if self.rawImage is None:
            showerror("Error", "No image found")

        features = featuresFile(image=self.rawImage)
        features = np.reshape(features, (1, -1))
        prediction = self.classifier.predict(features)
        showinfo("Classification", f"BIRADS {prediction[0]}")

    def getFeatures(self, screen):
        showFeatures(self.rawImage, screen)

    def features(self):
        textureWindow = TextureWindow(self.screen)
        start_new_thread(self.getFeatures, (textureWindow,))


def main():
    RootWindow()


if __name__ == '__main__':
    main()
