import cv2
import numpy as np
import dlib
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
import sys
from PyQt5.QtWidgets import QFileDialog
import windowShirts

#================================================= Metoda pentru a asigna event de click la label
def clickable(widget):
    class Filter(QObject):
        clicked = pyqtSignal()

        def eventFilter(self, obj, event):

            if obj == widget:
                if event.type() == QEvent.MouseButtonRelease:
                    if obj.rect().contains(event.pos()):
                        self.clicked.emit()
                        return True

            return False

    filter = Filter(widget)
    widget.installEventFilter(filter)
    return filter.clicked


#================================================= UI

class Ui_Form(object):
    R, G, B = 0, 0, 0
    changePhoto = False
    quit = False
    changeProgram = False
    uploadedPhoto = False
    pathToUploadedFile = ''

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(392, 590)
        self.buttonQuit = QtWidgets.QPushButton(Form)
        self.buttonQuit.setGeometry(QtCore.QRect(0, 550, 201, 41))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.buttonQuit.setFont(font)
        self.buttonQuit.setObjectName("buttonQuit")
        self.buttonTakePicture = QtWidgets.QPushButton(Form)
        self.buttonTakePicture.setGeometry(QtCore.QRect(110, 0, 91, 31))
        self.buttonTakePicture.setFont(font)
        self.buttonTakePicture.setObjectName("buttonTakePicture")
        self.buttonShirts = QtWidgets.QPushButton(Form)
        self.buttonShirts.setGeometry(QtCore.QRect(200, 550, 191, 41))
        self.buttonShirts.setFont(font)
        self.buttonShirts.setObjectName("buttonShirts")
        self.buttonUpload = QtWidgets.QPushButton(Form)
        self.buttonUpload.setGeometry(QRect(200, 0, 91, 31))
        self.buttonUpload.setFont(font)
        self.buttonUpload.setObjectName("buttonUpload")

        self.lipstickRed = QtWidgets.QLabel(Form)
        self.lipstickRed.setGeometry(QtCore.QRect(20, 220, 81, 131))
        self.lipstickRed.setText("")
        self.lipstickRed.setPixmap(QtGui.QPixmap("resources/red.png"))
        self.lipstickRed.setScaledContents(True)
        self.lipstickRed.setObjectName("lipstickRed")
        self.lipstickYellow = QtWidgets.QLabel(Form)
        self.lipstickYellow.setGeometry(QtCore.QRect(20, 380, 81, 131))
        self.lipstickYellow.setText("")
        self.lipstickYellow.setPixmap(QtGui.QPixmap("resources/yellow.png"))
        self.lipstickYellow.setScaledContents(True)
        self.lipstickYellow.setObjectName("lipstickYellow")
        self.lipstickPink = QtWidgets.QLabel(Form)
        self.lipstickPink.setGeometry(QtCore.QRect(320, 380, 81, 131))
        self.lipstickPink.setText("")
        self.lipstickPink.setPixmap(QtGui.QPixmap("resources/pink.png"))
        self.lipstickPink.setScaledContents(True)
        self.lipstickPink.setObjectName("lipstickPink")
        self.lipstickDarkPurpule = QtWidgets.QLabel(Form)
        self.lipstickDarkPurpule.setGeometry(QtCore.QRect(20, 50, 81, 131))
        self.lipstickDarkPurpule.setText("")
        self.lipstickDarkPurpule.setPixmap(QtGui.QPixmap("resources/darkpurple.png"))
        self.lipstickDarkPurpule.setScaledContents(True)
        self.lipstickDarkPurpule.setObjectName("lipstickDarkPurpule")
        self.sliderR = QtWidgets.QSlider(Form)
        self.sliderR.setGeometry(QtCore.QRect(110, 40, 41, 471))
        self.sliderR.setOrientation(QtCore.Qt.Vertical)
        self.sliderR.setObjectName("sliderR")
        self.sliderG = QtWidgets.QSlider(Form)
        self.sliderG.setGeometry(QtCore.QRect(180, 40, 41, 471))
        self.sliderG.setOrientation(QtCore.Qt.Vertical)
        self.sliderG.setObjectName("sliderG")
        self.sliderB = QtWidgets.QSlider(Form)
        self.sliderB.setGeometry(QtCore.QRect(250, 40, 41, 471))
        self.sliderB.setOrientation(QtCore.Qt.Vertical)
        self.sliderB.setObjectName("sliderB")
        self.lipstickOrange = QtWidgets.QLabel(Form)
        self.lipstickOrange.setGeometry(QtCore.QRect(320, 220, 81, 131))
        self.lipstickOrange.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.lipstickOrange.setText("")
        self.lipstickOrange.setPixmap(QtGui.QPixmap("resources/orange.png"))
        self.lipstickOrange.setScaledContents(True)
        self.lipstickOrange.setObjectName("lipstickOrange")
        self.lipstickGreen = QtWidgets.QLabel(Form)
        self.lipstickGreen.setGeometry(QtCore.QRect(320, 50, 81, 131))
        self.lipstickGreen.setText("")
        self.lipstickGreen.setPixmap(QtGui.QPixmap("resources/green.png"))
        self.lipstickGreen.setScaledContents(True)
        self.lipstickGreen.setObjectName("lipstickGreen")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(120, 520, 41, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(194, 520, 41, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setGeometry(QtCore.QRect(264, 520, 51, 20))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

        self.buttonTakePicture.clicked.connect(self.takePicture)
        self.buttonQuit.clicked.connect(self.closeApp)
        self.buttonShirts.clicked.connect(self.changeToShirts)
        self.buttonUpload.clicked.connect(self.uploadPicture)

        clickable(self.lipstickDarkPurpule).connect(lambda: self.set_LipstickColor(0, 0, 255))
        clickable(self.lipstickPink).connect(lambda: self.set_LipstickColor(150, 0, 150))  # pink
        clickable(self.lipstickRed).connect(lambda: self.set_LipstickColor(250, 0, 0))
        clickable(self.lipstickYellow).connect(lambda: self.set_LipstickColor(184, 185, 0))  # yellow
        clickable(self.lipstickOrange).connect(lambda: self.set_LipstickColor(125, 98, 0))  # orange
        clickable(self.lipstickGreen).connect(lambda: self.set_LipstickColor(0, 197, 12))  # green

        self.sliderR.setMinimum(0)
        self.sliderG.setMinimum(0)
        self.sliderB.setMinimum(0)
        self.sliderR.setMaximum(255)
        self.sliderG.setMaximum(255)
        self.sliderB.setMaximum(255)

        self.sliderR.valueChanged.connect(self.setRGB)
        self.sliderG.valueChanged.connect(self.setRGB)
        self.sliderB.valueChanged.connect(self.setRGB)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        Form.setWindowIcon(QtGui.QIcon("utcn.png"))
        self.buttonQuit.setText(_translate("Form", "Quit"))
        self.buttonTakePicture.setText(_translate("Form", "Open Webcam"))
        self.buttonShirts.setText(_translate("Form", "Open Shirts"))
        self.buttonUpload.setText(_translate("Form", "Upload Picture"))
        self.label.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:10pt; color:#d80000;\">R</span></p></body></html>"))
        self.label_2.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:10pt; color:#277500;\">G</span></p></body></html>"))
        self.label_3.setText(_translate("Form", "<html><head/><body><p><span style=\" font-size:10pt; color:#55aaff;\">B</span></p></body></html>"))

    def takePicture(self):
        self.changePhoto = True

    def uploadPicture(self):
        fileName, _ = QFileDialog.getOpenFileNames(None, "Open the file", "",
                                                  "All Files (*)")
        self.uploadedPhoto = True
        self.pathToUploadedFile = fileName[0]

    def setRGB(self):
        self.R=self.sliderR.value()
        self.G=self.sliderG.value()
        self.B=self.sliderB.value()

    def set_LipstickColor(self,R,G,B):
        self.R = R #0 0 255 -> Db , 184,185,0 ->Yellow, 250,0,0->Red , 150, 0 ,150 -> Pink
        self.G = G
        self.B = B

    def closeApp(self):
        self.quit = True

    def changeToShirts(self):
        self.changeProgram = True

#=================================================
def getLipsPhoto(img,points,scale=5,masked=False,cropped=True):
    if masked:
        mask=np.zeros_like(img)
        mask=cv2.fillPoly(mask,[points],(255,255,255))
        #cv2.imshow('',mask)
        img = cv2.bitwise_and(img,mask)
    if cropped:
        bbox = cv2.boundingRect(points)
        x,y,w,h = bbox
        imgCrop = img[y:y+h,x:x+w]
        imgCrop = cv2.resize(imgCrop,(0,0),None,scale,scale)
        return imgCrop
    else:
        return mask

#================================================= MAIN

def Work():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("lipDetectorSpeed.dat")
    cap = cv2.VideoCapture(0)

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Form()
    ui.setupUi(MainWindow)
    MainWindow.show()
    isPhotoUploaded = False

    while True:
        if ui.changePhoto:
            success,img = cap.read()
            originalHeight = np.size(img, 0)
            originalWidth = np.size(img, 1)

            img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
            imgOrignal = img.copy()
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(imgGray)

            for face in faces:
                # x1,y1 = face.left(),face.top()
                # x2,y2 = face.right(),face.bottom()
                # imgOrignal = cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                landmarks = predictor(imgGray, face)
                myPoints = []
                for n in range(12):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    myPoints.append([x, y])
                    #cv2.circle(imgGray,(x,y),1,(50,50,255),cv2.FILLED)

                myPoints = np.array(myPoints)
                imgLips = getLipsPhoto(img, myPoints, 3, masked=True, cropped=False)
                imgColorLips = np.zeros_like(imgLips)
                imgColorLips[:] = ui.B, ui.G, ui.R
                imgColorLips = cv2.bitwise_and(imgLips, imgColorLips)
                imgColorLips = cv2.GaussianBlur(imgColorLips, (7, 7), 10)
                imgColorLips = cv2.addWeighted(imgOrignal, 1, imgColorLips, 0.4, 0)  # 1 - 100% prima, 0.4 -> 40% masca

                if originalHeight < 700 & originalWidth < 700:
                    imgColorLips = cv2.resize(imgColorLips, (originalWidth, originalHeight))
                cv2.imshow('Lipstick', imgColorLips)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                ui.uploadedPhoto=False
                ui.changePhoto =False

        if ui.uploadedPhoto:
            img = cv2.imread(ui.pathToUploadedFile)
            cv2.imwrite("pictureToProcessLipsticks.png",img)
            isPhotoUploaded = True
            ui.uploadedPhoto = False
            ui.changePhoto = False

        if isPhotoUploaded:
            img = cv2.imread("pictureToProcessLipsticks.png")
            originalHeight = np.size(img, 0)
            originalWidth = np.size(img, 1)

            img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
            imgOrignal = img.copy()
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector(imgGray)

            for face in faces:
                # x1,y1 = face.left(),face.top()
                # x2,y2 = face.right(),face.bottom()
                # imgOrignal = cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                landmarks = predictor(imgGray, face)
                myPoints = []
                for n in range(11):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    myPoints.append([x, y])
                    # cv2.circle(imgOrignal,(x,y),2,(50,50,255),cv2.FILLED)

                myPoints = np.array(myPoints)
                imgLips = getLipsPhoto(img, myPoints, 3, masked=True, cropped=False)

                imgColorLips = np.zeros_like(imgLips)
                imgColorLips[:] = ui.B, ui.G, ui.R
                imgColorLips = cv2.bitwise_and(imgLips, imgColorLips)
                imgColorLips = cv2.GaussianBlur(imgColorLips, (7, 7), 10)
                imgColorLips = cv2.addWeighted(imgOrignal, 1, imgColorLips, 0.4, 0)  # 1 - 100% prima, 0.4 -> 40% masca

                if originalHeight < 700 & originalWidth < 700:
                    imgColorLips = cv2.resize(imgColorLips,(originalWidth,originalHeight))
                cv2.imshow('Lipstick', imgColorLips)

        if ui.changeProgram:
            cv2.destroyAllWindows()
            MainWindow.hide()
            windowShirts.Work()
            break

        if ui.quit:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == '__main__':
    Work()
