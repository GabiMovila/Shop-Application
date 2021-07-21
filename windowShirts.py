import os
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
import time
import numpy as np
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QFileDialog

import windowLipsticks

# =================================================
# Metoda pentru a asigna event de click la label
def clickable(widget):
    class Filter(QObject):
        clicked = pyqtSignal()

        def eventFilter(self, obj, event):

            if obj == widget:
                if event.type() == QEvent.MouseButtonRelease:
                    if obj.rect().contains(event.pos()):
                        self.clicked.emit()
                        # The developer can opt for .emit(obj) to get the object within the slot.
                        return True

            return False

    filter = Filter(widget)
    widget.installEventFilter(filter)
    return filter.clicked


# =============================================
class Ui_Form(object):
    quit = False
    changePhoto = False
    shirtName = "tshirt.png"
    changeProgram = False
    uploadedPhoto = False
    pathToUploadedFile = ''

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(278, 581)
        self.shirt = QtWidgets.QLabel(Form)
        self.shirt.setGeometry(QtCore.QRect(20, 50, 91, 101))
        self.shirt.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.shirt.setText("")
        self.shirt.setPixmap(QtGui.QPixmap("tshirt.png"))
        self.shirt.setScaledContents(True)
        self.shirt.setObjectName("shirt")
        self.shirt1 = QtWidgets.QLabel(Form)
        self.shirt1.setGeometry(QtCore.QRect(150, 50, 91, 101))
        self.shirt1.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.shirt1.setText("")
        self.shirt1.setPixmap(QtGui.QPixmap("tshirt(1).png"))
        self.shirt1.setScaledContents(True)
        self.shirt1.setObjectName("shirt1")
        self.shirt5 = QtWidgets.QLabel(Form)
        self.shirt5.setGeometry(QtCore.QRect(20, 170, 91, 101))
        self.shirt5.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.shirt5.setText("")
        self.shirt5.setPixmap(QtGui.QPixmap("tshirt(5).png"))
        self.shirt5.setScaledContents(True)
        self.shirt5.setObjectName("shirt5")
        self.shirt4 = QtWidgets.QLabel(Form)
        self.shirt4.setGeometry(QtCore.QRect(150, 170, 91, 101))
        self.shirt4.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.shirt4.setText("")
        self.shirt4.setPixmap(QtGui.QPixmap("tshirt(4).png"))
        self.shirt4.setScaledContents(True)
        self.shirt4.setObjectName("shirt4")
        self.shirt7 = QtWidgets.QLabel(Form)
        self.shirt7.setGeometry(QtCore.QRect(20, 290, 91, 101))
        self.shirt7.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.shirt7.setText("")
        self.shirt7.setPixmap(QtGui.QPixmap("tshirt(7).png"))
        self.shirt7.setScaledContents(True)
        self.shirt7.setObjectName("shirt7")
        self.shirt3 = QtWidgets.QLabel(Form)
        self.shirt3.setGeometry(QtCore.QRect(150, 290, 91, 101))
        self.shirt3.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.shirt3.setText("")
        self.shirt3.setPixmap(QtGui.QPixmap("tshirt(3).png"))
        self.shirt3.setScaledContents(True)
        self.shirt3.setObjectName("shirt3")
        self.shirt6 = QtWidgets.QLabel(Form)
        self.shirt6.setGeometry(QtCore.QRect(20, 410, 91, 101))
        self.shirt6.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.shirt6.setText("")
        self.shirt6.setPixmap(QtGui.QPixmap("tshirt(6).png"))
        self.shirt6.setScaledContents(True)
        self.shirt6.setObjectName("shirt6")
        self.shirt8 = QtWidgets.QLabel(Form)
        self.shirt8.setGeometry(QtCore.QRect(150, 410, 91, 101))
        self.shirt8.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.shirt8.setText("")
        self.shirt8.setPixmap(QtGui.QPixmap("tshirt(8).png"))
        self.shirt8.setScaledContents(True)
        self.shirt8.setObjectName("shirt8")
        self.lipstickButton = QtWidgets.QPushButton(Form)
        self.lipstickButton.setGeometry(QtCore.QRect(20, 540, 93, 28))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lipstickButton.setFont(font)
        self.lipstickButton.setObjectName("lipstickButton")
        self.quitButton = QtWidgets.QPushButton(Form)
        self.quitButton.setGeometry(QtCore.QRect(150, 540, 93, 28))
        self.quitButton.setFont(font)
        self.quitButton.setObjectName("quitButton")
        self.uploadButton = QtWidgets.QPushButton(Form)
        self.uploadButton.setGeometry(QtCore.QRect(50, 10, 93, 28))
        self.uploadButton.setFont(font)
        self.uploadButton.setObjectName("uploadButton")

        self.uploadPhotoButton = QtWidgets.QPushButton(Form)
        self.uploadPhotoButton.setGeometry(QtCore.QRect(140, 10, 93, 28))
        self.uploadPhotoButton.setFont(font)
        self.uploadPhotoButton.setObjectName("uploadPhotoButton")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

        self.uploadButton.clicked.connect(self.takePicture)
        self.quitButton.clicked.connect(self.close)
        self.lipstickButton.clicked.connect(self.changeToLipstick)
        self.uploadPhotoButton.clicked.connect(self.uploadPicture)

        clickable(self.shirt).connect(lambda: self.setShirt("tshirt.png"))
        clickable(self.shirt1).connect(lambda: self.setShirt("tshirt(1).png"))
        clickable(self.shirt3).connect(lambda: self.setShirt("tshirt(3).png"))
        clickable(self.shirt4).connect(lambda: self.setShirt("tshirt(4).png"))
        clickable(self.shirt5).connect(lambda: self.setShirt("tshirt(5).png"))
        clickable(self.shirt6).connect(lambda: self.setShirt("tshirt(6).png"))
        clickable(self.shirt7).connect(lambda: self.setShirt("tshirt(7).png"))
        clickable(self.shirt8).connect(lambda: self.setShirt("tshirt(8).png"))

    def changeToLipstick(self):
        self.changeProgram = True

    def close(self):
        self.quit = True

    def takePicture(self):
        self.changePhoto = True

    def uploadPicture(self):
        fileName, _ = QFileDialog.getOpenFileNames(None, "Open the file", "",
                                                  "All Files (*)")
        self.uploadedPhoto = True
        self.pathToUploadedFile = fileName[0]

    def setShirt(self, shirtName):
        self.shirtName = shirtName

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Shirts"))
        Form.setWindowIcon(QtGui.QIcon("utcn.png"))
        self.lipstickButton.setText(_translate("Form", "Lipsticks"))
        self.uploadPhotoButton.setText(_translate("Form", "Upload photo"))
        self.quitButton.setText(_translate("Form", "Quit"))
        self.uploadButton.setText(_translate("Form", "Take picture"))


# =============================================
def alg(image):

    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18

    frame = image
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    # input image dimensions for the network
    inWidth = 640
    inHeight = 480
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []
    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H
        if prob > threshold:
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else:
            points.append(None)
    return points

# =============================================
def overlay(points, imgBackground, imgShirtName):
    # +- pt ca imaginea cu tricoul e un dreptunghi(ca orice imagine)
    punctDownRight = points[7]
    temp = points[2]
    punctUpLeft = (temp[0]-40, temp[1]-30)
    # cv2.rectangle(frame,punctUpLeft,punctDownRight,color,2)

    overlay_t = cv2.imread(imgShirtName, -1)  # -1 loads with transparency

    def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
        """
        @brief      Overlays a transparant PNG onto another image using CV2

        @param      background_img    The background image
        @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
        @param      x                 x location to place the top-left corner of our overlay
        @param      y                 y location to place the top-left corner of our overlay
        @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None

        @return     Background image with overlay on top
        """

        bg_img = background_img.copy()

        if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

        # Extract the alpha mask of the RGBA image, convert to RGB
        b, g, r, a = cv2.split(img_to_overlay_t)
        overlay_color = cv2.merge((b, g, r))

        # Apply some simple filtering to remove edge noise
        mask = cv2.medianBlur(a, 5)

        h, w, _ = overlay_color.shape
        roi = bg_img[y:y + h, x:x + w]

        # Black-out the area behind the logo in our original ROI
        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))

        # Mask out the logo from the logo image.
        img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

        # Update the original image with our new ROI
        bg_img[y:y + h, x:x + w] = cv2.add(img1_bg, img2_fg)

        return bg_img

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!DE FACUT RESIZE LA TRICOU!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    cv2.imshow('Shirt', overlay_transparent(imgBackground, overlay_t, punctUpLeft[0], punctUpLeft[1],
                (abs(punctUpLeft[0] - punctDownRight[0])+30, abs(punctUpLeft[1] - punctDownRight[1]))))

# =============================================
def takePicture():
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Take Picture")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        frame = cv2.rectangle(frame, (225, 75), (450, 320), (255, 0, 0), 2)
        frame = cv2.putText(frame, "Press space to take picture", (200, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)
        cv2.imshow("Take picture", frame)

        k = cv2.waitKey(1)
        if k % 256 == 32:
            # SPACE pressed
            cv2.destroyAllWindows()
            numberOfSeconds = 10
            t_end = time.time() + numberOfSeconds
            while time.time() < t_end:
                ret, frame = cam.read()
                if t_end - time.time() > 1:
                    frame = cv2.rectangle(frame, (225, 75), (450, 320), (255, 0, 0), 2)
                    frame = cv2.putText(frame, "{}".format(int(t_end - time.time())), (300, 460),
                                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
                cv2.imshow("Take picture", frame)

                cv2.waitKey(1)
            img_name = "pictureToProcess.png"
            cv2.imwrite(img_name, frame)
            print("{} saved!".format(img_name))
            break
    cam.release()
    cv2.destroyAllWindows()
    return frame

# ============================================= Main function
def Work():
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QMainWindow()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()

    image = cv2.imread("opencv_frame.png")
    isPhotoUploaded = False

    while True:
        if isPhotoUploaded:
            overlay(points, image, ui.shirtName)

        if ui.changePhoto:
            image = takePicture()
            points = alg(image)
            ui.changePhoto = False
            isPhotoUploaded = True

        if ui.changeProgram:
            cv2.destroyAllWindows()
            Form.hide()
            windowLipsticks.Work()

        if ui.uploadedPhoto:
            image = cv2.imread(ui.pathToUploadedFile)
            points = alg(image)
            ui.uploadedPhoto = False
            isPhotoUploaded = True

        if ui.quit:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    Work()
    """app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QMainWindow()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()

    image = cv2.imread("opencv_frame.png")
    isPhotoUploaded = False

    while True:
        if isPhotoUploaded:
            overlay(points, image, ui.shirtName)
        if ui.changePhoto:
            image = takePicture()
            points = alg(image)
            ui.changePhoto = False
            isPhotoUploaded = True
        if ui.quit:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break"""
