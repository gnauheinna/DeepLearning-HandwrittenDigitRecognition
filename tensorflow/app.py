import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2
import tensorflow as tf

WINDOWSIZE = (640, 480)
WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDRYINC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0,)
RED =(255, 0, 0)

IMAGESAVE = False

# Loads pretrained model
MODEL = load_model("/Users/anniehuang/Desktop/DeepLearning-HandwrittenDigitRecognition/tensorflow/model.h5")

# keeps a label array for prediction
LABELS = {0:"Zero", 1:"One", 2:"Two", 3:"Three", 4:"Four", 5:"Five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"}

# initializing the pygame
pygame.init()
FONT = pygame.font.Font("freesansbold.ttf", 18)
DISPLAYSURF = pygame.display.set_mode(WINDOWSIZE)
pygame.display.set_caption("Digit Board")

# initializing variables and counters
iswriting = False
number_xcord = []
number_ycord = []
image_cnt =1
PREDICT = True


while True:
    for event in pygame.event.get():

        # event handler for closing window
        if(event.type == QUIT):
            pygame.quit()
            sys.exit()

        # event handler for mouse motion while drawing
        if event.type == MOUSEMOTION and iswriting:
            # Capture mouse coordinates and draw a white circle on the screen
            xcord, ycord = event.pos 
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)

        # event handler for mouse button press (start drawing)
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True

        # event handler for mouse button release (stop drawing)
        if event.type == MOUSEBUTTONUP:
            iswriting = False


            # checks if there are any recorded coordinates
            if number_xcord and number_ycord:
                # sorts the xcord and ycord lists
                number_xcord = sorted(number_xcord)
                number_ycord = sorted(number_ycord)

                #  rect_min_x and rect_min_y represents the top left corner of the bounding box
                rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDRYINC, 0 ), min(WINDOWSIZEX, number_xcord[-1]+BOUNDRYINC )
                # rrect_max_x and rect_max_y epresent the bottom right corner of the bounding box
                rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDRYINC, 0 ), min(number_ycord[-1]+BOUNDRYINC,  WINDOWSIZEY)

                # clears xcord and ycord
                number_xcord = []
                number_ycord = []

                # extracts drawn region as a greyscale numpy array
                img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)

            if IMAGESAVE:
                cv2.imwrite("image.png")
                image_cnt += 1

            # incorporate python with ML
            if PREDICT:
                image = cv2.resize(img_arr, (28, 28))
                image = np.pad(image, (1, 1), 'constant', constant_values=0)
                image = cv2.resize(image,(28,28))

                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])

                textSurface = FONT.render(label, True, RED, WHITE)
                textRecObj = textSurface.get_rect()
                textRecObj.left, textRecObj.bottom = rect_min_x, rect_max_y

                # Draw a red rectangle around the drawn digit
                pygame.draw.rect(DISPLAYSURF, RED, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 1)


                DISPLAYSURF.blit(textSurface, textRecObj)

        # event handler for keyboard press (refreshes the window)
        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()

