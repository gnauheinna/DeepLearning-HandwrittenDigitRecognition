import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WINDOWSIZE = (640, 480)

WHITE = (255, 255, 255)
BLACK = (0, 0, 0,)
RED =(255, 0, 0)

IMAGESAVE = False

MODEL = load_model("model.h5")

LABELS = {0:"Zero", 1:"One", 2:"Two", 3:"Three", 4:"Four", 5:"Five", 6:"Six", 7:"Seven", 8:"Eight", 9:"Nine"}

# create application window
pygame.init()

FONT = pygame.font.Font("freesansbold.tff", 18)

DISPLAYSURF = pygame.display.set_mode(WINDOWSIZE)

pygame.display.set_caption("Digit Board")

iswriting = False

number_xcord = []
number_ycord = []

while True:
    for event in pygame.event.get():
        if(event.type == QUIT):
            pygame.quit()
            sys.exit()
        
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos 
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True



