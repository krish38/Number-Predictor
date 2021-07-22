import pygame
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Defining Variables
width, height = 280, 280
box = 15
rows = 28
FPS = 500

win = pygame.display.set_mode((width,height))
clock = pygame.time.Clock()

# Colours
black = (0,0,0)
white = (255,255,255)
red = (255,0,0)
green = (0,255,0)
blue = (0,0,255)

model = tf.keras.models.load_model("Number-Predictor/numpredictor/m.model")

def draw():
    pos = pygame.mouse.get_pos()
    pygame.draw.rect(win, black, (pos[0]-(pos[0]%10), pos[1]-(pos[1]%10), 2*box, 2*box))

done = False 
win.fill(white)
print("Press Enter to choose number")
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.display.quit(), sys.exit()
        if pygame.mouse.get_pressed()[0]:
            draw()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                pixelArray = [[0 for j in range(28)] for i in range(28)]
                for i in range(28):
                    for j in range(28):
                        pixelCol = win.get_at((j*10,i*10))
                        if pixelCol == (0,0,0,255):
                            pixelArray[i][j] = 1
                done = True
                break
            
    pygame.display.flip()
    clock.tick(FPS)

results = model.predict(np.expand_dims(pixelArray,0))
print(np.expand_dims(pixelArray,0).ndim)
print(f"I predict the number is: {np.argmax(results)}")

pygame.display.quit(), sys.exit()
