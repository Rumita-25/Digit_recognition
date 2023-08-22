import pygame, sys
from pygame.locals import *
import pygame.surfarray
import numpy as np
from keras.models import load_model
import cv2

window_x=640
window_y=490

BOUNDARYINC= 5
WHITE= (255,255,255)
BLACK= (0,0,0)
RED= (255,0,0)

img_save= False

model = load_model("C:\\Users\\rumit\\Desktop\\Bharat intern\\Data Science\\bestmodel.h5")


labels={0:"Zero",1:"One",2:"Two",3:"Three",4:"Four",5:"Five",6:"Six",7:"Seven",8:"Eight",9:"Nine"}
#intializing pygame
pygame.init()

FONT= pygame.font.Font('FreeSans.ttf', 20)
DISPLAYSURF= pygame.display.set_mode((window_x, window_y))

WHILE_INIT= DISPLAYSURF.map_rgb(WHITE)

pygame.display.set_caption("Digit Recognizer")

is_writing=False
num_x_cord=[]
num_y_cord=[]
image_count=1
PREDICT=True

while True:
    for event in pygame.event.get():
        if event.type== QUIT:
            pygame.quit()
            sys.exit()
        if event.type ==MOUSEMOTION and is_writing:
            x_cord, y_cord= event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (x_cord, y_cord), 4,0)
            
            num_x_cord.append(x_cord)
            num_y_cord.append(y_cord)
            
        if event.type== MOUSEBUTTONDOWN:
            is_writing= True
        if event.type== MOUSEBUTTONUP:
            is_writing=False
            num_x_cord= sorted(num_x_cord)  #creating a rectangle around the drawn digit
            num_y_cord= sorted(num_y_cord)
            
            rect_min_x, rect_max_x =max(num_x_cord[0]-BOUNDARYINC, 0), min(window_x, num_x_cord[0]+BOUNDARYINC)  #for LHS & RHS, -1 index is taken as it is the last value
            
            rect_min_y, rect_max_y = max(num_y_cord[0]-BOUNDARYINC, 0), min(window_y, num_y_cord[0]+BOUNDARYINC) #for LS & RHS 
            
            #re-initialising
            num_x_cord= []
            num_y_cord=[]
        

# ...

            pygame.surfarray.use_arraytype("numpy")  # Set the array type to numpy
            img_arr = pygame.surfarray.array3d(DISPLAYSURF)[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float64)
            
            resized_image = cv2.resize(img_arr, (28, 28))  # Resize to 28x28
            preprocessed_image = resized_image.reshape(1, 28, 28,1)
            prediction = model.predict(preprocessed_image)
            predicted_label = labels[np.argmax(prediction)]



           # img_arr= np.array(pygame.surfarray(DISPLAYSURF))[rect_min_x: rect_max_x, rect_min_y:rect_max_y].T.astype(np.float64) #extracting the part which is inside the rectangle
            #taking the transpose of the metrics by .T.astype()
            
            #saving the image
            if img_save:
                cv2.imwrite("image.png")
                image_count += 1
                
                #INCORPORATING PYTHON WITH OUR ML MODEL
            if PREDICT:
                image=cv2.resize(img_arr, (28,28))
                image= np.pad(image, (10,10), 'constant',constant_values=0)
                image= cv2.resize(image, (28,28))/255  #normalizing the image
                
                labels=str(labels[np.argmax(model.predict(image.reshape(1,28,28,1)))])
                
                text_surf= FONT.render(labels, True, RED,WHITE)
                text_rect_obj=text_surf.get_rect()
                text_rect_obj.left, text_rect_obj.bottom= rect_min_x, rect_max_y
                
                DISPLAYSURF.blit(text_surf, text_rect_obj)
                
                if event.type==KEYDOWN:
                    if event.unicode=="n":
                        DISPLAYSURF.fill(BLACK)
                        
    pygame.display.update()
                