import numpy as np
from PIL import Image, ImageDraw

import cv2

def draw_it(raw_strokes):
    image = Image.new("P", (255,255), color=255)
    image_draw = ImageDraw.Draw(image)

    for stroke in eval(raw_strokes):
        for i in range(len(stroke[0])-1):

            image_draw.line([stroke[0][i], 
                             stroke[1][i],
                             stroke[0][i+1], 
                             stroke[1][i+1]],
                            fill=0, width=6)

    image.save('my.png')
    return np.array(image)


d = draw_it("[[[27,17,16,21,34,50,49,34,23,17],[47,58,73,81,84,67,54,46,47,51]],[[22,0],[51,18]],[[41,46,43],[45,11,0]],[[53,65,64,69,91,119,135,148,159,158,149,126,87,68,62],[68,68,58,51,36,34,38,48,64,78,85,90,90,83,73]],[[161,175],[70,69]],[[180,177,176,187,206,226,244,250,250,245,233,207,188,180,180],[68,67,61,50,42,40,48,58,72,80,87,89,83,76,71]],[[73,61],[85,113]],[[95,94],[88,126]],[[140,157],[90,118]],[[199,201,208],[90,116,122]],[[234,242,255],[89,105,112]]]")



