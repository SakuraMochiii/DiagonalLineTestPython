import math
import os
import numpy as np
import sys
import cv2 as cv
def show_wait_destroy(winname, img): #display image
    cv.imshow(winname, img)
    cv.waitKey(0)
    cv.destroyWindow(winname)

def read_img(img):
    directory_path = os.path.dirname(__file__)
    file_path = os.path.join(directory_path, "assets/" + img) #img path
    src = cv.imread(file_path, cv.IMREAD_COLOR)

    h, w = src.shape[:2]
    new_w = 500
    new_h = int(new_w * w / h)
    return cv.resize(src, (new_h, new_w)) #resize image to 500 px wide

def preprocess(src, invert):
    norm = np.zeros((src.shape[0], src.shape[1]))
    img = cv.normalize(src, norm, 0, 255, cv.NORM_MINMAX)
    denoised = cv.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15)
    buf = cv.addWeighted(denoised, 1.2, denoised, 0, 1.2) 
    gray = cv.cvtColor(buf, cv.COLOR_BGR2GRAY)
    if not invert: #if line is black
        gray = cv.bitwise_not(gray)
    return gray

def find_square(gray):
    edges = cv.Canny(gray, 200, 200)
    cnts = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) #draw outline
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    blank = np.zeros(gray.shape, dtype=np.uint8)
    cropped = cv.drawContours(blank,cnts,-1,(255,255,255),1)
    filtered = cv.bitwise_and(~gray, ~gray,mask = cropped)
    return filtered

def find_horiz(filtered):
    bw = cv.adaptiveThreshold(filtered, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    horizontal = np.copy(bw)
    cols = horizontal.shape[1]
    horizontal_size = cols // 20
    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)
    edges = cv.Canny(horizontal, 100, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 2, np.pi/180, 130)
    return lines

def find_vert(filtered):
    bw = cv.adaptiveThreshold(filtered, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    # find vertical
    vertical = np.copy(~bw)
    rows = vertical.shape[0]
    verticalsize = rows // 30
    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))
    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)
    vertical = cv.bitwise_not(vertical)

    edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    kernel = np.ones((2, 2), np.uint8)
    edges = cv.dilate(edges, kernel)
    smooth = np.copy(vertical)
    smooth = cv.blur(smooth, (2, 2))
    (rows, cols) = np.where(edges != 0)
    vertical[rows, cols] = smooth[rows, cols]

    edges = cv.Canny(vertical, 100, 150, apertureSize=3)
    lines = cv.HoughLines(edges, 1, np.pi/180, 130)
    return lines

def getAxis(theta):
    if math.isclose(theta, 1.5, rel_tol=0.2):
        return 0
    if math.isclose(theta, 0, rel_tol=0.5):
        return 1

def draw_lines(lines, draw):
    horiz = []
    vert = []

    for r_theta in lines:
        arr = np.array(r_theta[0], dtype=np.float64)
        r, theta = arr
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*r
        y0 = b*r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        axis = getAxis(theta)
        if axis == 0: #horiz lines save y value
            drawLine = True
            for i in horiz:
                if math.isclose(y0, i, rel_tol = 0.6):
                    drawLine = False
                    break
            if drawLine:
                horiz.append(y0)
                cv.line(draw, (x1, y1), (x2, y2), (255, 200, 180), 2)      
        if axis == 1: #vert lines save x value
            drawLine = True
            for i in vert:
                if math.isclose(x0, i, rel_tol = 0.2):
                    drawLine = False
                    break
            if drawLine:
                vert.append(x0)
                cv.line(draw, (x1, y1), (x2, y2), (255, 105, 180), 2)
    if len(horiz) > 2:
        print("horizontal error detected")
    if len(vert) > 1:
        print("vert error detected")
        
def find_lines(src, file_name):
    draw = np.copy(src)
    gray = preprocess(draw, False)

    buf = find_square(gray) #find black lines

    hlines = find_horiz(gray)
    vlines = find_vert(buf)
    if hlines is not None:
        draw_lines(hlines, draw)
    if vlines is not None:
        draw_lines(vlines, draw)

    filteredi = preprocess(draw, True)
    bufi = find_square(filteredi) #find white lines

    hilines = find_horiz(filteredi)
    vilines = find_vert(bufi)
    if hilines is not None:
        draw_lines(hilines, draw)
    if vilines is not None:
        draw_lines(vilines, draw)
    return draw

def main(argv):
    #read file
    src = read_img(argv[0])

    final = find_lines(src, argv[0])
    show_wait_destroy(argv[0] + ": detected lines", final)
    return 0
# if __name__ == "__main__":
#     main(sys.argv[1:])

main(["test8.jpg"])
main(["test3.jpg"])
main(["jam1.jpg"])
main(["norm1.jpg"])
main(["test1.jpg"])
main(["test4.jpg"])
main(["1ptvert.jpg"])
main(["1pthoriz.jpg"])
main(["1ptboth.jpg"])
main(["test6.jpg"])