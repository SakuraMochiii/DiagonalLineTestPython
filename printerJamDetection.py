import math
import os
import numpy as np
import sys
import cv2 as cv
def show_wait_destroy(winname, img):
    cv.imshow(winname, img)
    # cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

def read_img(img):
    directory_path = os.path.dirname(__file__)
    file_path = os.path.join(directory_path, img) #img path
    src = cv.imread(file_path, cv.IMREAD_COLOR)

    h, w = src.shape[:2]
    new_w = 500
    new_h = int(new_w * w / h)
    return cv.resize(src, (new_h, new_w))

def preprocess(src, invert):
    norm = np.zeros((src.shape[0], src.shape[1]))
    img = cv.normalize(src, norm, 0, 255, cv.NORM_MINMAX)

    # coords = np.column_stack(np.where(img > 0))
    # angle = cv.minAreaRect(coords)[-1]
    # if angle < -45:
    #     angle = -(90 + angle)
    # else:
    #     angle = -angle
    #     (h, w) = img.shape[:2]
    # center = (w //2, h //2)
    # M = cv.getRotationMatrix2D(center, angle, 1.0)
    # rotated = cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
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
    area_treshold = 5000

    blank = np.zeros(gray.shape, dtype=np.uint8)
    # blank = cv.cvtColor(blank, cv.COLOR_BGR2GRAY)
    cropped = blank.copy()
    # for c in cnts:
        # if cv.contourArea(c) > area_treshold:
            # epsilon = 0.05*cv.arcLength(c,True)
            # c2 = cv.approxPolyDP(c,epsilon,True)
        #     if len(c2) == 4:
        #         cropped = cv.drawContours(blank, [c2], -1, (255, 255, 255), cv.FILLED)
        #     else:
        #         rect = cv.minAreaRect(c)
        #         box = cv.boxPoints(rect)
        #         box = np.intp(box)
        #         cropped = cv.drawContours(blank,[box],-1,(255,255,255),cv.FILLED)
        # rect = cv.minAreaRect(c)
        # box = cv.boxPoints(rect)
        # box = np.intp(box)
        # cropped = cv.drawContours(blank,[c],-1,(255,255,255),1)
    # show_wait_destroy("cropped", cropped)
    cropped = cv.drawContours(blank,cnts,-1,(255,255,255),1)


    filtered = cv.bitwise_and(~gray, ~gray,mask = cropped)
    # border = cv.findContours(blank, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    # filtered = cv.drawContours(filtered, border[0], -1, (255, 255, 255), 5)
    # cv.imwrite("filtered.jpg", ~filtered)
    return filtered

def find_horiz(filtered):
    #find black horizontal lines
    bw = cv.adaptiveThreshold(filtered, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    # show_wait_destroy("gray", bw)
    
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
    verticalsize = rows // 20
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
    print("theta: ", theta)
    if math.isclose(theta, 1.5, rel_tol=0.2):
        return 0
    if math.isclose(theta, 0, rel_tol=0.5):
        return 1
    # return 1

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
        # cv.line(draw, (x1, y1), (x2, y2), (255, 105, 180), 2)
        # drawLine = True
        if axis == 0: #horiz lines save y value
            drawLine = True
            for i in horiz:
                if math.isclose(y0, i, rel_tol = 0.6):
                    # print("y0: ", y0)
                    # print("i: ", i)
                    drawLine = False
                    break
            if drawLine:
                horiz.append(y0)
                cv.line(draw, (x1, y1), (x2, y2), (255, 200, 180), 2)      
        # drawLine = True
        if axis == 1: #vert lines save x value
            drawLine = True
            for i in vert:
                if math.isclose(x0, i, rel_tol = 0.2):
                    print("y0: ", x0)
                    print("i: ", i)
                    drawLine = False
                    break
            if drawLine:
                vert.append(x0)
                cv.line(draw, (x1, y1), (x2, y2), (255, 105, 180), 2)
    print(vert)
    print(horiz)
        
def find_lines(src, file_name):
    draw = np.copy(src)
    gray = preprocess(draw, False)

    buf = find_square(gray) #find black lines

    hlines = find_horiz(gray)
    vlines = find_vert(buf)
    total = 0
    if hlines is None:
        print(file_name + ": no horizontal overlap errors detected")
        total += 1
    else:
        draw_lines(hlines, draw)
    if vlines is None:
        print(file_name + ": no vertical overlap errors detected")
        total += 1
    else:
        draw_lines(vlines, draw)

    filteredi = preprocess(draw, True)
    bufi = find_square(filteredi) #find white lines

    hilines = find_horiz(filteredi)
    vilines = find_vert(bufi)
    if hilines is None:
        print(file_name + ": no horizontal blank errors detected")
        total += 1
    else:
        draw_lines(hilines, draw)
    if vilines is None:
        print(file_name + ": no vertical blank errors detected")
        total += 1
    else:
        draw_lines(vilines, draw)
    if total == 4:
        print("no errors")
        return 0
    return draw

def main(argv):
    #read file
    src = read_img(argv[0])
    # show_wait_destroy("original", src)
    # if ratio btwn width height off, then return error

    final = find_lines(src, argv[0])
    show_wait_destroy("detected lines", final)
    return 0
# if __name__ == "__main__":
#     main(sys.argv[1:])

main(["assets/test3.jpg"])
main(["assets/jam1.jpg"])
main(["assets/norm1.jpg"])
main(["assets/test1.jpg"])
main(["assets/test4.jpg"])
main(["assets/1ptvert.jpg"])
main(["assets/1pthoriz.jpg"])
main(["assets/1ptboth.jpg"])
main(["assets/test6.jpg"])