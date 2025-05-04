import cv2
import time
import numpy as np
import pyautogui

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

# -------------------------------------------------------------------------------------------------------------------------

def ExtractWholeChessboard(image):
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = np.float32(gray)

    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(contours, key=cv2.contourArea)

    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0,0,255), 3)
    #cv2.drawContours(contour_image, [max_contour], -1, (0, 0, 255), 3)

    opposingCorners = (0, 0, 0, 0)
    maxArea = 0
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        if (w * h) > maxArea:
            maxArea = w * h
            opposingCorners = x, y, x + w, y + h 

    return gray[opposingCorners[1]:opposingCorners[3]+1, opposingCorners[0]:opposingCorners[2]+1]

# -------------------------------------------------------------------------------------------------------------------------

def ExtractChessGrid(image):

    # GRID FORMAT
    # 
    grid = {
        '1':{'a':None, 'b':None, 'c':None, 'd':None, 'e':None, 'f':None, 'g':None, 'h':None},
        '2':{'a':None, 'b':None, 'c':None, 'd':None, 'e':None, 'f':None, 'g':None, 'h':None},
        '3':{'a':None, 'b':None, 'c':None, 'd':None, 'e':None, 'f':None, 'g':None, 'h':None},
        '4':{'a':None, 'b':None, 'c':None, 'd':None, 'e':None, 'f':None, 'g':None, 'h':None},
        '5':{'a':None, 'b':None, 'c':None, 'd':None, 'e':None, 'f':None, 'g':None, 'h':None},
        '6':{'a':None, 'b':None, 'c':None, 'd':None, 'e':None, 'f':None, 'g':None, 'h':None},
        '7':{'a':None, 'b':None, 'c':None, 'd':None, 'e':None, 'f':None, 'g':None, 'h':None},
        '8':{'a':None, 'b':None, 'c':None, 'd':None, 'e':None, 'f':None, 'g':None, 'h':None}
    }

    edges = cv2.Canny(image, 50, 150)
    #cv2.imshow('Contours', edges)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=750, minLineLength=50, maxLineGap=5)
    line_image = image.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    #cv2.imshow('Lines', line_image)

    contours, _ = cv2.findContours(line_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
    #cv2.drawContours(contour_image, contours, -1, (0,0,255), 3)

    square_counter = 0
    for cont in contours:
        x1,y1 = cont[0][0]
        approx = cv2.approxPolyDP(cont, 0.01*cv2.arcLength(cont, True), True)
        if len(approx) >= 0:
            x, y, w, h = cv2.boundingRect(cont)
            ratio = float(w)/h
            if 0.8 < ratio < 1.2 and w > 10 and h > 10 and cv2.contourArea(cont) > 50:
                square_counter += 1
                print(f"Square detected at ({x1}, {y1})")
                square = image[y:y+h, x:x+w]
                cv2.imshow(f'Square {square_counter}', square)
                cv2.waitKey(0)


    print(f"Squares detected: {square_counter}")
    cv2.imshow('Contours', contour_image)

    #return gray[opposingCorners[1]:opposingCorners[3], opposingCorners[0]:opposingCorners[2]]

# -------------------------------------------------------------------------------------------------------------------------

def AutoChess():
    image = cv2.imread('AutoChess\ChessScreenshot.png', cv2.IMREAD_COLOR)
    #image = np.array(pyautogui.screenshot())
    chessboard = ExtractWholeChessboard(image)
    ExtractChessGrid(chessboard)

    cv2.waitKey(0)
    # print(f"Total corners detected: {count}")
    cv2.destroyAllWindows()

#time.sleep(3) 
AutoChess()