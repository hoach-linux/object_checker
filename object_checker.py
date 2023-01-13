import pyautogui
import schedule
import time
import cv2
import mss
import numpy as np
import pytesseract
from fuzzywuzzy import fuzz
import faulthandler


def run_program():
    # code to execute the program here
    # with mss() as sct:
    #   sct.shot(mon=0)

    print("Hello world")


def queryMousePosition():
    x, y = pyautogui.position()
    # positionStr = 'X:' + str(x).rjust(4) + '  Y:' + str(y).rjust(4)
    # print(positionStr, end='')

    # print('\b'*len(positionStr), end='', flush=True)

    return {"x": x, "y": y}

def click(a, b):
    pyautogui.click(x=a, y=b)
    time.sleep(0.1)
    
    return


schedule.every().day.at("22:53").do(run_program)
title = "AutoClick Bot"
sct = mss.mss()

ocr = {
    "enabled": True,
    "exclude": False,
    "list": []
}
last_catch_time = 0

while True:
    cursor = queryMousePosition()
    mon = {"top": 0, "left": 0, "width": 1919, "height": 1079}
    mon = {"top": cursor["y"] - 200, "left": cursor["x"] - 200, "width": 400, "height": 400}
    img = np.asarray(sct.grab(mon))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    green_lower = np.array([25, 52, 72], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    kernal = np.ones((5, 5), "uint8")

    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(img, img, mask = green_mask)

    contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            if ocr["enabled"]: 
                last_catch_time = time.time()

                while time.time() - last_catch_time < 2:
                    img = np.asarray(sct.grab(mon))
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  

                    pcm6 = pytesseract.image_to_string(rgb, lang='eng', config='--psm 6')
                    pcm7 = pytesseract.image_to_string(rgb, lang='eng', config='--psm 7')
                    find_width_in_pcm6 = pcm6.find("width")
                    find_width_in_pcm7 = pcm7.find("width")

                    if(find_width_in_pcm6) or (find_width_in_pcm7): 
                        if((fuzz.ratio(pcm6[find_width_in_pcm6:find_width_in_pcm6 + len("width")], "width")) > 50 or (fuzz.ratio(pcm7[find_width_in_pcm7:find_width_in_pcm7 + len("width")], "width")) > 50):
                            print("This is Width")
                        else:
                            print("No")

                            print(pcm6, pcm7)

                    
            else:
                last_catch_time = time.time()

    cv2.imshow(title, img)
    if cv2.waitKey(25) & 0xFF == ord("q"): 
        cv2.destroyAllWindows()
        break

    schedule.run_pending()

    time.sleep(0.1)