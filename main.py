import cv2
from tracker import *

tracker = EuclideanDistTracker()

goruntu = cv2.VideoCapture("trafik.mp4")

nesne_tespit = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
    ret, ana_goruntu = goruntu.read()
    goruntu_kesiti = ana_goruntu[340: 720, 500: 800]


    maskelenmis_goruntu = nesne_tespit.apply(goruntu_kesiti)
    _, maskelenmis_goruntu = cv2.threshold(maskelenmis_goruntu, 254, 255, cv2.THRESH_BINARY)
    kontur, _ = cv2.findContours(maskelenmis_goruntu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    motorsikletler = []

    for i in kontur:

        alan = cv2.contourArea(i)
        if alan > 300:
            x, y, w, h = cv2.boundingRect(i)
            if w<32 & 50<h:
                motorsikletler.append([x, y, w, h])

    boxes_ids = tracker.update(motorsikletler)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id

        cv2.putText(goruntu_kesiti, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.rectangle(goruntu_kesiti, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("Kesiti Alinmis Goruntu", goruntu_kesiti)
    cv2.imshow("Kameradan Alinan Asil Goruntu", ana_goruntu)
    cv2.imshow("Maskelenmis Goruntu", maskelenmis_goruntu)

    key = cv2.waitKey(30)
    if key == 27:
        break

goruntu.release()
cv2.destroyAllWindows()