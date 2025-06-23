import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import requests
import threading
import queue


path = './ImagesAttendance'
unknown_faces_path = './UnknownFaces'
images = []
classNames = []
myList = os.listdir(path)
print(f"Images found: {myList}")

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(f"Class names: {classNames}")


def send_email_alert_mailbluster(to_email, subject, body):
    API_KEY = 'e313cf59-d85d-4d29-9b00-cad3af9576c5' 
    API_URL = 'https://app.mailbluster.com/K37jeedQEg'
    
    email_data = {
        'from': 'anishsaini450@gmail.com', 
        'to': to_email,  
        'subject': subject, 
        'body': body,  
        'is_html': False  
    }

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(API_URL, json=email_data, headers=headers)
        print(f"Email API response status code: {response.status_code}")
        print(f"Email API response body: {response.text}")
        if response.status_code == 200:
            print("Email sent successfully!")
        else:
            print(f"Failed to send email. Status code: {response.status_code}")
            print(response.json())
    except Exception as e:
        print(f"An error occurred: {e}")


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(img)
        if encodings:
            encodeList.append(encodings[0])
        else:
            print("No faces found in image.")
    return encodeList


def markAttendance(name, markedNames):
    if name not in markedNames:
        with open('att.csv', 'r+') as f:
            myDataList = f.readlines()
            nameList = [line.split(',')[0] for line in myDataList]

            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%Y-%m-%d %H:%M:%S')
                f.writelines(f'\n{name},{dtString}')
                markedNames.add(name)
                print(f"Attendance marked for {name} at {dtString}")
            else:
                print(f"Attendance already marked for {name}")
    else:
        print(f"Attendance already marked for {name}")


def saveUnknownFace(faceImg):
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f'{unknown_faces_path}/Unknown_{timestamp}.jpg'
    cv2.imwrite(filename, faceImg)
    print(f"Unknown face saved as {filename}")


def process_frame(frame, encodeListKnown, classNames, markedNames, known_unknown_encodings):
    imgS = cv2.resize(frame, (0, 0), None, 0.125, 0.125)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS, model='cnn')
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    results = []

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex] and faceDis[matchIndex] < 0.42:
            name = classNames[matchIndex].upper()
            if name not in markedNames:
                markAttendance(name, markedNames)
            color = (0, 255, 0)
        else:
            name = 'Unknown'
            color = (0, 0, 255)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 8, x2 * 8, y2 * 8, x1 * 8
            faceImg = frame[y1:y2, x1:x2]  
            faceImgRGB = cv2.cvtColor(faceImg, cv2.COLOR_BGR2RGB) 
            
            unknown_encoding = face_recognition.face_encodings(faceImgRGB)
            if unknown_encoding:
                unknown_encoding = unknown_encoding[0]
                if not any(face_recognition.compare_faces(known_unknown_encodings, unknown_encoding)):
                    saveUnknownFace(faceImg)  
                    known_unknown_encodings.append(unknown_encoding)  
                    send_email_alert_mailbluster( 
                        'vdubey8511@gmail.com',
                        'Alert: Unknown Face Detected',
                        f'An unknown face was detected at {datetime.now()}. Please check the webcam feed.'
                    )
                else:
                    print("Duplicate unknown face detected, not saving.")
            
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 8, x2 * 8, y2 * 8, x1 * 8
        results.append((name, color, (x1, y1, x2, y2)))

    return results


def main():
    encodeListKnown = findEncodings(images)
    print("Encoding Done")

    markedNames = set()
    known_unknown_encodings = [] 

    cap = cv2.VideoCapture(0)

    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)

    def capture_frames():
        while True:
            success, frame = cap.read()
            if not success:
                break
            if not frame_queue.full():
                frame_queue.put(frame)

    def process_frames():
        while True:
            if not frame_queue.empty():
                frame = frame_queue.get()
                results = process_frame(frame, encodeListKnown, classNames, markedNames, known_unknown_encodings)
                if not result_queue.full():
                    result_queue.put((frame, results))

    capture_thread = threading.Thread(target=capture_frames, daemon=True)
    process_thread = threading.Thread(target=process_frames, daemon=True)

    capture_thread.start()
    process_thread.start()

    while True:
        if not result_queue.empty():
            frame, results = result_queue.get()
            for name, color, (x1, y1, x2, y2) in results:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                text_size = cv2.getTextSize(name, font, font_scale, thickness)[0]
                text_width, text_height = text_size

                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
                cv2.putText(frame, name, (x1 + (x2 - x1 - text_width) // 2, y2 - 10), font, font_scale, (255, 255, 255), thickness)

            cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
