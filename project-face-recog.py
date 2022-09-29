#importing important libraries 
import cv2
from imutils.video import VideoStream
#import Keras
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import face_recognition as f
from datetime import datetime as dt

#bringing in images from folders
def list_images():
    images = list(paths.list_images('./'))
    return(images)
images = list_images()
print(images)

#reading into cv2
def show_image(image):
    for img in image:
        image_1 = cv2.imread(image)
        #copy image
        image_copy = np.copy(image_1)
        #original image
        image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
        #copy rgb to gray
        image_1_gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

        plt.imshow(image_1, cmap ='gray')

def detect_faces(image_orig):
    #new_size_image_copy = cv2.resize(image_copy, (300,300))
    #new_size_image_origs = cv2.resize(image_orig, (0, 0), )
    #image_orig = cv2.resize(image_orig, (0, 0), None, 0.25, 0.25)
    #copy rgb to gray
    #image_origs = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)
    
    face_detect_weight = 'haarcascade_frontalface_default.xml'
    #resizing the image to make it the same throughout
    
    #using the haarcascade weight
    net = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    #print(net.empty())
    #fa = FaceAligner(net, desiredFaceWidth=256)
    #detecting the face
    
    
    face = net.detectMultiScale(image_origs, 1.3, 5)
    #print(face)
    
    return [face]
            

def encode(img):
    face_name = []
    known_encodings = []
    y = 0
    print(img)
    for im in img:
        print('image-link-',im)
        t = im.split('\\')
        x = t[-2]
        face_name.append(x)   
        known_img = f.load_image_file(im)
        known_img = cv2.resize(known_img, (300,300))
        known_img = cv2.cvtColor(known_img, cv2.COLOR_BGR2RGB)
        known_enc = f.face_encodings(known_img)[0]
        known_encodings.append(known_enc)
        y += 1
        print('image {} done'.format(y)) 
    print('Encoding images complete!')
   
    
    return known_encodings, face_name


def attendance_add(name):
    with open('Attendance.csv','r+') as fe:
        lines = fe.readlines()
        
        
        names = []
        for line in lines:
            name_time = line.split(',')
            the_name = name_time[0]
            names.append(the_name)
        if name not in names:
            now = dt.now()
            form = now.strftime('%H:%M:%S')
            fe.writelines(f'\n{name},{form}')
        
   
known_enc, known_name = encode(images)
#print(known_name)
processed = True
print("[INFO] starting video stream...")
video_reading = cv2.VideoCapture(0)


#print(known_enc)
print('processing...')
while True:
    ret,val = video_reading.read()
    #resizing images to 1/4 of its original size for faster processing
    image_orig = cv2.resize(val, (0, 0), fx = 0.25, fy = 0.25)
    #converting frame to RGB
    image_origs = cv2.cvtColor(image_orig, cv2.COLOR_BGR2RGB)

    if processed:
        
        #find the locations of the faces present in the current picture frame
        locations = f.face_locations(image_origs)
        #print('locations',locations)
        encodings = f.face_encodings(image_origs, locations)
        face_names = []
        for encoding, location in zip(encodings, locations):
            
            matches = f.compare_faces(known_enc, encoding)
            #print(matches)
            name = 'Unknown'

            #if True in matches:
                 #first_match_index = matches.index(True)
                 #name = known_name[first_match_index]

            face_distance = f.face_distance(known_enc, encoding)
            #print('facedistance',face_distance)
            if np.min(face_distance) >0.50:
                name = 'Unknown'
            else:
                best_match_index = np.argmin(face_distance)
                #print(best_match_index)
                if matches[best_match_index]:
                    name = known_name[best_match_index]

            
            #print(face_names)
            y,w,h,x = location
            x *= 4
            y *= 4
            h *= 4
            w *= 4
        
        
            cv2.rectangle(val, (x,y), (w, h), (0,255,0), 1)
            cv2.rectangle(val, (x, h - 35), ( w,h ), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(val, name, (x+ 6,  h- 6), font, 1.0, (255,255, 255))
            attendance_add(name)
            
    cv2.imshow('Video', val)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
    


#for (y,w,h,x),name in zip(locations,face_names):

