from keras_facenet import FaceNet
from retinaface import RetinaFace
import face_recognition as fr
import matplotlib.pyplot as plt
import os
import PIL
from PIL import Image, ImageDraw
import cv2
import numpy as np
import moviepy
from moviepy.editor import VideoFileClip
from moviepy.video.io import ImageSequenceClip
#from sklearn import *

embedder = FaceNet()

img1_path = "C:/Users/13215/Desktop/Test/img1.jpg"

faces = RetinaFace.extract_faces(img_path = img1_path, align = True)

for face in faces:
    plt.imshow(face)
    plt.show()

match_embeddings = embedder.embeddings(faces)

vid1_path = "C:/Users/13215/Desktop/Test/vid1.mp4"

vid1 = VideoFileClip(vid1_path)

vid1_audio = vid1.audio

#command = "ffmpeg -i ", vid1_path, " -ab 160k -ac 2 ar 44100 -vn audio.wav"

#subprocess.call(command, shell=True)

vidObj = cv2.VideoCapture(vid1_path)
fps = vidObj.get(cv2.CAP_PROP_FPS)

censor_filename = "censor_clip.mp4"

frames = []

success = 1

while success:
        success, image = vidObj.read()

        npimage = np.array(image)

        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        except cv2.error:
            success = False
            print("Video Processed")

        locations = []
        if success: 
            faces = RetinaFace.detect_faces(npimage)
            
            for _, identity in faces.items():
                locations.append(identity["facial_area"])
            faces = RetinaFace.extract_faces(img_path = npimage, align = True)

            embeddings = embedder.embeddings(faces)

            maxdist = 0
            i = 0

            pil_image = Image.fromarray(image)

            draw = ImageDraw.Draw(pil_image)
            distances = []
            for (left, top, right, bottom), embedding in zip(locations, embeddings):
                for match in match_embeddings:
                    distances.append(embedder.compute_distance(match, embedding))

                best_match_index = np.argmin(distances)
                draw.rectangle(((left, top), (right, bottom)), fill=(0,0,0))
                cv2.rectangle(image, (left, top), (left + right, top + bottom), (0, 0, 0), 2)

            cv2.imshow('window-name', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            del draw
            

            #plt.imshow(pil_image)

            #plt.show()
            frames.append(pil_image)
                                     
        #count += 1

censor_clip = ImageSequenceClip.ImageSequenceClip(frames, fps)

censor_video = censor_clip.set_audio(vid1_audio)

censor_video.write_videofile(censor_filename)
    
