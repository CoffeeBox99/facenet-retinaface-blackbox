import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1
from retinaface.pre_trained_models import get_model
from retinaface.utils import vis_annotations
import albumentations as A
from tqdm import tqdm
import matplotlib.pyplot as plt 
import os
from PIL import Image, ImageDraw
import cv2, mmcv
import numpy as np
import moviepy
from moviepy.editor import VideoFileClip
from moviepy.video.io import ImageSequenceClip
import time

globalTestPath = "C:/Users/13215/Desktop/Test/" #Path containing Program

data_dir = globalTestPath + "Test3" #Folder I used to store images

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 
print('Running on device: {}'.format(device)) #Check for GPU, otherwise use CPU

if format(device) == 'cpu': # or 1 == 1:
    print("########")
    print("Be advised, that while the CPU *CAN* run this program, it will be")
    print("significantly slower than with your system's GPU, if present.")
    print()
    print("If possible, ensure that your build of PyTorch is connected to")
    print("your GPU via CUDA/CUDNN.")
    print("########")
    print()

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("FaceNet Model Loaded.") #Loads FaceNet Model

model = get_model("resnet50_2020-07-20", max_size=2048, device=device)
model.eval() #Loads RetinaFace Model (May be outdated version)
print("RetinaFace Model Loaded.")

convert_tensor = transforms.ToTensor() #Deprecated during development

transform = A.Compose([A.SmallestMaxSize(max_size=256, p=1), 
                       A.CenterCrop(height=224, width=224, p=1),
                       A.Normalize(p=1)]) #Rules for Image Transformation

def loadImg(img_path, numpy = False):
    #Load image into cv2 for reading
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if numpy:
        img = np.array(img) #Converts image to np.array

    return img

def getFaceData(image, emb = True, loc = True):
    #Retinaface gathers data on all faces in the image
    with torch.no_grad():
        faces = model.predict_jsons(image)
    
    locations = [] #2d array representing coordinates of faces in the image
    embeddings = [] #vector array representing face values
    try:
        with torch.no_grad():
            for face in tqdm(faces, disable=True):
                #save face coordinates for use later
                locations.append(face['bbox'])
                #extract bounding box coordinates
                x, y, w, h = face['bbox']
                
                #Shrink image to show only the selected face
                imgTransform = transform(image=image[y:h, x:w])['image']
                #convert shrunken image into np array, then PyTorch Tensor
                model_input = torch.from_numpy(np.transpose(imgTransform, (2, 0, 1))).to(device) 
                #Add expanded tensor value (vector array) to list of embeddings
                embeddings.append(resnet(model_input.unsqueeze(0)))
    except Exception as e: #If detection fails, return nothing
        if emb and loc:
            return 0, 0
        else:
            return 0
        
    if emb and loc:
        return embeddings, locations
    if emb and not loc:
        return embeddings
    if loc and not emb:
        return locations
    
def boxFaces(match_embeddings, image, drawType = "PIL"):
    #PIL used when black box is applied to a copy of the image being processed
    #cv2 used when black box is applied to the image itself
    
    #Get data from image
    embeddings, locations = getFaceData(image)
    #Run only if data was returned
    if embeddings != 0 and locations != 0:
        if drawType == "PIL":
            pil_image = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_image)
        
        distances = []
        
        for (left, top, right, bottom), embedding in zip(locations, embeddings):
            #For every face embedding in the image,
            #Get the euclidean distance between it and each of
            #the target embeddings
            for match in match_embeddings:
                distances.append((match - embedding).norm().item())

                #print((match - embedding).norm().item()) #Test line

            #The embedding with the closest distance is the best match
            best_match_index = np.argmin(distances)
            #Loop until absolute best match is found

        #Get coordinates of best-matching face
        left, top, right, bottom = locations[best_match_index] 
        #print(np.min(distances)) #Test line

        #Cover face with black box
        if drawType == "PIL":
            draw.rectangle(((left, top), (right, bottom)), fill=(0,0,0))
            del draw 
            return np.array(pil_image)

        else:
            if drawType == "cv2":
                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 0), -1)


def main():
    img1_path = "C:/Users/13215/Desktop/Test/img1.jpg" #Test Image

    img1 = loadImg(img1_path) #Loads Test Image

    match_embeddings = []

    ##I tried getting an aggregate embedding using several different images
    ## of myself, but I must have done something wrong, since it somehow made
    ## the model even less accurate
    """matches = []

    for image in os.listdir(data_dir): 
        avgEmbedding = 0
        img = loadImg(image)
        embeddings = getFaceData(img, loc = False)
        for embedding in embeddings:
            match_embeddings.append(embedding)

    matches.append(torch.mean(torch.stack(match_embeddings)))"""

    embeddings = getFaceData(img1, loc = False)

    #Add test image face to list of target embeddings
    for embedding in embeddings:
        match_embeddings.append(embedding)

    option = ""
    #Menu to select input format
    print("1 - Image | 2 - Video | 3 - Stream")
    print("-1 to quit")
    option = input("Choice: ")

    while option != "-1":
        print()
        match option:
            case "1":
                img2_path = globalTestPath + input("Enter file name: ")

                img2 = loadImg(img2_path)

                plt.imshow(boxFaces(match_embeddings, img2))
                plt.show()
            case "2":
                vid1_path = globalTestPath + input("Enter file name: ")
                vid1 = VideoFileClip(vid1_path); vid1_audio = vid1.audio

                video = mmcv.VideoReader(vid1_path)
                ##mmcv may not be necessary for this step, cv2 has a similar function
                frames = [np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
                frames_tracked = []

                for i, frame in enumerate(frames):
                    start_time = time.time() #Start time for operation
                    print ('\rTracking frame: {}'.format(i+1))
                    
                    frames_tracked.append(boxFaces(match_embeddings, frame))
                    
                    print (time.time() - start_time)

                #convert processed frames to .mp4 at original fps
                censor_clip = ImageSequenceClip.ImageSequenceClip(frames_tracked, video.fps)
                #Attach original audio to new .mp4 file
                censor_video = censor_clip.set_audio(vid1_audio)
                #Save .mp4 file
                censor_video.write_videofile("censor.mp4")
            case "3":
                print("Setting Up Webcam connection...")
                #Link program to computer webcam
                capture = cv2.VideoCapture(0)
                while True:
                    ret, frame = capture.read()

                    boxFaces(match_embeddings, frame, drawType = "cv2")

                    cv2.imshow("Frame", frame)

                    key = cv2.waitKey(1)
                    if key == ord("q"):
                        break

                capture.release()
                cv2.destroyAllWindows()

            case _:
                print("Sorry, that wasn't quite right. Try again.")
                    
        print()
        print("1 - Image | 2 - Video | 3 - Stream")
        print("-1 to quit")
        option = input("Choice: ")
        

if __name__ == "__main__":
    main()
