import cv2 #type:ignore
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FERModel(nn.Module):
    def __init__(self):
        super(FERModel, self).__init__()
        # Define convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # Input channels: 1 (grayscale), Output channels: 32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)


        # Define pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Define fully connected layers
        self.fc1 = nn.Linear(256 * 1 * 1, 128)  # Assuming input image size is 48x48 after 4 pooling layers
        self.fc2 = nn.Linear(128, 7)  # 7 output emotions

    def forward(self, x):
        # Convolutional layers with ReLU activation and max pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))


        # Flatten feature maps
        x = torch.flatten(x, 1) # flatten all dimensions except batch      

        # Fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



model = FERModel()
model.load_state_dict(torch.load('cnnfer_net.pth',map_location=torch.device('cpu')))
model.to('cpu')

# start the webcam feed
cap = cv2.VideoCapture(0)  # 0, varsayılan kamera için

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
#cap = cv2.VideoCapture("C:\\JustDoIt\\ML\\Sample_videos\\emotion_sample6.mp4")

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')


    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = frame[y:y + h, x:x + w]
        cropped_img = cv2.resize(roi_gray_frame, (48, 48))  # Yeniden boyutlandırma
        cropped_img = cropped_img.transpose((2, 0, 1))  # Kanalların düzenlenmesi: [height, width, channels] => [channels, height, width]
        cropped_img = np.expand_dims(cropped_img, axis=0)
        # predict the emotions
        with torch.no_grad():
            emotion_prediction = model(torch.Tensor(cropped_img))
        
        maxindex = int(np.argmax(emotion_prediction))
        classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        predicted_emotion = classes[maxindex]
        cv2.putText(frame, predicted_emotion, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
