# YOLOv5-Object-Classification
This repo acts as a tutorial guide for training and running YOLOv5 for FRC games. Made by LASER3284.

## Why YOLO?
YOLO is an algorithm that uses neural networks to provide real-time object detection. This algorithm is popular because of its speed and accuracy. It has been used in various applications to detect traffic signals, people, parking meters, and animals. Additionally, using a method of AI object detection is often faster than manually building a custom detection algorithm with something like OpenCV.

# How to run YOLO
## 1 Setting up Your Python Environment
After cloning this repo and navigating into its directory, [install python](https://www.python.org/downloads/) and then setup your pip environment by running this command:
```
pipenv install
```
**This command will install all the packages from the pipfile and will take A LONG TIME. Be patient, the terminal will print out success when finished.**
If you're on windows some of the dependencies may fail to install. You will have to troubleshoot them manually.

Then, enter your environment by typing:
```
pipenv shell
```

## 2 Creating a Dataset for Training
In order to train a YOLO neural network, you first need to gather data about the objects you want to track. In our case, we want to compile a bunch of images from the FRC game feild. But you can't just collect a bunch of images and throw it at a training algorithm, you have to tell the algorithm what things in your images are objects. I find that the easiest way to do this is by using an annotation program or website to label and export your images. 
   ###### Good Annotation Tools:
   - RoboFlow (Free if you make your projects public, also my favorite)
   - Labelimg
   - LabelMe
   - CVAT

The annotation process consists of going through each image and drawing a bounding box of the object you want to detect and giving that bounding box a label consistant and relating to the object. **Once your annotation is done, resize your data to 640x640 and export it in YOLOv5 or any compatible YOLO format.**

**I have compiled and annotated some images for the FRC 2022 game, they can be found [here](https://drive.google.com/drive/folders/1jMB4qO-iwuESWnIYX0BQRPlOYbCC2tI7?usp=sharing)**
### Example
  Annotation:
  ![Example of annotating an image with RoboFlow](https://miro.medium.com/max/1400/0*wApVYCGhdmAXSjuo)
  When using RoboFlow you can apply resizing and augment images automatically. Augmentation allows you to create mutliple different images from one, increasing your dataset size. **Careful though, to many augmentations can confuse the neural network!**
  
  ![Example of annotating an image with RoboFlow](https://miro.medium.com/max/500/1*w3BcUrZ4Y7xadXTIqklg9w.png)
  ![Example of annotating an image with RoboFlow](https://miro.medium.com/max/500/1*opfegllnoEDsA2T_NdZ3BQ.png)
  
## 3 Organizing Your Dataset for YOLO
Now that you have your labeled data, we need to organize it so that our learning algorithm can find the images and labels. Organize your train images and labels like so: YOLOv5 assumes /coco128 is inside a /datasets directory next to the /yolov5 directory. YOLOv5 locates labels automatically for each image by replacing the last instance of /images/ in each image path with /labels/. Refer to the datasets folder in this repo if you are still unsure of the file structure.

Your annotation program should have also given you a yaml file. This file is used by the learning algorithm to locate the images and annotation files, it also stores a list of the object types. You will need edit the yaml to specify the path of the dataset images yourself. It should be similar to the one in the datasets folder in this repo that looks like this. (your object list will differ from the example file)
```
path: ../datasets/coco128  # dataset root dir
train: images  # train images (relative to 'path')
val: images

nc: 5
names: ['OBJECT0', 'OBJECT1', 'OBJECT2', 'OBJECT3', 'OBJECT4']
```

## 4 Training Your Convolutional Neural Network
Finally, we made it past the difficult and tedious part of YOLO. Next, we will actually begin building a neural network from our data. All we have to do is run one command: (make sure you have entered your pip environment or you'll get dependency errors)
```
python3 train.py --img 640 --batch -1 --epochs (number of training runthroughs) --data ../datasets/(yaml name).yaml --weights yolov5s.pt
```
- The batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters. By setting this number to -1, the computer will autodetermine this for us.
- The number of epochs is a hyperparameter that defines the number of times that the learning algorithm will work through the entire training dataset. If you are getting poor results and you have a good dataset, try increasing this number.
- The yolov5s.pt file is a base weights file that the program will build onto, this will be auto downloaded if you don't have it.

While training you can look at the P, R, mAP@.5, and mAP@.5:.95 numbers to get an idea of how well your model is training. They should slowly increase to 1.00 if your model is doing well.
![Example terminal output](https://github.com/LASER3284/YOLOv5-Object-Classification/blob/main/video%20splitter/Images/terminal.png)

## Running a Test Detection on Your YOLOv5 Model
Now, we can test the model by using the detect program. The training program stores the model in the ```runs/train/exp0-9/weights/``` as a ```best.pt``` file.
Run the detection program with the command:
```
python3 detect.py --weights runs/train/exp/weights/best.pt --img-size 640 --source (path to image or video files to run detection on)
```
![Example of detection](https://github.com/LASER3284/YOLOv5-Object-Classification/blob/main/video%20splitter/Images/demo.png)

After the program is done it will store the resulting images or video file in the ```runs/detect/``` directory.

## 5 Deploy Trained Model in Something Cool
Now, it's up to you to run the best.pt model on something cool like a raspberry pi to detect objects. You can use this tutorial to setup the pip environment on anything else and modify the detect.py from the yolov5 folder to do the detection.


### More Tutorials and Helpful Links
https://github.com/LASER3284/YOLOv5-Object-Classification/tree/main/yolov5
https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data
https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results
https://www.stereolabs.com/docs/object-detection/custom-od/#code-example
