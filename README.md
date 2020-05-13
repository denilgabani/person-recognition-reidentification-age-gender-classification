# person-recognition-reidentification-age-gender-classification

## Introduction
This demo project is used to detect the person and used for person re-identification with age and gender recognition. It can export the images of detected person and also export csv file with image id and corrosponding age and gender.

## Demo

### Demo video of Synchronous version:

[![Demo video sync](https://img.youtube.com/vi/a5AhTBV9XUc/0.jpg)](https://www.youtube.com/watch?v=a5AhTBV9XUc)

### Demo video of Asynchronous version:

[![Demo video Async](https://img.youtube.com/vi/Aq81WoRY7g0/0.jpg)](https://www.youtube.com/watch?v=Aq81WoRY7g0)

**Synchronous version takes around 88 sec time and Asynchronous version takes around 58 sec time for prediction on 30 sec demo video. So asynchronous version is much faster than synchronous version.**

## Project Set Up and Installation

### Setup

#### Prerequisites
  - You need to install openvino successfully. <br/>
  See this [guide](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html) for installing openvino.

#### Step 1
Clone the repository:- https://github.com/denilDG/person-recognition-reidentification-age-gender-classification

#### Step 2
Initialize the openVINO environment:-
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```

#### Step 3

Open a new terminal in project directory and run the following commands:

1. Initialize the openVINO environment:-
```
source /opt/intel/openvino/bin/setupvars.sh -pyver 3.5
```

2. Run the app:-

**For running the synchronous version**:

```
python main.py -i media/sample2.mp4 -image_dir images -is_export_csv=True
```

**For running the asynchronous version**:

```
python main_async.py -i media/sample2.mp4 -image_dir images -is_export_csv=True
```

## Documentation

### Documentatiob of used models

1. [Person Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html)
2. [Person Re-identification Model](https://docs.openvinotoolkit.org/latest/_models_intel_person_reidentification_retail_0300_description_person_reidentification_retail_0300.html)
3. [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_retail_0005_description_face_detection_retail_0005.html)
4. [Age and Gender Recognition Model](https://docs.openvinotoolkit.org/latest/_models_intel_age_gender_recognition_retail_0013_description_age_gender_recognition_retail_0013.html)

### Command Line Arguments for Running the app

Following are commanda line arguments that can use for while running the main.py file ` python main.py `:-

  1. -h                : Get the information about all the command line arguments
  2. -i     (required) : Specify the path of input video file or enter cam for taking input video from webcam
  3. -d     (optional) : Specify the target device to infer the video file on the model. Suppoerted devices are: CPU, GPU,                            FPGA (For running on FPGA used HETERO:FPGA,CPU), MYRIAD(For VPU).
  4. -l     (optional) : Specify the absolute path of cpu extension if some layers of models are not supported on the device.
  5. -pt  (optional) : Specify the probability threshold for person detection and face detection model to detect the face accurately from video frame.
  6. -image_dir    (optional) : Specify the path of a directory if you want to save detected person's image.
  7. -is_export_csv (optional) : Specify False if you don't want to export the csv file of the predicted data and default value is true.
  


