Object-Tracking-with-YOLO
=
Simple Realtime Person Tracking with YOLOv3, YOLOv4 as Object
Detection and MOSSE, KCF, CSRT as Object Tracking

### Performance:
#### Parallel Thread - GTX 1650 - Resolution: 640x480
- #### YOLOv3, YOLOv4: 9fps
- #### YOLOv3-Tiny, YOLOv4-Tiny: 15fps
- #### MOSSE: 180fps
- #### KCF: 20fps
- #### CSRT: 12fps

## Requirements
- #### Anaconda
- #### Tensorflow 2.3.0 with CUDA 10.1 (Not Supported for Tensorflow 2.4)

## How To Run:
- #### Get Git Project:
    ```bash
    git clone https://github.com/Windrist/Object-Tracking-with-YOLO
    conda env create OTRT
    conda activate OTRT
    conda install cudatoolkit=10.1
    pip install tensorflow-gpu==2.3.0
    cd Object-Tracking-with-YOLOv3
    ```

- #### Download Datasets and put on Main Folder: [Datasets](https://drive.google.com/drive/folders/19EVsAOLwDqWoZrrB9j4sU9Kv1qVrcdri)
- #### Change Configurations on configs.py File, Example:
    ```bash
    TRACKER_TYPE = "KCF"  # MOSSE or KCF or CSRT
    YOLO_TYPE = "YOLO-v4"  # YOLO-v4 or YOLO-v3
    YOLO_FOCUS_OBJECT = "person"
    TRAIN_YOLO_TINY = False
    ```

- #### Run Demo with Webcam:
    ```bash
    python main.py
    ```

- #### Check Output Video on Output Folder!


## Credits:
* #### YOLO Model by PyLessons
* #### Link Repo: [Github](https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3)
* #### Link Website: https://pylessons.com/

