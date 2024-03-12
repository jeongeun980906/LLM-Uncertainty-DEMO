## DEMO CODE for CLARA

**This code is official implement of the paper CLARA for the real-world demonstraions, with UR5e and RG2 gripper**

For details, please check [Project Page](https://clararobot.github.io/) or [Paper Page](https://ieeexplore.ieee.org/document/10336901)


Config 1
<figure class="video_container">
 <video controls="true" allowfullscreen="true">
    <source src="./figs/ex1.mp4" type="video/mp4">
 </video>
</figure>

Config 2
<figure class="video_container">
 <video controls="true" allowfullscreen="true">
    <source src="./figs/ex2.mp4" type="video/mp4">
 </video>
</figure>

## Code
|No.|Contents|Details| Parameters|
| -------- | ------- |  ------- | ------- |
|1|타겟 보드에서 사람-로봇 사이의 대화를 통한 언어의 불확실성 해소후 로봇 작업 수행 데모 및 패키지(SW)| Overall Package |  
|2| 언어모델 기반 로봇의 작업 추론 모듈| [Code](./llm-uncertainty/llm/chat.py) Line 73  def inference(self):| returns: pick object, give person
|3| 언어의 불확실성 추정 모듈 |  [Code](./llm-uncertainty/llm/chat.py) Line 35  def plan_with_unct(self): | returns: robot plans, scores for each plans, uncertainty
|4| 사람에게 불확실성의 이유를 설명하고, 모르는 것을 되묻는 모듈 | [Code](./llm-uncertainty/llm/chat.py) Line 218  def question_generation(self):| returns: uncertainty reason, user question, feasibility|
|5| 언어모델이 추론한 로봇작업을 실제 로봇의 경로로 바꾸는 모듈 |  [Code](./llm-uncertainty/model/robot_client.py) Line 188 def pick(self, p_target), line 231 def give(self, user_id)| input: [x,y,z] of pick object, or id (0,1,2,3) of give human
|6| 사람과 로봇이 대화 할 수 있는 인터페이스 | [Code](./llm-uncertainty/llm/gui.py) | python class for visualizer


## INSTALL
First, please set your own openai API key in 
```
llm-uncertainty/key/key.txt
```

Install pytorch in the python environment
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
But for the ORIN, please check out [docs](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) to install pytorch

```
pip install openai
pip install spacy
python -m spacy download en_core_web_lg
pip install scipy==1.12.0
pip install ultralytics
```
For detailed installation in ORIN, please check [Installation](
https://forums.developer.nvidia.com/t/running-yolov8-on-jetson-orin-nano-developer-kit/271414/10) document for the details. 

## RUN
#### Run DEMO without the robot or camera. 
First, you need to put the inference image in the path
```
./data/det/ori_[index].jpg
``` 
for tabletop image, and 
```
./data/det/human_ori_[index].jpg
``` 
for user images. The sample for index 1, 2 is provided.

Then, run the code
```
cd llm-uncertainty
python3 demo.py --move_robot 0 --index [n]
```
**REMEBER: Press !!Enter key!! after you type in goal or answers!**

#### If you want to use REALSENSE CAM + OWL-VIT, then run
OWL-VIT dependencies, If you are using small devices, please set --on_board 1 in perception engine. This will run YOLOv8 instead of OWL-ViT. 
```
pip install transformers
pip install mujoco-python-viewer
pip install mujoco
pip install pymodbus
```
Please download the owl-vit checkpoints if you are using those from [hugging_face](https://huggingface.co/google/owlvit-base-patch32). 
```
llm-uncertainty/owl_vit_model
```

launch the robot node (UR5e) (For the installation, please follow [UR-ROS2](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver))
```
ros2 launch rilab_ur_launch ex-ur5e.launch.py ur_type:=ur5e robot_ip:=[robot_ip] launch_rviz:=false reverse_ip:=[your_ip]

```
Launch realsense node. (For the installation, please follow [Realsense](https://github.com/IntelRealSense/realsense-ros))
```
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
```
Then excecute the robot code
```
cd llm-uncertainty
python3 perception.py --index [n] --on_board [1 or 0]
python3 demo.py --index [n]
```
**REMEBER: Press !!Enter key!! after you type in goal or answers!**
