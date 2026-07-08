# MIME (Multiple Interactions Made Easy)

## Description
MIME (Multiple Interactions Made Easy) is one of the largest and most diverse robotic demonstration datasets available. It was created to advance research in Imitation Learning in robotics, especially for complex and multi-task manipulation tasks. The dataset contains demonstrations of 20 distinct robotic tasks, ranging from simple actions such as pushing to more difficult tasks such as stacking household objects and opening bottles. Each data point includes human demonstrations (HD) and robotic demonstrations (RD) collected with a Baxter robot. The human demonstrations are captured with a head-mounted Kinect (RGBD), while the robotic demonstrations include data from an overhead-mounted Kinect (RGBD), two soft kinetic sensors mounted on the wrists (RGBD), and joint-angle data from the Baxter. The dataset was introduced in 2018.

## Statistics
- **Samples:** 8260 human-robot demonstrations.
- **Tasks:** 20 distinct robotic tasks.
- **Version:** Single version, introduced in the CoRL 2018 paper.
- **Size:** The total size of the dataset was not explicitly provided on the official page or in the paper, but the nature of the data (RGBD videos and kinesthetic data) suggests a considerable size (likely on the order of hundreds of GB or TB).

## Features
- **Task Diversity:** Contains demonstrations for 20 different robotic tasks, spanning a wide range of manipulations.
- **Human and Robotic Demonstrations:** Includes both demonstrations of humans performing the tasks and demonstrations by the Baxter robot.
- **Multimodal Data:** Each data point is rich in information, including RGBD data from multiple cameras (human head, robot overhead view, robot wrists) and kinesthetic data (Baxter robot joint angles).
- **Focus on Multi-task Imitation Learning:** Designed to enable the training of imitation policies that can generalize across multiple tasks.

## Use Cases
- **Multi-task Imitation Learning:** Training robotic agents to perform a wide variety of tasks from demonstrations.
- **Visual Representation Learning:** Use of RGBD data to learn robust visual representations for robotic manipulation.
- **Trajectory Prediction:** Evaluation of models that predict robotic trajectories based on demonstrations.
- **Robotic Skill Learning:** Development of control policies for robots such as the Baxter.

## Integration
The dataset can be downloaded via a Dropbox link provided on the official project page. The download can be done for the complete dataset or by individual task.
**Main Download Link:** `https://www.dropbox.com/scl/fo/cgvsmxayv6qqw1ynvxrfr/AFnncxm2YUHp86AI1mm0rSI?rlkey=iqeeazs9bfnx283zt4a3vg4cc&e=1&dl=0`
**Usage Instructions:** The dataset is typically used to train imitation learning models. The kinesthetic data (joint angles) and the visual data (RGBD) are used to map observations to robot actions. The original paper ("Multiple Interactions Made Easy (MIME): Large Scale Demonstrations Data for Imitation") should be consulted for details on the data structure and processing methods.

## URL
[https://sites.google.com/view/mimedataset](https://sites.google.com/view/mimedataset)
