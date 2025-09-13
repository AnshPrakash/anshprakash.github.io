---
layout: distill
title: MimicPlay on Franka Arm and its Extension
description: This blog is part of our university’s project lab, where we are working on replicating MimicPlay using a real one-arm robotic platform in our lab. Building on this setup, we aim to extend the approach to bi-manual systems such as the Tiago robot. Our work explores how abundant human play data can be leveraged to guide efficient low-level robot policies.
tags: Imitation-Learning, Learning-from-Human, Long-Horizon-Manipulation, pearl-lab
date: 2025-09-08
citation: true
related_publications: true
related_posts: false
giscus_comments: false


authors:
  - name: Ansh Prakash
    url: "https://github.com/AnshPrakash"
    affiliations:
      name: TU Darmstadt
  - name: Xiaoqi Zhou
    url: "https://github.com/Xiaoqi-Z7"
    affiliations:
      name: TU Darmstadt

bibliography: 2025-09-08-mimicplay.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Introduction
    subsections:
      - name: Behavioural Cloning
  - name: Related Works
  - name: MimicPlay
    subsections:
      - name: High Level Latent Planner
      - name: Low Level Robot Policy
  - name: Franka Teleoperation system
  - name: Data Collection Pipeline
    subsections:
      - name: Human Play data
        subsections:
          - name: Hand Tracking
            subsections:
              - name: ABC
              - name: XYZ
          - name: Miscellaneous
      - name: Low level Teleoperation Data
        subsection:
          - name: Sampler
          - name: robomimimc style data format
  - name: High Level Latent Planner
    subsections:
      - name: Model
      - name: Latent space
      - name: Multi-modality
      - name: Training
  - name: Low Level Policy
    subsections:
      - name: Model
      - name: Observation Space(Inputs)
      - name: Action Space(Outputs)
  - name: Differences in the original Setup and our Setup
    subsecctions:
      - name: Cameras
      - name: Environment
  - name: Experiments
    subsections:
      - name: High Level Planner
      - name: Low Level Planner
  - name: Extension to Bimanual Tiago
    subsections:
      - name: Update to Hand Tracking system to two hands
      - name: High Level Planner
      - name: Low level Robot Policy update
        subsections:
          - name: Model level update
          - name: Teleoperation system
  - name: Conclusion
  - name: Acknowledgements

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---
## Introduction

Teaching robots to carry out general-purpose manipulation tasks efficiently has been a long-standing challenge. Recent advances in Imitation Learning (IL) have made notable progress toward this objective, particularly through supervised training with human teleoperation demonstrations or expert policy trajectories <d-cite key="pomerleau1988alvinn"> </d-cite> <d-cite key="zhang2018deep"> </d-cite> .
Although promising, imitation learning has mostly been restricted to short-horizon skills, as collecting demonstrations for long-horizon, real-world tasks is time-consuming and labor-intensive.
Two connected directions have emerged in recent literature to scale up imitation learning to complex
long-horizon tasks: *hierarchical imitation learning* and *learning from play data*.
1. **Hierarchical imitation learning** improves sample efficiency by breaking down end-to-end deep imitation learning into two stages: learning high-level planners and low-level visuomotor controllers  <d-cite key="mandlekar2020learning"> </d-cite> <d-cite key="shiarlis2018taco"> </d-cite> .

2. **Learning from play data** uses a different type of robot training data known as play data <d-cite key="lynch2020play"> </d-cite>, which is collected via human-operated robots exploring their environment without explicit task instructions. Such data captures more diverse behaviors and situations than task-specific demonstrations <d-cite key="lynch2020play"> </d-cite> <d-cite key="cui2022play"> </d-cite>. Methods that leverage play data typically train hierarchical policies, where the high-level planner models intent and the low-level controllers handle goal-directed actions <d-cite key="lynch2020play"> </d-cite>. Nonetheless, collecting real-world play data is resource-intensive; for instance, C-BeT <d-cite key="cui2022play"> </d-cite> requires 4.5 hours of play data for manipulation skills in one scene, while TACO-RL <d-cite key="rosete2022latent"> </d-cite> needs 6 hours for a single 3D tabletop environment.


<!-- Ease of collecting human play data -->
<div class="row mt-3">
    <div class="col-sm text-center" id="fig:humanplay-collection" >
        {% include figure.liquid loading="eager" path="assets/img/mimicplay/scale_data.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig1: Humans can complete long-horizon tasks much faster than teleoperated robots. Inspired by this, MIMICPLAY<d-cite key="wang2023mimicplaylonghorizonimitationlearning"></d-cite> is implemented as a hierarchical imitation learning framework that learns high-level planning from inexpensive human play data and low-level control policies from a small set of multi-task teleoperated robot demonstrations.
</div>


### MimicPlay

MimicPlay <d-cite key="wang2023mimicplaylonghorizonimitationlearning"></d-cite> suggests that data for learning both high-level planning and low-level control can take various forms, potentially lowering the cost of imitation learning for complex, long-horizon tasks.

Building on this idea, the authors propose a learning paradigm where robots acquire high-level plans from human play data, in which humans freely interact with the environment using their hands. This type of data is faster and easier to gather than robot teleoperation data, enabling large-scale collection that captures a wide range of behaviors and scenarios <a href="#fig:humanplay-collection">Fig 1</a>.

Subsequently, the robot learns low-level manipulation policies from a limited set of demonstrations collected via human teleoperation. While demonstration data is more expensive to obtain, it avoids the challenges arising from differences between human and robot embodiments.


<div class="row mt-3">
    <div class="col-sm text-center">
        {% include figure.liquid loading="eager" path="assets/img/mimicplay/mimic-play-inspiration.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>    

<div class="row mt-3">
  <div class="col-sm text-center">
      {% include figure.liquid loading="eager" path="assets/img/mimicplay/mimicplay-fillgap.png" class="img-fluid rounded z-depth-1" zoomable=true %}
  </div>
</div>

<div class="caption">
    Human is able to complete a long-horizon task much faster than a teleoperated robot. This observation is the inspiration for MimicPlay, a hierarchical imitation learning algorithm that learns a high-level planner from cheap human play data and a low-level control policy from a small amount of multi-task teleoperated robot demonstrations.
</div>




---

## Related Works


---

## Franka Teleoperation system

We developed our own teleoperation system to collect low-level demonstration data. Using a Meta Quest VR controller, we operated the Panda arm, with the headset tracking the controller’s pose in real time. The pose differences from the controller were transformed into corresponding end-effector movements on the robot, enabling us to perform various pick-and-place tasks.

We used a Cartesian impedance controller for safer operation and additionally calibrated gravity compensation for a different gripper. This ensures that the end-effector neither drops nor unintentionally lifts depending on the load. Instructions for calibration can be found [here](https://github.com/nbfigueroa/franka_interactive_controllers/blob/main/doc/instructions/external_tool_compensation.md).

Here is the code for teleoperation: [![GitHub Repo](https://img.shields.io/badge/GitHub-Franka--Teleop-blue?logo=github)](https://github.com/AnshPrakash/franka_teleop)


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid path="assets/video/mimicplay/teleop_demo.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include video.liquid path="assets/video/mimicplay/teleop_demo_front.mp4" class="img-fluid rounded z-depth-1" controls=true %}
    </div>
</div>
<div class="caption">
    Here is a video of Teleoperation system in action
</div>

---

## Data Collection Pipeline


### Human Play data

We store the human play data in **mp4 format** with a frame rate of **20 FPS**. Afterwards, we apply some **post-processing** to convert it into the required **robomimic format**.

1. **Hand detection**  
   We use a pretrained hand detection model[![GitHub Repo](https://img.shields.io/badge/GitHub-handobj-blue?logo=github)](https://github.com/ddshan/hand_object_detector) to locate human hands in the video frames. In total, we collected **10 demonstrations**. After filtering, we discarded several demos where the hands could not be reliably detected.

2. **3D triangulation and dataset conversion**  
   Using the **calibrated stereo camera setup** (two synchronized viewpoints), we triangulate the detected hand positions to obtain their **3D coordinates in the world frame**. These 3D hand trajectories are then converted into the **robomimic dataset format**.

Additionaly, we also do a **Projection validation (visualization check)** To verify the correctness of the calibration, we re-projected the obtained 3D points back to the image plane and visually inspected their alignment with the detected 2D hand positions. This ensured that the existed **camera parameters** were consistent with the real-world coordinate system. Below is the detection code used for this visualization check:

```python
out_dir = "buffer/Slow_version_Human_prompts_0"
os.makedirs(out_dir, exist_ok=True)
# --- Load HDF5 ---
hdf5_path = "/home/xiaoqi/MimicPlay/mimicplay/datasets/playdata/Slow_version_Human_prompts/demo_0_new.hdf5"   # update with your file path
with h5py.File(hdf5_path, "r") as f:
    # Extract robot0 end-effector positions (605, 1, 3)
    eef_pos = f["data/demo_0/obs/robot0_eef_pos"][:]  # shape (605,1,3)
    eef_pos = eef_pos.squeeze(axis=1)  # now (605, 3)

    # Extract images if needed
    agentview_img = f["data/demo_0/obs/agentview_image"][:] 
    agentview_img2 = f["data/demo_0/obs/agentview_image_2"][:] 

# --- Save raw 3D positions ---
np.savetxt(os.path.join(out_dir, "robot0_eef_pos.txt"), eef_pos, fmt="%.6f")

ZEDA_LEFT_CAM = CameraModel(
    fx=1059.9764404296875,
    fy=1059.9764404296875,
    cx=963.07568359375,
    cy=522.3530883789062,
    R_wc=R.from_quat([-0.404974467935380, -0.808551385290863, 0.425767747250020, 0.031018753461827]).as_matrix(),
    t_wc=np.array([0.903701253331141, 0.444249176547482, 0.598645500102408])
)

ZEDB_RIGHT_CAM = CameraModel(
    fx=1060.0899658203125,
    fy=1059.0899658203125,
    cx=958.9099731445312,
    cy=561.5670166015625,
    R_wc=R.from_quat([0.81395177, -0.40028226, -0.07631803, -0.41404371]).as_matrix(),
    t_wc=np.array([0.11261126, -0.52195948, 0.55795671])
)

# scale factor from 1920x1080 -> 640x360
sx = 640.0 / 1920.0   # = 1/3
sy = 360.0 / 1080.0   # = 1/3

ZEDA_LEFT_CAM  = ZEDA_LEFT_CAM.scaled(sx, sy)
ZEDB_RIGHT_CAM = ZEDB_RIGHT_CAM.scaled(sx, sy)


# --- Project and overlay ---
left_count, right_count = 0, 0         
both_count, none_count = 0, 0          

for i, (pos, img1, img2) in enumerate(tqdm(zip(eef_pos, agentview_img, agentview_img2), total=len(eef_pos))):
    uv1 = ZEDA_LEFT_CAM.project_point(pos).astype(int)
    uv2 = ZEDB_RIGHT_CAM.project_point(pos).astype(int)

    img1_draw = img1.copy()
    img2_draw = img2.copy()

    inside1, inside2 = False, False

    if 0 <= uv1[0] < img1_draw.shape[1] and 0 <= uv1[1] < img1_draw.shape[0]:
        cv2.circle(img1_draw, (uv1[0], uv1[1]), radius=5, color=(0, 255, 0), thickness=-1)
        inside1 = True
        left_count += 1

    if 0 <= uv2[0] < img2_draw.shape[1] and 0 <= uv2[1] < img2_draw.shape[0]:
        cv2.circle(img2_draw, (uv2[0], uv2[1]), radius=5, color=(0, 255, 0), thickness=-1)
        inside2 = True
        right_count += 1

    # wrap
    if inside1 and inside2:
        both_count += 1
    elif not inside1 and not inside2:
        none_count += 1

    out1 = os.path.join(out_dir, f"agentview1_{i:04d}.png")
    out2 = os.path.join(out_dir, f"agentview2_{i:04d}.png")

    cv2.imwrite(out1, cv2.cvtColor(img1_draw, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out2, cv2.cvtColor(img2_draw, cv2.COLOR_RGB2BGR))

    print(f"[Frame {i}] saved → {out1}, {out2} | inside1={inside1}, inside2={inside2}")

# statistic results
print("========== check and statistical results ==========")
print(f"left detecting: {left_count}")
print(f"right detecting: {right_count}")
print(f"both detecting: {both_count}")
print(f"both not detecting: {none_count}")
print(f"total numbers:   {len(eef_pos)}")
print(f"Saved projections and images in '{out_dir}/'")
```



### Low level Teleoperation Data

We record rosbag from various topics. Here is the list of topics we record. However, this will need further post-processing because all the topics are published at different frequncies.

```
topics:
  - /franka_state_controller/franka_states
  - /franka_gripper/joint_states
  - /franka_state_controller/joint_states_desired
  - /franka_state_controller/O_T_EE
  - /franka_state_controller/joint_states
  - /cartesian_impedance_controller/desired_pose
  - /zedA/zed_node_A/left/image_rect_color 
  - /zedB/zed_node_B/left/image_rect_color 
```

We first estimated the frequencies of all the topics and then used our sampling algorithm to resample at a fixed frequency, corresponding to the rate at which we want our policy controller to operate.

<!-- Pre-processed frequencies -->
<div class="row mt-3">
    <div class="col-sm text-center">
        <strong>Before Sampling</strong>
        {% include figure.liquid loading="eager" path="assets/img/preprocessed_freq/cartesian_impedance_controller_desired_pose_hist.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <p>/cartesian_impedance_controller/desired_pose @ 50Hz</p>
    </div>
    <div class="col-sm text-center">
        <strong>Before Sampling</strong>
        {% include figure.liquid loading="eager" path="assets/img/preprocessed_freq/franka_state_controller_O_T_EE_hist.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <p>/franka_state_controller/O_T_EE @ 607Hz</p>
    </div>
</div>

<!-- Post-processed frequencies -->
<div class="row mt-4">
    <div class="col-sm text-center">
        <strong>After Sampling</strong>
        {% include figure.liquid loading="eager" path="assets/img/postprocessed_freq/cartesian_impedance_controller_desired_pose_hist.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <p>/cartesian_impedance_controller/desired_pose @ 13Hz</p>
    </div>
    <div class="col-sm text-center">
        <strong>After Sampling</strong>
        {% include figure.liquid loading="eager" path="assets/img/postprocessed_freq/franka_state_controller_O_T_EE_hist.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <p>/franka_state_controller/O_T_EE @ 13Hz</p>
    </div>
</div>

<div class="caption mt-2 text-center">
    Frequencies of both topics are aligned after applying our sampling algorithm, from highly different original rates (50Hz vs 607Hz) to a unified 13Hz (hyperparameter).
</div>


**Here is the pseudo code for our sampling algorithm which ensures equal observations from all topics:**

```python

# Synchronize multiple topics to a target frequency

start_time = min_timestamp(topics)
end_time   = max_timestamp(topics)

dt = 1 / target_freq
t  = start_time

while t <= end_time:
    for topic in topics:
        msg = select_message(topic, timestamp <= t) # the msg from the topic which has the greatest timestamp, but timestamp is <= t
        topic_buffer[topic] = msg

    combined_msgs = [topic_buffer[topic] for topic in topics]
    t += dt
```

You can find the sampler package here.[![GitHub Repo](https://img.shields.io/badge/GitHub-Sampler-blue?logo=github)](https://github.com/AnshPrakash/MimicPlay/tree/main/sampler)

Further, we transform the data into robomimic style hdf5 format [![GitHub Repo](https://img.shields.io/badge/GitHub-rosbag2hdf5-blue?logo=github)](https://github.com/AnshPrakash/MimicPlay/tree/main/rosbag2hdf5)


> The final teleoperation dataset, formatted in **robomimic style**, is now ready to be used in the training pipeline.



```
FILE_CONTENTS {
 group      /
 group      /data
 group      /data/demo_0
 dataset    /data/demo_0/actions
 group      /data/demo_0/obs
 dataset    /data/demo_0/obs/O_T_EE
 dataset    /data/demo_0/obs/back_camera
 dataset    /data/demo_0/obs/ee_pose
 dataset    /data/demo_0/obs/front_camera
 dataset    /data/demo_0/obs/gripper_joint_states
 dataset    /data/demo_0/obs/joint_states
 dataset    /data/demo_0/obs/joint_states_desired
 group      /data/demo_1
 dataset    /data/demo_1/actions
 group      /data/demo_1/obs
 dataset    /data/demo_1/obs/O_T_EE
 dataset    /data/demo_1/obs/back_camera
 dataset    /data/demo_1/obs/ee_pose
 dataset    /data/demo_1/obs/front_camera
 dataset    /data/demo_1/obs/gripper_joint_states
 dataset    /data/demo_1/obs/joint_states
 dataset    /data/demo_1/obs/joint_states_desired
 group      /mask
 dataset    /mask/train
}
```

---

<!-- Training process -->
<div class="row mt-3">
    <div class="col-sm text-left">
        <strong>Method</strong>
        {% include figure.liquid loading="eager" path="assets/img/mimicplay/training.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="caption mt-2 text-center">
    Overview of MimicPlay <d-cite key="wang2023mimicplaylonghorizonimitationlearning"></d-cite>
</div>

## High Level Latent Planner
### Model
With the collected human play data and the corresponding 3D hand trajectories \( \tau \), we formalize the latent plan learning problem as a **goal-conditioned 3D trajectory generation task**. In this formulation, the planner must generate feasible hand trajectories conditioned on the specified goal state.  

To model this distribution, we adopt a **Gaussian Mixture Model (GMM)** as the high-level planner. The GMM captures the multi-modal nature of human demonstrations, where multiple valid trajectories may exist for achieving the same goal. This provides several advantages:

- **Goal-conditioning**: ensures that the generated trajectory is consistent with the task objective.  
- **Flexibility**: supports multiple valid solutions instead of collapsing to a single mode.  
- **Robustness across tasks**: enables the planner to generalize across diverse demonstrations collected from different tasks.  

In summary, the GMM-based planner learns to represent the distribution of goal-conditioned trajectories, which allows for generating diverse yet feasible high-level plans.

### Latent plan
Our high-level planner is formulated as a **latent plan generator**.  
We use a pretrained **GMM model** to produce latent trajectory plans from the collected demonstrations.  
These latent plans are not directly executed by the robot but are instead passed to the **low-level controller**, which converts them into executable motor commands.  
This hierarchical setup defines the high-level component as a latent plan rather than direct control.

### Multi-modality
The training model takes **multi-modal inputs** to construct the high-level planner.  
Specifically, it receives **two-view RGB images** together with the corresponding **hand position information** as inputs, and outputs a **GMM trajectory distribution**.  
This setup allows the model to learn from both visual context and motion data when generating latent plans.

### Training

#### Setup

For the collected demonstration dataset, we used **one demo as the validation set**, while the remaining demos were used for **training**. The training was conducted following the **configuration provided in the reference paper**.
For hyperparameters, we mainly relied on the **default settings from the official repository**, while performing **additional tuning** based on our own dataset to improve performance, e.g. "goal image range" and "std".

#### Evaluation
We evaluated the high-level planner using two metrics:

1. **GMM likelihood probability (training phase)**  
   During training, we monitored the **likelihood of the ground-truth data under the learned GMM model**. This serves as a measure of how well the model captures the distribution of the demonstrations.

2. **Distance error (test phase)**  
   On the test prompts, we computed the **distance error** between the predicted trajectories and the ground-truth hand positions. Since our high-level planner is a **probabilistic model**, we performed **multiple samples for each time step** in the sequence. The final error metric was obtained by averaging across the entire video sequence and across all samples.





## Low Level Policy


During **training**, the low-level policy receives a latent embedding of the robot’s trajectory from the high-level latent planner. This embedding provides rich contextual information, significantly reducing the need for large amounts of teleoperation data.

Additionally, we used `negative log likelihood` loss for training the models.

<!-- Training low level planner -->
<div class="row mt-3">
    <div class="col-sm text-center">
        {% include figure.liquid loading="eager" path="assets/img/mimicplay/low-level-training.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <p>Loss curve for training low-level policy</p>
    </div>
</div>

<div class="caption mt-2 text-center">
    Divergence between validation and training loss after epoch 16 occurs due to the poor performance of the high-level planner, which was unable to generalize well across similar trajectories.
</div>



During **testing**, the low-level policy instead receives a latent embedding of the human trajectory. This acts as a *human prompt*, guiding the robot to replicate the demonstrated actions. At the same time, the policy continuously collects observations from onboard cameras and proprioceptive signals (via ROS topics) at the desired frequency.

Below is the pseudocode illustrating how the system acquires observations at a fixed frequency in the real robot setup:


```python

# Get observations at a desired frequency

# 1. Compute how long we should wait between observations
dt = 1 / target_frequency

while not shutting_down():
    # 2. Wait until *all* topics have fresh data newer than last_obs_time + dt
    if all_topics_ready(threshold_time=last_obs_time + dt):
        
        # 3. Snapshot the latest messages and timestamps
        msgs, times = snapshot_latest_messages()

        # 4. Convert each message into a NumPy-friendly format
        data = {topic: convert_to_numpy(msgs[topic]) for topic in msgs}

        # 5. Update last observation time and return a dictionary
        last_obs_time = min(times.values())
        return {
            "timestamp": last_obs_time,
            "data": data,
            "times": times,
        }

    # 6. Otherwise, wait briefly and try again
    sleep_a_bit()
```

> Actual code for reference here [![GitHub Repo](https://img.shields.io/badge/GitHub-PolicyController-blue?logo=github)](https://github.com/AnshPrakash/franka_teleop/blob/b088a9c38e2cb60ba15d4b1b7c3e7edeb2698313/scripts/policy_controller.py#L345)


In the original paper, the robot policy operated at 17 Hz. However, our ZED camera could capture observations at a maximum frequency of 14 Hz, which set the upper bound for our deployed policy. Ultimately, we chose to run the robot policy at 13 Hz.


<!-- Low level policy -->
<div class="row mt-3">
    <div class="col-sm text-center">
        {% include figure.liquid loading="eager" path="assets/img/mimicplay/low-level-policy.drawio.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="caption mt-2 text-center">
    The low-level policy receives a latent plan, image observations, and proprioceptive inputs, then samples an action from a multimodal Gaussian distribution. <d-cite key="wang2023mimicplaylonghorizonimitationlearning"></d-cite>
</div>


# Differences in the original Setup and our Setup



## Experiments

### High Level Planner

After completing the training of the high-level latent planner, we first collected **video prompts** and performed a **visual inspection of the predicted trajectories**. This step allowed us to qualitatively evaluate whether the generated trajectories aligned with the expected task goals and to compare them against the ground-truth trajectories from the demonstrations. Below we show example visualizations of the predicted trajectories。

<div class="row mt-4">
    <div class="col-sm text-center">
        <!-- <strong>After Sampling</strong> -->
        {% include figure.liquid loading="eager" path="assets/img/high_level/single_view/start_with_traj.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <p>current states of hand with 10 steps future trajectory</p>
    </div>
    <div class="col-sm text-center">
        <!-- <strong>After Sampling</strong> -->
        {% include figure.liquid loading="eager" path="assets/img/high_level/single_view/goal.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <p>goal states of hand</p>
    </div>
</div>

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include video.liquid path="assets/video/high_level/single_view/traj_video.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true %}
  </div>
</div>
<div class="caption text-center">
  trajectory through time steps
</div>


### Low-Level Policy — Policy Controller (Live System)

[![GitHub Repo](https://img.shields.io/badge/GitHub-PolicyController-blue?logo=github)](https://github.com/AnshPrakash/franka_teleop/blob/robot-policy/scripts/policy_controller.py)

Below we present our evaluation results for the low-level policy. Although the success rate was 0%, we have developed a solid understanding of the underlying reasons for this outcome.

Here is our evaluation video results:


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
      <h5>Human Prompts</h5>
        {% include video.liquid path="assets/video/mimicplay/Human_prompts/data-2025-09-06_10-56-20/zedA_zed_node_A_left_image_rect_color.mp4" class="img-fluid rounded z-depth-1" controls=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
       <h5>Robot Policy Acting</h5>
        {% include video.liquid path="assets/video/mimicplay/lowlevel-eval-policy_evaluation/robot-policy-eval-recordings/demo_0/data-2025-09-07_16-11-12/zedA_zed_node_A_left_image_rect_color.mp4" class="img-fluid rounded z-depth-1" controls=true %}
    </div>
</div>

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
      {% include video.liquid path="assets/video/mimicplay/Human_prompts/data-2025-09-06_10-58-04/zedA_zed_node_A_left_image_rect_color.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
      {% include video.liquid path="assets/video/mimicplay/lowlevel-eval-policy_evaluation/robot-policy-eval-recordings/demo_2/data-2025-09-07_16-29-01/zedA_zed_node_A_left_image_rect_color.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0">
      {% include video.liquid path="assets/video/mimicplay/Human_prompts/data-2025-09-07_14-51-35_demo3/zedA_zed_node_A_left_image_rect_color.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
  <div class="col-sm mt-3 mt-md-0">
      {% include video.liquid path="assets/video/mimicplay/lowlevel-eval-policy_evaluation/robot-policy-eval-recordings/demo_3/data-2025-09-07_15-51-07/zedA_zed_node_A_left_image_rect_color.mp4" class="img-fluid rounded z-depth-1" controls=true %}
  </div>
</div>



<div class="caption">
    Left: Human prompts, Right: Robot policy acting
</div>


### Key Limitations Observed

1. **High-Level Planner — Poor Embedding Quality**

   * We found that the high-level planner produced **high prediction errors** for trajectories, which resulted in **poor latent embeddings**.
   * Through hyperparameter tuning, we discovered that our dataset required **fewer modes** for accurate trajectory prediction.
   * Due to these weak embeddings, the low-level policy experienced **high variance between similar trajectories**, preventing it from fully leveraging the advantages of human guidance.

2. **Absence of Wrist Camera**

   * There was a significant **distribution shift** between training and evaluation image inputs from the front and back cameras.
   * The original authors used a **wrist-mounted camera**, which helped stabilize the robot policy.
   * Adding a wrist camera in our setup would likely **reduce distribution shift** and improve performance—**provided that a robust latent embedding of the human prompt is available**.



---

## Extension to Bimanual Tiago — Future Work

### Update to Hand Tracking system to two hands

The current pretrained hand detection model is able to distinguish between the **left and right hands**. However, since our setup only uses **two calibrated camera views**, the detection results can vary significantly. One major challenge arises when the **two hands occlude each other**, in which case it may be impossible to reliably observe both hands in both camera views at the same time. This directly limits our ability to obtain accurate **3D hand position estimates** through triangulation.

To address this issue, one potential approach we are exploring is **temporal interpolation**. Specifically, when a hand temporarily disappears due to occlusion, we use its **2D infomation before and after the disappearance** to interpolate the missing frames. By filling in these occluded intervals, we aim to maintain more consistent 3D hand trajectory estimation for bimanual tasks.

We can use a Kalman filter to estimate the position of the occluded part by modeling the trajectory of the hand with a simple linear dynamics model.


### High-level planner & Low-level planner - Bimanual

Only minor changes to the model are required to enable it for a bimanual scenario. Specifically, the action dimension needs to be doubled to account for the additional arm, and more observations must be added to track the positions of both end-effectors. The more challenging aspect lies in fine-tuning hyperparameters—such as the number of modes in the GMM decoder of the high-level planner—since data multimodality increases with two arms.

---

# Conclusion



---

## Acknowledgements

We would like to thank our supervisor, [Franziska Herbert](https://pearl-lab.com/people/franziska-herbert/), for her guidance and support throughout this project. We also extend our gratitude to the course organizer and the lab staff for providing the resources and assistance that made this work possible. Finally, we thank the authors of [**MimicPlay**](https://mimic-play.github.io/) for making their code publicly available.

---



### BibTeX

```bibtex
@misc{prakashzhou2025mimicplay,
  author       = {Prakash, Ansh and Zhou, Xiaoqi},
  title        = {MimicPlay on Franka Arm and its Extension},
  year         = {2025},
  howpublished = {\url{https://anshprakash.github.io/blog/2025/mimicplay/}},
  note         = {IROBMAN Lab Blog}
}
```

---