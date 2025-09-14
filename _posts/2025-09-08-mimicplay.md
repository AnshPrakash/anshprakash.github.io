---
layout: distill
title: MimicPlay on Franka Arm and its Extension
description: This blog is part of our university’s project lab, where we are working on replicating MimicPlay using a real one-arm robotic platform in our lab. Building on this setup, we aim to extend the approach to bi-manual systems such as the Tiago robot. Our work explores how abundant human play data can be leveraged to guide efficient low-level robot policies.
tags: Imitation-Learning, Learning-from-Human, Long-Horizon-Manipulation, pearl-lab
date: 2025-09-14
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
  - name: Franziska Herbert
    url: https://pearl-lab.com/people/franziska-herbert/
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
  - name: Related Works
  - name: MimicPlay
  - name: Implementation
    subsections:
    - name: Comparison Between Original and Our Setup
    - name: Franka Teleoperation system
    - name: Data Collection Pipeline
      subsections:
        - name: Human Play data
        - name: Low level Teleoperation Data
  - name: Training
    subsections:
      - name: High Level Latent Planner
      - name: Low Level Policy
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
    Fig1<d-cite key="wang2023mimicplaylonghorizonimitationlearning"></d-cite>: Humans can complete long-horizon tasks much faster than teleoperated robots. Inspired by this, MIMICPLAY<d-cite key="wang2023mimicplaylonghorizonimitationlearning"></d-cite> is implemented as a hierarchical imitation learning framework that learns high-level planning from inexpensive human play data and low-level control policies from a small set of multi-task teleoperated robot demonstrations.
</div>


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

**Imitation learning from demonstrations**: Imitation Learning (IL) enables robots to perform various manipulation tasks <d-cite key="calinon2010learning"></d-cite>, <d-cite key="ijspeert2002movement"></d-cite> , <d-cite key="schaal1999imitation"></d-cite> , <d-cite key="kober2010imitation"></d-cite> , <d-cite key="englert2018manipulation"></d-cite> , <d-cite key="finn2017oneshot"></d-cite> , <d-cite key="billard2008rpd"></d-cite> , <d-cite key="argall2009survey"></d-cite> . Classical methods like DMP and PrMP <d-cite key="schaal2006dmp"></d-cite>, <d-cite key="kober2009primitives"></d-cite>, <d-cite key="paraschos2013promp"></d-cite>, <d-cite key="paraschos2018promp"></d-cite> are sample-efficient but limited with high-dimensional inputs and closed-loop control. Deep IL approaches <d-cite key="mandlekar2021offline"></d-cite>, <d-cite key="zhang2018deep"></d-cite>, <d-cite key="mandlekar2020learning"></d-cite>, <d-cite key="lynch2020grounding"></d-cite>, <d-cite key="reed2022gato"></d-cite>, <d-cite key="lee2022multimodal"></d-cite>, <d-cite key="shridhar2022cliport"></d-cite>  offer more flexibility but require many human demonstrations, which is labor-intensive <d-cite key="jang2022bcz"></d-cite>, <d-cite key="shafiullah2022robosuite"></d-cite>. MimicPlay reduces this burden by leveraging easily collected human play data, minimizing the need for on-robot demonstrations.

**Hierarchical imitation learning** Previous methods for hierarchical policy learning relied solely on costly teleoperated robot demonstrations for both planning and control. In contrast, Mimicplay's approach combines inexpensive human play data for high-level planning with a small amount of robot demonstrations for low-level control, improving planning ability while reducing data requirements.



**Learning from human videos**  Previous work has explored using large-scale human video data to support robot policy learning , with approaches like R3M <d-cite key="tobin2017domain"></d-cite> and MVP <d-cite key="andrychowicz2020learning"></d-cite> leveraging the Ego4D dataset <d-cite key="akkaya2019solving"></d-cite>  to pretrain visual representations. However, domain diversity makes transferring these features to specific manipulation tasks difficult, and even simple augmentations can be similarly effective <d-cite key="rahmatizadeh2018vision"></d-cite> . To reduce this gap, some methods use in-domain human videos, enabling sample-efficient reward shaping and imitation learning, though they mainly extract rewards or features rather than aiding low-level action generation. In contrast, Mimicplay's work derives trajectory-level task plans from human play data, offering high-level guidance that improves low-level control in long-horizon manipulation tasks.



**Learning from play data** The proposed approach extends prior work on learning from play <d-cite key="yu2018one"></d-cite>, <d-cite key="lynch2020play"></d-cite>, <d-cite key="cui2022play"></d-cite> by replacing labor-intensive teleoperated play data with human play data collected through freehand interactions with the environment. This strategy provides rich trajectory-level guidance in only minutes, enabling robots to master complex long-horizon tasks with less than 30 minutes of teleoperation data.


---

## MimicPlay

<div class="row mt-3">
    <div class="col-sm text-left" id="fig:overview-mimicplay">
        {% include figure.liquid loading="eager" path="assets/img/mimicplay/overview.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption mt-2 text-center">
    Figure 2: Overview of MimicPlay <d-cite key="wang2023mimicplaylonghorizonimitationlearning"></d-cite>
</div>

**Learning 3D-aware latent plans from human play data** For long-horizon tasks defined by goal images, the problem is framed as hierarchical policy learning, where a goal-conditioned planner extracts features from the goal observation and converts them into low-dimensional latent plans to guide a low-level controller. To handle the multimodality of goal distributions without requiring massive datasets, this approach leverages inexpensive and easy-to-collect human play data.

**Learning multimodal latent plans** With human play data and the associated 3D hand trajectory $\tau$, the task is framed as goal-conditioned 3D trajectory generation. An observation encoder $E$ extracts features from the observation $o^h_t$ and goal image $g^h_t$, which are mapped by an MLP-based encoder into a latent plan vector $p_t$. Conditioned on $p_t$ and the hand location $l_t$, an MLP-based decoder predicts the 3D trajectory. To handle the multimodal nature of human motions, the trajectory distribution is modeled using a Gaussian Mixture Model (GMM) <d-cite key="bishop1994mdn"></d-cite>.

$$
p(\tau \mid \theta) = \sum_{z} p(\tau \mid \theta, z) \, p(z \mid \theta),
$$

The final learning objective of our GMM model is to minimize the `negative log-likelihood` of the detected 3D human hand trajectory $$ \tau $$ as


$$
\mathcal{L}_{\text{GMM}}(\theta) = - \mathbb{E}_{\tau} \left[ 
\log \left( \sum_{k=1}^{K} \eta_k \, \mathcal{N}(\tau \mid \mu_k, \sigma_k) \right) 
\right],
\quad \text{where } 0 \leq \eta_k \leq 1, \; \sum_{k=1}^{K} \eta_k = 1.
$$

**Handling visual gap between human and robot domains.** The setup assumes humans and robots act in the same environment, but visual differences between domains hinder transferring the latent planner to robot control. To bridge this gap, the method minimizes the distance between human and robot feature embeddings (distribution's mean and variance) $$ Q^h = E(o^h) $$ and $$ Q^r = E(o^r) $$ using a KL divergence loss: {% raw %}
$$
\mathcal{L}_{\text{KL}} = D_{\text{KL}}(Q^r \; || \; Q^h)
$$
{% endraw %}



Importantly, this does not require paired human–robot video data—$V^h$ and $V^r$ may involve different behaviors or tasks. Only image frames are needed to reduce the representation gap. The final loss for training the latent planner is defined as: $$ \mathcal{L} = \mathcal{L}_{\text{GMM}} + \lambda \cdot \mathcal{L}_{\text{KL}}, $$  where $$ \lambda $$ is a hyperparameter balancing the two losses.

**Plan-guided multi-task imitation learning**  MIMICPLAY addresses multi-task imitation learning, where a single policy is trained to execute multiple goal-conditioned tasks. Unlike prior end-to-end approaches <d-cite key="cui2022play"></d-cite>, <d-cite key="yu2018one"></d-cite>, <d-cite key="andrychowicz2017her"></d-cite> that require large amounts of teleoperation data (e.g., 4.5–6 hours <d-cite key="cui2022play"></d-cite>, <d-cite key="rosete2022latent"></d-cite>), MIMICPLAY leverages a latent planner $P$ pretrained on just 10 minutes of human play data to compress high-dimensional inputs into low-dimensional latent plans $p_t$. These latent plans provide rich 3D guidance, allowing the low-level policy $\pi$ to efficiently learn the mapping from plans $p_t$ to actions $a_t$.

**Video prompting for latent plan generation.** 
Instructing a robot to perform long-horizon visuomotor tasks is challenging due to complex goal specifications. The latent planner $$ P $$, trained on human play videos, can interpolate 3D-aware task-level plans directly from human motion, serving as an interface for guiding long-horizon manipulation. Specifically, a one-shot video $$ V $$ (either human $$ V^h $$ or robot $$ V^r $$) is used as a goal prompt for the pretrained planner to generate robot-executable latent plans $$ p_t $$. The video is first converted into a sequence of image frames, and at each time step, the high-level planner $$ P $$ takes the current frame $$ g_t $$ as a goal input to produce a latent plan $$ p_t $$, which guides the low-level action $$ a_t $$. After executing $$ a_t $$, the next frame in the sequence is used as the new goal image.


<!--  laten planner -->
<div class="row mt-3">
    <div class="col-sm text-center">
        {% include figure.liquid loading="eager" path="assets/img/high_level/latent-planner.gif" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption mt-2 text-center">
    Latent planner processing a human prompt: The planner operates in a sliding-window manner, taking the current observation image and the goal image provided by the human prompt. It encodes these into latent vectors for the start and goal, which are updated continuously as the end-effector progresses toward the target.
</div>

**Transformer-based plan-guided imitation.** Decoupling planning from control enables the policy to focus on precise action execution. High-level plans are combined with wrist camera and proprioceptive features to form token embeddings, which a transformer <d-cite key="vaswani2017attention"></d-cite> processes for long-horizon predictions. Actions are generated through a GMM-based decoder to handle multimodal robot behaviors.

---

## Implementation



### Comparison Between Original and Our Setup


<div class="row mt-3">
    <div class="col-sm text-center">
        {% include figure.liquid loading="eager" path="assets/img/mimicplay/our_setup.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption mt-2 text-center">
    Lab setup: two external cameras (front and back) are used, but no wrist-mounted camera is available
</div>

<div class="row mt-3">
    <div class="col-sm text-center">
        {% include figure.liquid loading="eager" path="assets/img/mimicplay/mimicplay_setup.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption mt-2 text-center">
    Original Robot setup from Mimicplay authors
</div>





### Franka Teleoperation system

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

### Data Collection Pipeline


### Human Play data

We collected a dataset of human play data by zed camera in two views. Here, human play data refers to videos that capture human hand–environment interactions during manipulation tasks. In our experiments, the demonstrations primarily consist of pick-and-place activities, such as picking up a plastic chili model from one bowl and placing it into another.

The raw demonstrations are stored in MP4 format with a frame rate of 20 FPS. Afterwards, we apply a series of post-processing steps to transform these recordings into the standardized robomimic format

- **Hand detection**  
   We use a pretrained hand detection model<d-cite key="Shan20"></d-cite>[![GitHub Repo](https://img.shields.io/badge/GitHub-handobj-blue?logo=github)](https://github.com/ddshan/hand_object_detector) to locate human hands in the video frames. We process the videos frame by frame, detecting the position of the hands in each frame and recording it.

- **3D triangulation**  
   We triangulate the detected hand positions to obtain their **3D coordinates in the world frame** by using the **calibrated stereo camera setup** (two synchronized viewpoints).

- **Dataset conversion**
   After extracting the image observations and the corresponding 3D hand positions, we store the processed data in the robomimic format. In addition, for each frame, we compute the trajectories of the subsequent 10 time steps based on temporal differences, and these predicted short-horizon trajectories are also saved in the same robomimic format.

<div class="row mt-3">
  <div class="col-sm mt-3 mt-md-0 text-center">
    {% include video.liquid path="assets/video/high_level/data_show.mp4" class="img-fluid rounded z-depth-1" controls=true autoplay=true %}
  </div>
</div>
<div class="caption text-center">
  Each frame with its corresponding future ten-step ground-truth trajectory. Green dots indicate the ground-truth trajectories over the subsequent 10 time steps.
</div>

Through the above processing pipeline, we obtain our dataset with follow stucture:
```
Group: data/demo_0
  - attributes:
    - num_samples: 26
Dataset: data/demo_0/actions, shape=(27, 30), dtype=float64
Dataset: data/demo_0/dones, shape=(26,), dtype=float64
Dataset: data/demo_0/interventions, shape=(27, 1), dtype=float64
Group: data/demo_0/obs
Dataset: data/demo_0/obs/agentview_image, shape=(27, 360, 640, 3), dtype=uint8
Dataset: data/demo_0/obs/agentview_image_2, shape=(27, 360, 640, 3), dtype=uint8
Dataset: data/demo_0/obs/front_image_1, shape=(27, 360, 640, 3), dtype=uint8
Dataset: data/demo_0/obs/front_image_2, shape=(27, 360, 640, 3), dtype=uint8
Dataset: data/demo_0/obs/hand_act_1, shape=(27, 1, 12), dtype=float64
Dataset: data/demo_0/obs/hand_act_2, shape=(27, 1, 12), dtype=float64
Dataset: data/demo_0/obs/hand_loc, shape=(27, 1, 3), dtype=float64
Dataset: data/demo_0/obs/hand_loc_1, shape=(27, 1, 12), dtype=float64
Dataset: data/demo_0/obs/hand_loc_2, shape=(27, 1, 12), dtype=float64
Dataset: data/demo_0/obs/robot0_eef_pos, shape=(27, 1, 3), dtype=float64
Dataset: data/demo_0/obs/robot0_eef_pos_future_traj, shape=(27, 30), dtype=float64
Dataset: data/demo_0/policy_acting, shape=(27,), dtype=float64
Dataset: data/demo_0/rewards, shape=(26,), dtype=float64
Dataset: data/demo_0/states, shape=(0,), dtype=float64
Dataset: data/demo_0/user_acting, shape=(27, 1), dtype=float64
```

Additionaly, we also do a **Projection validation (visualization check)** To verify the correctness of the calibration, we re-projected the obtained 3D points back to the image plane and visually inspected their alignment with the detected 2D hand positions. This ensured that the existed **camera parameters** were consistent with the real-world coordinate system.

During validation, we observed that in certain frames the back-projection failed to recover valid hand positions. Such frames were labeled as invalid frames. To assess data quality, we compared the ratio of invalid frames to the total number of frames for each demonstration. Based on this criterion, 3 out of the 10 collected demonstrations exhibited excessive invalid frames and were discarded. The remaining 7 demonstrations were retained.

Below is the detection code used for this visualization check:

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



#### Low level Teleoperation Data

We record rosbag from various topics. Here is the list of topics we record. However, this will need further post-processing because all the topics are published at different frequncies.

```
topics:
  - /franka_state_controller/franka_states           => Didn't use
  - /franka_gripper/joint_states                     => Gripper Joint state
  - /franka_state_controller/joint_states_desired    => Didn't use
  - /franka_state_controller/O_T_EE                  => End-effector position
  - /franka_state_controller/joint_states            => Joint states of the robot
  - /cartesian_impedance_controller/desired_pose     => desired EE- position published by the teleop system
  - /zedA/zed_node_A/left/image_rect_color           => Front camera
  - /zedB/zed_node_B/left/image_rect_color           => Back camera
```

We first estimated the frequencies of all the topics and then used our sampling algorithm to resample at a fixed frequency, corresponding to the rate at which we want our policy controller to operate.



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

## High Level Latent Planner

### Dataset

For the training of the high-level latent planner, we utilize the 7 valid demonstrations obtained after post-processing and filtering. These demonstrations are randomly split into a training set and a validation set: 6 demonstrations are assigned to the training set, while the remaining 1 demonstration is reserved for validation.

To train the GMM-based high-level planner, we design the input-output structure of the training data as follow figure. The inputs consist of two RGB images and the current 3D hand position. Among the two images, the current image represents the present frame, while the goal image corresponds to a frame sampled from a future time step within the same demonstration. The label for each training sample is defined as the ground-truth trajectory over the subsequent 10 time steps starting from the current frame.



### Training

#### Setup

The training was conducted following the **configuration provided in the reference paper**<d-cite key="wang2023mimicplaylonghorizonimitationlearning"></d-cite>.
For hyperparameters, we mainly relied on the **default settings from the official repository**[![GitHub Repo](https://img.shields.io/badge/GitHub-mimicplay-blue?logo=github)](https://github.com/j96w/MimicPlay/blob/main/mimicplay/configs/highlevel_human.json), while performing **additional tuning** based on our own dataset to improve performance, In particular, we focused on two key hyperparameters:

**goal_image_range**: defines the temporal distance between the current image and the goal image. A larger range allows the planner to consider goals further into the future, whereas a smaller range constrains the model to short-horizon predictions.

**std** in GMM: corresponds to the standard deviation parameter in the Gaussian Mixture Model (GMM), which controls the smoothness and variability of the learned trajectory distribution. Adjusting this value influences how tightly the model clusters motion patterns and how much uncertainty it tolerates in trajectory generation..

#### Loss Function

The loss function is defined as the negative log-likelihood of the ground-truth trajectory under the distribution modeled by the Gaussian Mixture Model (GMM). Concretely, given the predicted GMM parameters at each time step, the true future trajectory is evaluated against the probability density of the mixture, and the optimization objective minimizes the negative of this log-probability.

#### Evaluation

We evaluated the high-level planner using two metrics:

1. **GMM likelihood probability (training phase)**  
   During training, we monitored the **likelihood of the ground-truth data under the learned GMM model**. This serves as a measure of how well the model captures the distribution of the demonstrations.

2. **Mean Squared Error(MSE)**  
   In the subsequent testing phase, we evaluate the high-level planner by computing the distance error between the predicted trajectories and the ground-truth hand positions. Specifically, for each frame in the video, the GMM model generates 10 sampled predictions, and we take their average as the final trajectory estimate. The overall MSE is then calculated across the entire sequence to assess the effectiveness of the high-level planner.

<!-- #### Results

After completing the above experimental setup, we trained the high-level planner on the processed dataset. The figure below presents the training loss curve, which illustrates how the optimization objective decreases over time, indicating stable convergence of the model. -->





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
    The latent planner takes the current observation and a goal image (from the human prompt) to generate a latent trajectory plan. The low-level policy then combines this latent plan with image observations and proprioceptive inputs to sample actions from a GMM, whose parameters are determined by a GPT model followed by an MLP layer <d-cite key="wang2023mimicplaylonghorizonimitationlearning"></d-cite>.
</div>


## Experiments

### High Level Planner

Based on the implementation details described earlier, we trained the high-level planner using the processed human play dataset. After configuring the model and hyperparameters as specified in the Implementation section, we obtained both the training results and the test results, which are reported below.

#### Training Results

The training loss curves demonstrate that we can get a good model by training.

<div class="row mt-3">
    <div class="col-sm text-center">
        {% include figure.liquid loading="eager" path="assets/img/high_level/single_view/single_view_loss.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>

<div class="caption mt-2 text-center">
    Training loss of High level planner
</div>

#### Test Results

We evaluated the trained high-level planner on a set of newly collected prompts. For each prompt, we computed the Mean Squared Error (MSE) between the predicted trajectories and the ground-truth hand trajectories. In addition, we visualized the predicted trajectories for each frame alongside the corresponding ground-truth trajectories, providing an intuitive comparison of the model's performance. The predicted trajectories are mean trajectories of sampled 10 times from GMM.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
      <!-- <h5>Human Prompts</h5> -->
      <p>Prompt 0</p>
        {% include video.liquid path="assets/video/high_level/single_view/single_views_demo_3_h264.mp4" class="img-fluid rounded z-depth-1" controls=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
       <!-- <h5>Robot Policy Acting</h5> -->
       <p>Prompt 1</p>
        {% include video.liquid path="assets/video/high_level/single_view/single_views_demo_4_h264.mp4" class="img-fluid rounded z-depth-1" controls=true %}
    </div>
</div>
<div class="caption text-center">
  The visualization of high level planner with single view images. The Green line indicates ground truth trajectory. The blue line indicates predicted by high level planner
</div>

#### Improvements

The results presented above indicate that the initial performance of the high-level planner was not fully satisfactory. To address this, we conducted further experiments using **two-view video data** as input, allowing the model to benefit from richer visual observations.

In addition, after training the low-level policy, we realized that the choice of hyperparameters should be considered in relation to the overall system rather than in isolation. Specifically, for the high-level planner, we argue that the **hand motion speed** in human demonstrations should be better aligned with the robot motion speed in the low-level policy to ensure consistency across layers.

Given that our task trajectories are relatively simple, we also reduced the **number of GMM modes** to avoid overly complex distribution estimation. Under these revised settings, we re-collected a new set of human play data and corresponding prompts for testing. The new training results are reported below.

<div class="row mt-4">
    <div class="col-sm text-center">
        <!-- <strong>After Sampling</strong> -->
        {% include figure.liquid loading="eager" path="assets/img/high_level/two_views_traj/bi_views_loss.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <p>Training loss with new configuration</p>
    </div>
</div>

  <div class="row mt-4">
    <div class="col-sm text-center">
        <!-- <strong>After Sampling</strong> -->
        {% include figure.liquid loading="eager" path="assets/img/high_level/box_plot.png" class="img-fluid rounded z-depth-1" zoomable=true %}
        <p>Compare between previews model and new model</p>
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
      <!-- <h5>Human Prompts</h5> -->
      <p>Prompt 2</p>
        {% include video.liquid path="assets/video/high_level/two_views/bi_views_demo_0_h264.mp4" class="img-fluid rounded z-depth-1" controls=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
       <!-- <h5>Robot Policy Acting</h5> -->
       <p>Prompt 3</p>
        {% include video.liquid path="assets/video/high_level/two_views/bi_views_demo_1_h264.mp4" class="img-fluid rounded z-depth-1" controls=true %}
    </div>
</div>
<div class="caption text-center">
  The visualization of high level planner with two views images. The Green line indicates ground truth trajectory. The blue line indicates predicted by high level planner
</div>



### Low-Level Policy — Policy Controller (Live System)

[![GitHub Repo](https://img.shields.io/badge/GitHub-PolicyController-blue?logo=github)](https://github.com/AnshPrakash/franka_teleop/blob/robot-policy/scripts/policy_controller.py)

**Multi-Stage Success Evaluation.** We evaluated the low-level policy using a three-stage success metric: (1) grasping the target object, (2) reaching the designated drop location, and (3) placing the object correctly. The policy achieved a 0% success rate at the first stage, which consequently led to failure in the subsequent stages. Nonetheless, we have clear insights into understanding of the underlying causes.

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
    Robot policy consistently follows the same path regardless of the human prompt, indicating overtraining. While the best-epoch model showed more variation, their movements were too erratic to safely evaluate on the real robot
</div>


### Key Limitations Observed

1. **High-Level Planner — Poor Embedding Quality**

   * We found that the high-level planner produced **high prediction errors** for trajectories, which possibly resulted in **poor latent embeddings**.
   * Through hyperparameter tuning, we discovered that our dataset required **fewer modes**(num_mode = 2 worked the best for two views high level latent planner)  for accurate trajectory prediction.
   * Due to these weak embeddings, the low-level policy suffered from **poor representations** (see MSE error of the single-view high-level planner), which prevented it from fully leveraging the benefits of human guidance.

2. **Absence of Wrist Camera**

   * There was a significant **distribution shift** between training and evaluation image inputs from the front and back cameras.
   * The original authors used a **wrist-mounted camera** to help stabilize the robot policy. However, we could not include one in our setup due to **shared robot setup with another group**.
   * Adding a wrist camera in our setup would likely **reduce distribution shift** and improve performance—**provided that a robust latent embedding of the human prompt is available**.

3. **Human playdata collection** We observed that keeping our hands consistently within the camera frame is crucial; otherwise, the training data becomes corrupted with noisy trajectories.


**Additionally, Lab Constraints**: The robot became unavailable at a certain point, which limited our ability to re-evaluate the low-level policy with the improved high-level planner and prevented a fully iterative process.


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