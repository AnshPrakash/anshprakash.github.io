---
layout: distill
title: MimicPlay on Franka Arm and its Extension
description: This blog is part of our university’s project lab, where we are working on replicating MimicPlay using a real one-arm robotic platform in our lab. Building on this setup, we aim to extend the approach to bi-manual systems such as the Tiago robot. Our work explores how abundant human play data can be leveraged to guide efficient low-level robot policies.
tags: Imitation-Learning, Learning-from-Human, Long-Horizon-Manipulation
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
  - name: Franka Teleoperation system
  - name: High Level Latent Planner
    subsections:
      - name: Model
      - name: Latent space
      - name: Multi-modality
      - name: Training
  - name: Low Level Policcy
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


---

## Related Works


---

## MimicPlay


---

## Data Collection Pipeline


### Human Play data


### Low level Teleoperation Data


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