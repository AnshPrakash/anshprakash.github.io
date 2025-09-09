---
layout: distill
title: Mimic Play on Franka Single Arm and its Extension
description: Replicating MimicPlay on a real one arm robot, and its extension to  bi-manual robot
tags: MimicPlay at IROBMAN lab
date: 2025-09-08
citation: true
related_publications: true

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
  - name: Citations
  - name: Footnotes
  - name: References

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



## Citations

<d-cite key="wang2023mimicplaylonghorizonimitationlearning"></d-cite>
---