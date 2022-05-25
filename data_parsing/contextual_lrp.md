## Contextual LRP (cLRP) data collection protocol
LRP has been used succesfully in various applications to understand how deep learning models make decisions as 
demonstrated in EEG, activity recognition, and images. Post-hoc XAI methods aim to generate heatmaps that are understandable
to human eyes. For many modalities like images, we can make sense of the heatmap when being overlayed on the original 
input. Unfortunately, for tri-axial accelerometer data, we don't really know what's happening at each point in time except 
for the high-level features like trend, repetition and drift, which would make it difficult to understand the generated 
heatmap as we can't put the activations into context. Contextual LRP aims to address this by making use of concurrent video
recording with accelerometry, which provides the semantics around the important key frames given a segment of accelerometer 
trace.


### I. Activity class of interest
Since the purpose of cLRP is to understand AoT or verify that AoT is not learning non-sense, we would be interested in collecting
data of the following activities which are listed in the order of increasing difficulty for AoT task:

* Shaking hands
* Typing
* Walking (big step)
* Dancing
* Playing tennis

The main factor that determines the difficulty for the AoT is how symmetric is activity on a temporal dimension. 


### II. Data collection protocol
There are two key properties that the collected data needs to have for a fair comparison across different activities:

* Ease of alignment between video and accelerometry 
* Same duration for the activity performed.


#### II.a Baseline recording
For each of five activities listed, we will have a five-minute continuous recording with concurrent video footage and 
accelerometry data. Before the onset and termination each activity, we should have a 30-sec of rest period with wrist
resting flat on a surface to mark the start and ending for this recording period. 

#### II.b Activities with rest
To explore how AoT-based features treat stationary periods, we will also record activities of shorter intervals which are 
separated by stationary windows. We will repeatedly perform certain activity for five seconds and keep still for five seconds
for a cycle of ten times. Similarly, 30-sec of resting period at the start and the end will also be applied. In the post-processing,
this will allow us to place the rest period both before or after the real movement periods.

In sum, for a particular activity the `baseline recording` sequence is the following:

0. Turn on the camera 
1. 30-sec rest 
2. Perform an action for five min
3. 30-sec rest 
4. Turn off the camera

Recording sequencing for `activities with rest` is:

0. Turn on the camera 
1. 30-sec rest 
2. Perform an action for five sec
3. Rest of five sec
4. Repeat step 2 and 3 for ten times   
3. 30-sec rest 
4. Turn off the camera

We might apply some off-the-shelf optic flow detection for videos for better understanding of the body movement.

Total Subject count: one

Video collection tool: a smartphone

Accelerometer: Axitivity 
