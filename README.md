# In-silico Neuroscience with Meta TRIBEv2 
Validating neural encoding models through naturalistic face perception

## Terms
- In-silico: refers to scientific experiments or research performed on a computer, or through a computer simulation, as opposed to in living organisms (In-vivo) or in a lab (In-vitro)
- fMRI
- Functional Localization
- Parcellation
- Fusiform Face Area (FFA): One of the most well-established findiings in cognitive neuroscience. A part of the human visual system that is specialized for facial recognition, located in the inferior temportal cortex in Bordmann area 37. 

## What is Meta TRIBEv2?
A self-supervised vision transformer model introduced by Meta that acts as a digital twin of human neural activity. It's able to predict how the brain responds to sights, sounds, and language. [Introducing TRIBE v2 - Meta](https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/)

## What this project?
### Learning Outcomes
- ML pipeline design (video processing -> neural prediction -> statistical analysis)
- Computational Neuroscience (fMRI, functional localization, parcellation)
- Critical thinking about AI Ethics and societal impact of this technology
- Full-stack Engineering (Python/PyTorch backend, web frontend, GPU deployment)
### Process 
1. Video Preparation: Selecing a 3-5 minute movie clip with clear variation between face-heavy scenes and face-free scenes. Using ffmpeg for any trimming or format conversion
2. Face Detection: Run a face detector on every frame to produce a continuous "face presence" signal 
3. TRIBEv2 Inference: Load the pretrained model on colab, run the movie clip through, and save the prediction array and events
4. Analysis: 
5. Visualization
6. Open-source tool creation
