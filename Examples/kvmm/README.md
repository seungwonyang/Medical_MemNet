# Implementation of Memory Network in pyTorch

This repo including implementations of End-to-End Memory Network and Key-Value Memory Network

# Reuirements
You must have Python 2.7+, pip, and virtualenv installed. There are no other requirements.

# Data generation
You can use the singularity container on SDSC Comet. 
I created singularity image so that you can copy it into your home directory.
The location is /home/neesittg/scale_test/kvmm/pytorch-gpu-sdsc.img.
Please see this file, "[using_singularity.txt](https://github.com/seungwonyang/Medical_MemNet/blob/master/Examples/kvmm/using_singularity.txt)" for using the singularity container.

# Run
* % ./setup\_processed\_data.sh
* run the singularity in order to gen\_dict.py to generate the dictionary file
* run the singularity in order to gen\_sim\_data.py to generate the data for training similarity function
* After exiting the singularity container, you can submit the job into slurm scheduler.
* % sbatch slurm\_script.sb
