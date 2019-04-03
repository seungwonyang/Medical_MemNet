Scalable KVmemNN architecture
(v 1.0)
- primarily focusing on the use with HPC environments (comet and bridges) with GPU-equipped nodes
- container with Singularity
- distributed and parallel task management with DASK
- distributed and parallel NN models (train and test) with PyTorch
- distributed memory for large datasets


(Resource)

Comet
- 36 nodes (4 x K80), 36 nodes (4 x P100)
- Singularity support
- Sheduler : SLURM
- Limits :  upto 8 nodes
- Lustre file system :  /oasis/scratch/comet/$your username/$temp_project/


Bridges
- Bridges AI :  10 nodes (1 DGX-2 + 9 V100)
  Bridges GPU : 48 nodes (16 nodes (4 x K80 (2 cards)), 32 nodes (2 x P100))
- Singualarity supported


(Workflow)
1. start Dask-sheduler and dask-worker with NFS accessible scheduler-file with qsub
2. start main script 
  - launch dask client
  - three steps are launched by client for three phases of Data, Learning, and Inference
  
 (Key considerations)
 1. config file for the entire workflow description
     - this an extension of Berlino's config file for KV-MemNN
     - description on an ensemble of tasks for Data, Learning, and Inference phases for Dask scheduling
     


