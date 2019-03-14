## Implementation of Memory Network in pyTorch
This repo including implementations of End-to-End Memory Network and Key-Value Memory Network

## Data generation
You can use the singularity container on SDSC Comet. 
I created singularity image so that you can copy it into your home directory.
The location is **/home/neesittg/scale_test/kvmm/pytorch-gpu-sdsc-cyoun.simg**.

## Run
Before running the following steps, you need to correct the path in each slurm scripts.
1. Get the processed data
    * % ./setup_processed_data.sh
2. Generate the dictionary file
    * % sbatch slurm_gen_dict_script.sb
3. Generate the data for training similarity function
    * % sbatch slurm_gen_sim_data_script.sb
4. Distribute train the similarity function
    * % sbatch slurm_mlp_script.sb
5. Generate the data for reader
    * % sbatch slurm_gen_reader_data_script.sb
6. Distribute train the key-value memory network model
    * % sbatch slurm_kvmm_script.sh