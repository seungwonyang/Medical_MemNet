# Implementation of Memory Network in pyTorch
This is implemeted on my desktop which is ruiining ubuntu linux.


# Reuirements
You must have Python 2.7+, pip, and virtualenv installed. There are no other requirements.
# Run
``% virtualenv venv``<br>
``% source venv/bin/activate``<br>
``% pip install -r requirements.txt``<br>
``% ./setup_processed_data.sh``<br>
``% python gen_dict.py``<br>
``% python gen_sim_data.py``<br>
``% python train_mlp.py``<br>
