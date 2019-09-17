from IPython import get_ipython
import numpy as np
import os
import matplotlib.pyplot as plt
import azureml
from azureml.core import Workspace
from azureml.core import Experiment
import urllib
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
import shutil
from azureml.train.dnn import TensorFlow


# check core SDK version number
print("Azure ML SDK Version: ", azureml.core.VERSION)

ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep='\n')

script_folder = './keras-mnist'
os.makedirs(script_folder, exist_ok=True)

exp = Experiment(workspace=ws, name='fashion-mnist-adam')


os.makedirs('./data/mnist', exist_ok=True)
urllib.request.urlretrieve('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz', filename='./data/mnist/train-images.gz')
urllib.request.urlretrieve('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz', filename='./data/mnist/train-labels.gz')
urllib.request.urlretrieve('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz', filename='./data/mnist/test-images.gz')
urllib.request.urlretrieve('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz', filename='./data/mnist/test-labels.gz')

ds = ws.get_default_datastore()
ds.upload(src_dir='./data/mnist', target_path='mnist', overwrite=True, show_progress=True)

# choose a name for your cluster
cluster_name = "racer"

try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D2_V2', 
                                                           max_nodes=4)

    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)

    # can poll for a minimum number of nodes and for a specific timeout. 
    # if no min node count is provided it uses the scale settings for the cluster
    compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

# use get_status() to get a detailed status for the current cluster. 

# the training logic is in the keras_mnist.py file.
shutil.copy('./train.py', script_folder)

# the utils.py just helps loading data from the downloaded MNIST dataset into numpy arrays.
shutil.copy('./utils.py', script_folder)

script_params = {
    '--data-folder': ds.path('mnist').as_mount(),
    '--batch-size': 64,
    '--learning-rate': 0.001
}

estimator = TensorFlow(source_directory=script_folder,
                 script_params=script_params,
                 compute_target=compute_target, 
                 pip_packages=['keras', 'matplotlib','sklearn','onnxmltools'],
                 entry_script='train.py', 
                 use_gpu=False)


run = exp.submit(estimator)





