from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.conda_dependencies import CondaDependencies 
from azureml.core.image import ContainerImage
from azureml.core.webservice import AciWebservice, Webservice


ws = Workspace.create(name='fashiondeeplearning',
                   subscription_id='02e39ba6-b26e-47cd-a81e-90c4c236aabb', 
                   resource_group='myresourcegroup',
                   create_resource_group=True,
                   location='westeurope' 
                  )


model = Model.register(model_path = "fashion.onnx",
                       model_name = "FashionDLModel",
                       description = "Fashion Keras Model",
                       workspace = ws)

myenv = CondaDependencies()
myenv.add_pip_package("numpy")
myenv.add_pip_package("azureml-core")
myenv.add_pip_package("onnxruntime")

with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())

image_config = ContainerImage.image_configuration(execution_script = "score.py",
                                                  runtime = "python",
                                                  conda_file = "myenv.yml",
                                                  description = "test"
                                                 )
image = ContainerImage.create(name = "myonnxmodelimage",
                              models = [model],
                              image_config = image_config,
                              workspace = ws)

image.wait_for_creation(show_output = True)

aciconfig = AciWebservice.deploy_configuration(cpu_cores = 1, 
                                               memory_gb = 1, 
                                               tags = {"data": "fashion_mnist", "type": "classification"}, 
                                               description = 'Fashion recognition')

service_name = 'keras-mnist-classification'
service = Webservice.deploy_from_image(deployment_config = aciconfig,
                                            image = image,
                                            name = service_name,
                                            workspace = ws)

service.wait_for_deployment(show_output = True)
print(service.state)
