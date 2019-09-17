
import json
import sys

from azureml.core.model import Model
import onnxruntime
import numpy as np

def init():
    global model_path
    model_path = Model.get_model_path(model_name = 'FashionDLModel')

def run(raw_data):
    try:
        data = json.loads(raw_data)['data']
        data = np.array(data, dtype=np.float32)

        session = onnxruntime.InferenceSession(model_path)
        first_input_name = session.get_inputs()[0].name
        first_output_name = session.get_outputs()[0].name
        result = session.run([first_output_name], {first_input_name: data})
        # NumPy arrays are not JSON serialisable
        result = result[0].tolist()

        return {"result": result}
    except Exception as e:
        result = str(e)
        return {"error": result}
