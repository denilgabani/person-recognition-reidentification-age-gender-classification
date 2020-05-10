import cv2
import numpy as np
from openvino.inference_engine import IECore

class PersonReidentificationModel:

    def __init__(self, model_name, device='CPU', extensions=None, num_requests=1):
        
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_structure = self.model_name
        self.model_weights = self.model_name.split('.')[0]+'.bin'
        self.num_requests = num_requests
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None
        self.is_sync = None

    def load_model(self):

        self.plugin = IECore()
        self.network = self.plugin.read_network(model=self.model_structure, weights=self.model_weights)
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        
        
        if len(unsupported_layers)!=0 and self.device=='CPU':
            print("unsupported layers found:{}".format(unsupported_layers))
            if not self.extensions==None:
                print("Adding cpu_extension")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network = self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers)!=0:
                    print("After adding the extension still unsupported layers found")
                    exit(1)
                print("After adding the extension the issue is resolved")
            else:
                print("Give the path of cpu extension")
                exit(1)
                
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device,num_requests=self.num_requests)
        if self.num_requests == 1:
            self.is_sync = True
        self.input_name = next(iter(self.network.inputs))
        self.input_shape = self.network.inputs[self.input_name].shape
        self.output_names = next(iter(self.network.outputs))
        self.output_shape = self.network.outputs[self.output_names].shape
        
    def predict(self, image, cur_req_id=None, next_req_id=None):

        
        img_processed = self.preprocess_input(image.copy())
        if self.is_sync:
            outputs = self.exec_net.infer({self.input_name:img_processed})
            rei_vector = self.preprocess_output(outputs)
            return rei_vector, True
        self.exec_net.start_async(request_id=next_req_id, inputs={self.input_name:img_processed})
        if self.exec_net.requests[cur_req_id].wait(0) == 0:
            outputs = self.exec_net.requests[cur_req_id].outputs
            rei_vector = self.preprocess_output(outputs)
            return rei_vector, True
        return None, False

    def check_model(self):
        ''

    def preprocess_input(self, image):

        image_resized = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        img_processed = np.transpose(np.expand_dims(image_resized,axis=0), (0,3,1,2))
        return img_processed
            

    def preprocess_output(self, outputs):

        output = outputs[self.output_names][0]
        return output
        

