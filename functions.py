from PIL import Image
import os
import numpy as np
import math

def get_folder_RGB(folder):
    total_images=[]
    filenames=os.listdir(folder)
    for f in filenames:
        path=os.path.join(folder,f)
        img=Image.open(path)
        img=img.convert("RGB")
        img=img.resize((128,128))
        arr=np.array(img)
        total_images.append(arr)
    return(total_images)    
#lists all files in directory, runs through them, imports all into rgb, changes definition


class neuron:
    
    def __init__(self,input_count:int):
        self.weights=np.random.randn(input_count);
        self.bias=np.random.randn(1)
        self.grad=0
        self.inputs=np.empty(input_count)
        self.output_value=0
    

    def compute_output(self,inputs ):
        total=0
        for i in range(len(inputs)):
            total+=self.weights[i]*inputs[i]
        total=total+self.bias
        return np.tanh(total)
#neuron class holds weights, uses non linear function tanh

    

class layer:
    def __init__(self,layer_number:int,neuron_count:int,prev_layer_neuron_count:int,):
        self.layer_number=layer_number
        self.neuron_count=neuron_count
        self.neuron_array=np.empty(neuron_count,dtype=object)
        self.prev_layer_neuron_count=prev_layer_neuron_count
        self.neuron_output=np.empty(neuron_count)
        for n in range(neuron_count):   
            self.neuron_array[n]=neuron(prev_layer_neuron_count)

    def forward(self,prev_outputs:object):
        for n in range(self.neuron_count):
            total=0
            total=self.neuron_array[n].compute_output(prev_outputs)
            self.neuron_output[n]=total
        return self.neuron_output
    

class network:
    def __init__(self,amount_layers:int,neuron_taper:int,output:int,input_size):
        self.amount_layers=amount_layers
        self.neuron_taper=neuron_taper
        self.layer_array=np.empty(amount_layers,dtype=object)
        prev_count = input_size
        for l in range(amount_layers):
            neuron_count = max(input_size - neuron_taper * l, output)
            self.layer_array[l] = layer(l, neuron_count, prev_count)
            prev_count = neuron_count   # remembers what ACTUALLY happened, floor included



    def forward_pass(self, inputs):
        current = inputs
        for l in range(self.amount_layers):
        current = self.layer_array[l].forward(current)
        return current

    def backprop(self):
        





                

    
     
        


        



    


    




        
        
                        

        

