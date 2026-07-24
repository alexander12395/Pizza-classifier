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


class layer:
    def __init__(self,number_of_neurons:int,number_of_previous_neurons:int):
        self.neuron_count=number_of_neurons
        self.bias_array=np.zeros(number_of_neurons)
        self.weight_array=np.random.randn(number_of_neurons,number_of_previous_neurons)/np.sqrt(number_of_previous_neurons)#gives the weights array, rows are neurons, columns are weights, divides by number of inputs to give a make the weighted sum not saturate
        self.neuron_outputs=np.zeros(number_of_neurons)
        self.grad_array=np.zeros(number_of_neurons)
class network:
    def __init__(self,number_of_layers:int,neurons_per_layer:list,input_size:int):
        self.layer_array=np.empty(number_of_layers,dtype=object)
        self.number_of_layers=number_of_layers
        for l in range(number_of_layers):
            if l==0:
                self.layer_array[l]=layer(neurons_per_layer[l],input_size)
            else:
                self.layer_array[l]=layer(neurons_per_layer[l],neurons_per_layer[l-1])

    def forward_pass(self,input_images):
        for l in range(self.number_of_layers):
            weights=self.layer_array[l].weight_array
            bias=self.layer_array[l].bias_array  
            
            if l==0:
                self.layer_array[l].neuron_outputs=np.tanh(weights@input_images +bias)
            else:
                prev_output=self.layer_array[l-1].neuron_outputs
                self.layer_array[l].neuron_outputs=np.tanh(weights@prev_output +bias)
        return(self.layer_array[-1].neuron_outputs)
    

    def backpropagation(self,target_outputs:list,learning_rate:float,input_images):
        target_outputs=np.array(target_outputs)
        MSE=np.mean((self.layer_array[-1].neuron_outputs-target_outputs)**2)
        self.layer_array[-1].grad_array=(2*(self.layer_array[-1].neuron_outputs-target_outputs)*(1-self.layer_array[-1].neuron_outputs**2))
        for l in range(self.number_of_layers-2, -1, -1):
            a = self.layer_array[l].neuron_outputs
            next_weights = self.layer_array[l+1].weight_array
            next_grad = self.layer_array[l+1].grad_array
            self.layer_array[l].grad_array = (next_weights.T @ next_grad) * (1 - a**2)
            
        for l in range(self.number_of_layers):  
            inputs=input_images if l==0 else self.layer_array[l-1].neuron_outputs
            
            self.layer_array[l].bias_array-=self.layer_array[l].grad_array*learning_rate
            
            self.layer_array[l].weight_array-=learning_rate*np.outer(self.layer_array[l].grad_array,inputs)
        return MSE