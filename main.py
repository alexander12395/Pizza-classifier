from functions import*

#not_pizza=get_folder_RGB("not_pizza")
#np.save("not_pizza.npy",np.stack(not_pizza))
#pizza=get_folder_RGB("pizza")
#np.save("pizza.npy",np.stack(pizza))

pizza=np.load("pizza.npy")
not_pizza=np.load("not_pizza.npy")


n=neuron(3)
print(n.compute_output([1,2,3]))