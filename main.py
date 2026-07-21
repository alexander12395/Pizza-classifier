from functions import *
import random
import pickle

pizza = np.load("pizza.npy")
not_pizza = np.load("not_pizza.npy")

training_data = []
for img in pizza:
    training_data.append((img, [1, 0]))
for img in not_pizza:
    training_data.append((img, [0, 1]))

neural_net = network(4,60,2,260,len(pizza[0].flatten()))

target_mse = 0.05     # stop once average MSE drops below this
max_epochs = 1000     # safety cap so it can never loop forever

epoch = 0
avg_mse = 1.0         # start high so the loop runs at least once

while avg_mse > target_mse and epoch < max_epochs:
    random.shuffle(training_data)
    total_mse = 0
    for img, target in training_data:
        image = img.flatten() / 255
        prediction = neural_net.forward_pass(image)
        mse = ((prediction[0] - target[0])**2 + (prediction[1] - target[1])**2) / 2
        total_mse += mse
        neural_net.backprop(target, 0.05, image)
    avg_mse = total_mse / len(training_data)
    epoch += 1
    print(f"epoch {epoch}: average MSE = {avg_mse}")

print(f"Training finished after {epoch} epochs, final MSE = {avg_mse}")

# --- accuracy check ---
correct = 0
incorrect = 0
for img, target in training_data:
    image = img.flatten() / 255
    prediction = neural_net.forward_pass(image)
    # the network's guess is whichever output neuron is highest
    predicted_class = np.argmax(prediction)
    actual_class = np.argmax(target)
    if predicted_class == actual_class:
        correct += 1
    else:
        incorrect += 1

total = correct + incorrect
percentage = (correct / total) * 100
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {percentage:.2f}%")

with open("trained_network.pkl", "wb") as f:
    pickle.dump(neural_net, f)
print("Network saved to trained_network.pkl")