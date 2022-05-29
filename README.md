# Neural-Network-From-Scratch

**A Neural Network built from scratch using numpy. The dataset I use is the MNIST dataset. It contains 70000 28x28 images of hand written digits.**

![320px-MnistExamples](https://user-images.githubusercontent.com/47878915/170891491-1036b2b8-1d4c-43b9-bc57-068e7ad4854b.png)

The Neural Network has 4 total layers.

**Input Layer:**
  - inputs the dataset of images.   
  - the images are put into a array.
  - since 28 x 28 = 784, this layer will have 784 nodes.

**Hidden Layer 1:**
  - in this layer we simply reduce the number of nodes from 784 to 128.
 
**HIdden Layer 2:**
  - this layer does the same as the previous layer. Reduces the number of nodes to 64.
 
**Output Layer:**
  - reduces the number of nodes to 10. We evaulate the nodes against label.
  - label is an array of 10 elements where 1 is 1 and the rest are 0's.

**The output can be calculated using this formula.** 

![1__Jr6WYMq09Pia_VppNOv7Q](https://user-images.githubusercontent.com/47878915/170891546-a0767fd1-bad1-46e3-9159-d9015fa86a6a.png)

**We use the ReLu function as out activation function due to its simplicity and compute time. **

![images](https://user-images.githubusercontent.com/47878915/170891556-06ab71da-cbba-40a0-97ab-99e64176c8d7.jpeg)
 
 **In the last layer (output layer), we turn the output into probability distribution. We do this by applying the Softmax function. **
 
![0ab139bc-3ff6-49d2-8b36-dcc98ef31102](https://user-images.githubusercontent.com/47878915/170891594-017ad1f6-21b9-40b4-9155-93554ecdb210.png)
 
 
