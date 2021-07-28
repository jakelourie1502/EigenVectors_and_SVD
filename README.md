# Effect of Principal Components Analysis on Classification Performance of MNIST
In this notebook, we use a simple Torch model to classify MNIST digits. We then apply principle components analysis to see how well the model can classify the reduced dimension feature representations.

We are going to reduce the dimensionality of our dataset(currently 28 x 28 x 1) by using eigenvectors. This will allow us to capture the most important variations between images. We will then see if we can apply just this reduced number of dimensions to a neural net and gain a similar performance level to that which we achieved with a ResNet on the full images (~85%)

Steps:

1- Flatten arrays

2- normalize data across each dimension

3- Compute N eigenvectors and the covariance matrix

4- Project images onto eigenspace using projection matrix:

projection = B @ (B.T@B)^-1 @ B.T }
where B is new basis. B.T@B will be the identity matrix if the eigenvectors are normalised, so it would just become:

projection = B @ B.T
We will experiment with various choices of N, but 50 is a good place to start

CONCLUSION
It's clear that we can achieve the same classification performance from around 50 dimensions. 
The time to train is slightly unstable, but it appears we can also reduce the training time of the model by around 50% using a lower dimensional representation of the data too.
