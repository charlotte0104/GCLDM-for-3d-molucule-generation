We provide pre-trained model weights for testing. For retraining the model, we recommend using multiple A6000 GPUs in parallel to accelerate the process. Otherwise, the training will take significantly longer.


We recommend training with a small batch size such as 64. Alternatively, you can use a large batch size for the first 1,500 training iterations, followed by 500 iterations with a small batch size to achieve better convergence.
