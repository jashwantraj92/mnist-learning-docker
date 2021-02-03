# mnist-contaier

Learning MNIST using TensorFlow in a Docker container


## Prepare
MNIST data set has to downloaded and prepared for users to specify a path to test images

the script prepare.sh will download the dataset to /mnt/data/mnist

the script runs `download-mnist-data.py --out-dir=<out_dir>` downloads the training and test sets into
the required folder/file structure:

- \<out_dir> / \<train|test> / \<label> / \<image_index>.png

By default `<out_dir>="/mnt/data/mnist"` 


#### Build the Docker image

To build the Docker image, run the command.sh script with two arguments.
The first argument specifies "build" or "run". The second argument specfies the device "CPU" or "GPU"
Note, for GPU, you need to comment line-1 and uncomment line 4 in Dockerfile. 

#### Run the container for training and inference

Run the command.sh script with <run> <device>.
By default the training is set up to run for 3 epochs and batch size 16. These parameters are configurable by passing command line arguments to the docker run command.
 -n (num epochs)
 -b (batch_size)

Once training is complete, the model is saved at /mnt/data/mnist/mnist-model. Then the container waits for user-input for an image path. User can enter any path available from /mnt/data/mnist/test/<folder-number>. Example /mnt/data/mnist/test/1/1129.png
 
Note that, only for first time container execution, training is trigerred. For subsequent runs, the trained model saved in /mnt/data/mnist/mnist-model will be resused for prediction.
To re-run the training, the folder can be deleted. 
