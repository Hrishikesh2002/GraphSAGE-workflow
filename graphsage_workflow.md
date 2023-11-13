# Working of GraphSage

## Running the Supervised training algorithm

The supervised training algorithm is run using the following command:

>python -m graphsage.supervised_train --train_prefix ./example_data/toy-ppi --model graphsage_mean --sigmoid


This calls supervised_train.py with arguments for training dataset, model type (aggregator type) and the activation function. supervised_train.py first runs the tf.app.run command, which parses the input arguments and runs the main function. The main function first loads the training data using load_data function and then passes this data to the train function. 

The train function extracts the graph information (structure and features) and the node labels and creates an object minibatch of class NodeMinibatchIterator. It initializes the neighbourhood sampler (uniform) and the model (as per the aggregator type specified). Then a tensorflow session is initialised, which defines the operations and allocates space for variables and intermediaries. The feed_dict is used to pass placeholders to the tensorflow network while training.

Then the train function start the main training loop, and runs it for specified number of epochs in minibatches, and at regular intervals validates the data on validation dataset. The NodeMinibatchIterator class has functions which help in iterating over individual nodes while training.

## Running the Unsupervised training algorithm

The unsupervised training algorithm is run using the following command:

>python -m graphsage.unsupervised_train --train_prefix ./example_data/toy-ppi --model graphsage_mean --max_total_steps 1000 --validate_iter 10

The working of this code is similar to the supervised case, except that the train function does not use the node labels and that the loss function used is different (unsupervised loss, which tries to minimize the distance between the embeddings of nodes that are nearby in the graph).

## Description of other files

### layers.py

This file contains the definition of a general layer (class Layer) and a specific Dense layer (derived class Dense), which is later used in defining Multi Layer Perceptron in the models.py file.

### models.py

This file contains the definitions of the different models used in the code. The models could be the Multi-Layer Perceptron (MLP) used for supervised training, or the sample and aggregate type models used for unsupervised training or a simple implementation of node2vec model/Deepwalk model. The sample and aggregate models have multiple variants depending on the type of aggregators i.e. mean, max, lstm, pool, etc.

### neigh_samplers.py

This file contains sampler for sampling neighbourhood of a node. It has only the uniform sampler, which gives every neighbourhood node equal probability of being sampled.

### metrics.py

This file contains definitions of metrics like accuracy, l2 loss adn cross-entropy which are used for evaluating the performance of the model during training.

