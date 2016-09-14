# Recurrent Neural Network

## Introduction
The purpose of this quick tutorial is to get you a very big, very useful neural network up and running in just a few hours. The goal is that anyone with a computer, some free time, and no knowledge of what neural networks are or how they work can easily begin playing with this technology as soon as possible. Technical explanations of what RNNs are abound on the internet, so this tutorial will skip explanation and focus solely on building.

With this neural network, you'll be able to feed in a body of text as input, and as output receive whatever the neural network believes is a good imitation of that input text. For example, given 3MB of recipes from cookbooks.com, my recurrent neural network (RNN from here on out) has so far produced thousands of culinary gems, including Salmon Cookies, Wee's Dippes, Pickles Quiche, Mexican Lasagna, Cajun Frosting, Cranberry Cocktrot Salad, and simple but delectable Bread Candy:

BREAD CANDY 

Ingredients
- 1  lb. cooked ham (fresh)
- 1  onion, chopped fine

Preparation

Combine butter, melted butter, sugar, vanilla and 1 cup with eggplant mixture. Spread evenly with cracker crumbs, butter cream and celery.  Add remaining ingredients except whips.  Chill in a large bowl according to package, and thinly slice mixture over the graham cracker crumbs.  Refrigerate.

### Who are you?
You don't need to have any keep knowledge of what a neural network is or how it works. In fact, you don't need to have any knowledge whatsoever. Though the concepts and mathematics behind neural networks are fascinating and not *too* involved, if you like you can treat the neural network as a black box where something goes in and something comes out. Similarly, you don't need to to be a skilled programmer. In fact, the only real requirement is that you are comfortable opening up the terminal and can navigate directories which, if you don't already know how to do, I promise can be learned in 15 minutes. For the most part, all you will need to do is download a few things off the internet and copy and paste the few short lines of code provided for you in this tutorial.

### Who built this?
One week ago, I sat down at my computer and decided I would like to play around with the latest AI technology. I wanted to see what a neural network would produce if given 50% Shakespeare and 50% Donald Trump. There are a few tutorials out there,

https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1#.catnczqp3


but these are usually limited to explaining how vanilla NNs work and getting you started with an NN that can crunch a small array of numbers. But what if you want to build something big? 

Before I began trying to build one from scratch, I was lucky enough to stumble upon a tutorial of a tutorial of a tutorial which, amazingly, lets you set up a state-of-the-art RNN with almost zero effort. The project in question is here, created by Andrej Karpathy, a researcher at OpenAI who holds a PhD in deep learning from Stanford: 
https://github.com/karpathy/char-rnn

His project has been maintained and optimized here:
https://github.com/jcjohnson/torch-rnn

And *his* project has received a docker image here:
https://github.com/crisbal/docker-torch-rnn

The nice thing about this project, and largely the reason I'm creating this tutorial, is that running this on Docker means you don't have to worry about dependencies and compatibility (stuff to install that may or may not work on your computer). It occurred to me that the entire process, end-to-end, was so simple that someone with almost no knowledge of programming could do it. 

So, let's begin:

#### Step #1: Install Docker
Go to https://www.docker.com/ and install Docker. Open it up.

#### Step #2: Run Docker
Open up terminal and type:

docker run --rm -ti crisbal/torch-rnn:base bash

#### Step #3: Select your training data
Inside of the docker instance there's some sample Shakespeare data, but you're here because you want to train an RNN on something unique. Recipes? Donald Trump tweets? A math textbook? The Twilight series? 5MB of Miss Manners? Sci-Fi screenplays? Yelp reviews?

Whatever it is, you'll want at least 1MB of it, and we're going to move it from your local drive into the docker instance. 

#### Step #4: Moving your data into docker
Open up a new terminal window (not the one running docker) and type: 

docker ps

This will give you information about your docker instances that you will need. On the far right is a "NAME" column. I've opened up a few by now, and they all have superb names.

Now, in terminal, you'll need to navigate to the directory that contains your data file. If you don't already know how to do that, there are plenty of resources on the internet if you google. The first that came up for me is this one, and it seems like a good start

http://www.macworld.com/article/2042378/master-the-command-line-navigating-files-and-folders.html

Once you have done this, you're going to copy this file over to the data folder in docker, like so:

docker cp FILENAME.txt NAMEOFYOURDOCKER:/root/torch-rnn/data/FILENAME.txt

where NAMEOFYOURDOCKER is what we found earlier with "docker ps" and FILENAME is the name of your data. For example, I will put:

docker cp allrecipes.txt sleepy_snyder:/root/torch-rnn/data/allrecipes.txt

#### Step #5: Preprocessing

Go back to the terminal window running docker, and type:

python scripts/preprocess.py \
  --input_txt data/FILENAME.txt \
  --output_h5 data/FILENAME.h5 \
  --output_json data/FILENAME.json
  
#### Step #6: Training Parameters

Now we're going to train our data. This is essentially the last step, but a few words before we do so.

Training the network takes a long time, minimum perhaps 6-12 hours. Fortunately, this goes on in the background without, at least for my MacBook, noticeably hurting your performance. Just a warning.

You will also need to set the training parameters to match your dataset. The more data you use and the bigger you make your model, the better it will be. Obviously, bigger model = more time spent training.

The details of setting parameters are included in the original repository, and if you want to get a well-performing model then you would be well-served to review the notes under "Tips and Tricks"

https://github.com/karpathy/char-rnn

But I will summarize and give you a very basic rule of thumb. 

- 1MB of data is considered pretty small, but is certainly still doable. The more data the better, but try something in the 2MB-5MB range first and see how it does before using larger datasets and potentially waiting multiple days for training to complete. You might be satisfied with the smaller-scale performance.
- If you increase the dataset size you should increase the size of the neural network, done in two ways: number of layers and network size. 
  - Layers: The default is 2 layers. The author does not suggest using more than 3 layers. 
  - Size: The default is 128. With more than 2MB, you should increase the size and your network should work significantly better. I used size=250 for 3MB and got pretty good performance. The author says that with 6MB you can increase the size to 300 or more. 
- The size of the model vs size of the dataset comes down to trial and error. It depends what your data looks like (Shakespearean language or highly regular recipe formats), how much time you're willing to spend training your network, and what you think is satisfactory performance. It's better to err on the side of a model that's too big, so for your first try, I suggest you do something on the rough scale of
  - 1MB, 2 layers, size 128
  - 2MB, 2 layers, size 200
  - 3MB, 2 layers, size 250
  - 6MB, 2 layers, size 400
And see how it goes. After that, you can refer to the authors "Tips and Tricks" section and play with network type, dropout rate, layers, etc.

### Step #7: Training

To train, open the docker terminal window and type

th train.lua -input_h5 data/FILENAME.h5 -input_json data/FILENAME.json -gpu -1 -rnn_size 200 -num_layers 2

- "-gpu -1" If you don’t have a CUDA compatible GPU (you’d probably know if you did) or don’t know what that is, just add -gpu -1. If you do have a GPU, then lucky you: you should definitely omit -gpu -1 to get training to run a whole lot faster. If not, don't worry about it, it's just a speed boost.
- "-rnn_size 200" means we've selected a size of 200. 128 is the default.
- "-num_layers 2" just means we've selected 2 layers. 2 layers is the default, so if you want 2 layers, you can alternatively omit this.

Now you're going to see this for a while.



The program is going to run 50 epochs, stopping occassionally to compute the cross-validation loss. If you're interested, take a look at the original repository: these numbers can tell you how your model is performing and whether you need to increase/decrease model size on your next run.

### Step #8: Sampling

Eventually your model will finish training and the prompt will reappear.

Now type 

th sample.lua -checkpoint cv/checkpoint_10000.t7 -length 2000 -gpu -1

Where "-lenth 2000" is the size of the sample you're taking. This generates 2000 characters and only takes a few seconds. Feel free to increase this.

That's it. You trained a recurrent neural network and generated samples from it. Congratulations! 

