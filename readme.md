# Welcome To my Deep Learning Repo!
In here contains my deep learning works from my 50-day deep learning self-challenge!

## CNN with Poker Cards
Per my researched methods of preventing overfitting in CNNs, i added augmentations to images in the transform pipeline, L2 regularisation to the optimizer and a dropout layer before the final output layer.

What these do;
Augmentations: The images are randomly flipped horizontally, randomly rotated by plus or minus (range of) ten degrees, and lastly random cropping of the images covering 80%-100% of its original area while resizing it 224x224. This can help the model generalize better.

L2 regularisation: Added to the optimiser, using weight_decay argument. This penalises large weights to reduce overfitting.

The dropout layer is added in the architecture, within the last two Linear layers. It randomly sets a fraction of the neurons to 0 during training, forcing the model to not rely too heavily on specific neurons, thereby improving generalisation. 

This time i trained the model first for 25 epochs, with resulted yet in overfitting. I restarted and trained again for 15 epochs only this time. The training logs are captured below.

I will build yet another CNN model using a different data. This time experimenting with early stopping and callbacks.

## I sought a dataset that wouldn't overwhelm my hardware, and i came across Sugarcane Leaf Disease Dataset on kaggle 👉 https://lnkd.in/dP8R8rvJ. 
A manually collected image dataset of sugarcane leaf disease. It has mainly five main categories in it. Healthy, Mosaic, Redrot, Rust, and Yellow disease. The dataset has been captured with smartphones of various configurations to maintain the diversity. It contains a total of 2569 images including all categories. This database has been collected in Maharashtra, India. The database is balanced and contains good variety. The image sizes are not constant as they originate from various capturing devices. All images are in RGB format.

I made a three layer architecture and carried out training for eight epochs. Resulted in overfitting. So i went back and added augmentations in my transform pipeline and trained again, for eight epochs. The metrics were much better with train accuracy at 75% and validation accuracy at 74%. Full log is captured below. Feels good not to overfit. However testing shows the model generalises well on every class but consistently misclassifies all Rust images. This could be due to insufficient features or feature overlap.

I will explore, again, callbacks and checkpointing.

## Brain Cancer - MRI Dataset from 👉 https://lnkd.in/d8C2jTGz

The Bangladesh Brain Cancer MRI Dataset is a comprehensive collection of MRI images categorized into three distinct classes:
Brain_Glioma: 2004 images
Brain_Menin: 2004 images
Brain Tumor: 2048 images
The dataset includes a total of 6056 images, uniformly resized to 512x512 pixels. These images were collected from various hospitals across Bangladesh with the direct involvement of experienced medical professionals to ensure accuracy and relevance. This dataset is valuable due to the difficulty in obtaining such medical imaging data and offers a reliable resource for developing and testing diagnostic tools.

I experimented with callbacks on the CNN model i built to classify and detect brain cancer from MRI scans. This time i added an extra layer to the usual three layers, and added augmentations to my transformation pipeline, such as random vertical and horizontal flips on the images. I set patience for the early stopping to be three epochs and set it to train for ten epochs. Overfit.

I went back again, reducing the learning rate, removed a layer, increased dropout, while adding more augmentations. I also tried a new scheduler; the ReduceLROnPlateau scheduler, which reduces learning rate if validation loss stalls. This whole training process/fine tuning was carried out/repeated ~five times.

This time, the early stopping was triggered after the fifth epoch, with training accuracy at 67% and validation accuracy at 53%. More can and will be done to improve this peak and drop of the validation accuracy.
