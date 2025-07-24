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


## Cat Breed Classifier (Dataset https://www.kaggle.com/datasets/nikolasgegenava/cat-breeds)

I went back to the Cat Breed dataset with 66 classes. To enable this run fine on my hardware, i manually removed 50 of these classes, using stepwise selection, twice over the classes.

Resnet 50 was to heavy and slow for my hardware from the last time, so i went for the lesser Restnet34 this time, also from torchvision. I applied KFold cross-validation, to test the model on every part of the data, rather than one. I split the whole data into five parts, and trained for six epochs each.

Prior to that, i changed the final fully connected layers from Resnet34, from its out_feature of 1000 to suit my 16 classes. After which it had 135,440 trainable parameters out of 21,420,112.
 
There was fine generalisation, with loss decreasing across epochs. Validation loss was used as the tracker during the training. 

There was a minor fluctuation in validation accuracy, from 59% to 68%, but overall upward trend.


## Fine-Tuning YOLO To Detect and Classify Road Signs
(https://universe.roboflow.com/costom-s6gdy/road-signs-c6yoa)

I began by exploring Roboflow, an amazing place and wonder how i didnt know about the platform until TODAY. However, i downloaded a dataset on traffic signs. Roboflow allows the addition of augmentation and preprocessing which i found really great. For this preprocessing this dataset, i resized all images to 224x224, Auto-Adjusted Contrast using histogram equalization, and auto applied auto-orientation. Augmentations added include duplicating each sample for clockwise, counter-clockwise 90 degrees rotation, rotation between -15° and +15° and hue between -15° and +15°.

I started out locally but moved to Google Colab midway to harness the power of GPU. I moved on there, and loaded Yolov5, by cloning the Ultralytics Yolov5 repo on github. Then i set up the dataset folder and wrote a yaml file (where i added custom classes as YOLO isn't trained to detect road signs) for the training.

Finally i trained the model on the road signs dataset for 50 epochs with 16 batches. Training lasted just about half an hour. Though i had to do this thrice due to runtime disconnects. I came across the mAP scoring metric, which stands for mean Average Precision. It tells you how good the model is at both locating and correctly classifying objects. AP (Average Precision) is calculated for each class, and mAP is the average of all the APs across all classes. Usually, mAP@0.5 which translates to IoU threshold is 0.5 (fairly loose match) and mAP@0.5:0.95, stricter, averaged across IoUs from 0.5 to 0.95. As mentioned two days ago, IoU is a metric used to evaluate how well the predicted bounding box (model's detection/mapping) overlaps with the ground truth box (the actual object on the image). 

The mAP@0.5 and mAP@0.5:0.95 scores are 92% and 73% respectively. Other metrics; precision of 88% and recall of 90.5%. The full results are captured below, along with some tests. These metrics show the model’s ability to detect signs accurately and consistently, with particularly strong performance on sign class like "Give Way", "School Ahead", and "Go Slow". Some classes like "Narrow Bridge Ahead", however, had slightly lower performance.

## Face Recognition with MTCNN
This project started with a simple goal: pull a YouTube video and run face recognition on it. I picked one of MrBeast’s videos where Neymar featured, clipped the first minute, and built a system that could tell if MrBeast, Neymar, neither, or both were in each frame.

I loaded the saved reference embeddings I generated yesterday and added the comparison logic. For each new face, MTCNN took care of detection and cropping, then InceptionResnetV1 extracted a 512 dimensional embedding, then compared its similarity to the saved embeddings.

If the similarity crossed 0.9, it got tagged as a known face. If not, it was labelled “Unknown.”

Overall, it worked decently. But one unknown face was misclassified as MrBeast. I only used five images of him to extract his faceprints, so I’m guessing the average embedding isn’t diverse enough to represent him properly. Perhaps more varied reference images would help improve this.

Still, it’s satisfying seeing the full system in action from start to finish.
