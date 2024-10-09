 

# ✨Introduction to Image Classification using Convolutional Neural Networks (CNNs)


Deep Learning is one of the most fascinating fields of Artificial Intelligence, especially because it adds another dimension of Multimodality to learn from the World around us. As we probably have seen before, there are many different ways of studying images'content, and today we will delve even deeper with the interesting concepts of Convolutional Neural Networks alias CNNs.

## Prerequisites
As image data are considerably voluminous, sourcing them and processing them usually require consequent processing power and energy consumption. We may collect images from the Internet thanks to Big Data, social media, streaming multimedia content, movies, photography, etc., hence the need for parallel-computing, which leads us to the extensive usage of TPUs and GPUs.

This added sophistication is consequently resource-intensive, therefore we choose to setup our environment around Keras and Tensorflow which are popular powerful libraries for Image Processing.

[In some projects, we may encounter uses of other libraries such as Theano, Scikit-learn (image) and Caffe...and other ecosystems we will briefly mention later.]

## Image Processing Techniques
There are many Image Processing Techniques we can perform, just by mastering Convolutional Neural Networks basics, before gradually embracing different levels of complexity. We can cite for instance:

- ✔ Image Enhancement: to improve image quality by reducing blur or adjusting standard parameters such as brightness, contrast, and sharpness,
- ✔ Image Restoration: to restore degraded images (antique black and white photos for eg.) using methods like inverse filtering, 🎞
- ✔ Image Segmentation: to divide an image into "semantic/meaningful" parts according to the pixel disrtibution, using techniques like thresholding and edge detection, (eg. self-driving cars),
- ✔ Image Compression: to reduce the file size through lossy or lossless methods,
- ✔ Image Generation: it used GANs (Generative Adversarial Networks) to create new image or enhance existing ones,
- ✔ Morphological Processing: Alter image structures based on shape.



## Lesson Overview :pencil2:
Without further ado, let's explore the essential concepts behind image classification using Convolutional Neural Networks (CNNs), a key element in the field of computer vision. For now, we’ll break down how CNNs work, focusing on critical components such as:
* Convolution, 
* Filters, 
* and Pooling layers. 

Additionally, we’ll look at practical applications of CNNs in tasks like object detection and image classification, providing a clear understanding of their significance in real-world scenarios.
Next, we may experiment with other flexible frameworks such as AutoKeras to handle various formats or automate model selections, while also exploring techniques for manipulating CNNs in tasks such as image regression for quality assessment, grading items, or estimating age from facial features.

## Learning Objectives :notebook:
By the end of this lesson, we will be able to:

• Explain the basic structure and functioning of Convolutional Neural Networks (CNNs).
• Describe the role of convolution, filters, and pooling layers in CNNs.
• Identify key use cases for CNNs in image classification and object detection.
• Apply the concepts learned to analyze a simple CNN architecture.
• Discuss the advantages and limitations of using CNNs for image classification tasks.

## Key Definitions and Examples :key:


### Convolutional Neural Networks (CNNs) Definition
Convolutional Neural Networks (CNNs) are a type of deep learning model specifically designed to process structured data like images. They are advanced machine learning vision models, we often encounter beside vision transformers (ViTs). CNNs leverage a mathematical operation called convolution, which allows them to automatically identify patterns and features within images through multiple processing layers.

### Convolutional Neural Networks (CNNs) Example
Think of an image as a 2D matrix filled with pixel values. A CNN applies a filter (or kernel) over this matrix to perform convolution, extracting features such as edges or textures. For instance, when a 3x3 filter is applied to an image, it slides across the matrix, performing element-wise multiplication and summing the results to generate a feature map that highlights specific characteristics.

-**Convolution Definition:**
Convolution is a mathematical operation that combines two functions to create a third function, illustrating how one function modifies another. In CNNs, convolution involves sliding a filter over an input image to compute feature maps.

-**Convolution Example:**
For example, if we have an input image with pixel values and a 3x3 filter designed to detect vertical edges, the convolution operation will emphasize areas in the image where there are significant changes in pixel intensity vertically. This results in a new feature map that indicates where vertical edges are present.

-**Pooling Layers Definition:**
Pooling layers are integral to CNNs as they reduce the spatial dimensions of feature maps while preserving important information. This down-sampling process helps lower computational demands and mitigate overfitting by summarizing features detected by previous layers.

-**Pooling Layers Example:**
A common pooling method is Max Pooling, which selects the maximum value from each patch of the feature map defined by a specified window size (e.g., 2x2). For instance, if we apply Max Pooling on a 4x4 feature map using a 2x2 window, we would reduce it to a 2x2 matrix by taking the maximum value from each 2x2 region.



### Tools and Ecosystems:
  Google TensorFlow and Keras are the most used frameworks for building CNNs, in image classification and object detection use cases. But there are also other popular libraries:
  - Pytorch: prefferred for its dynamic computation graph, useful for image segmentation and object detection with CNN architecture like ResNet,
  - KNIME Analytics Platform: a code-free environment to implement CNN for image classification tasksfor deriving useful information from images.

    
## Additional Resources :clipboard:
If you would like to study these concepts before the class or would benefit from some remedial studying, please utilize the resources below:
- [Deep Learning for Computer Vision with Python](https://www.pyimagesearch.com/deep-learning-computer-vision-python-book)
- [Stanford CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [Deep Learning for Computer Vision](
http://introtodeeplearning.com/2019/materials/2019_6S191_L3.pdf)
- [Andrew Ng Notes on CNNs by Ashish Patel](https://github.com/ashishpatel26/Andrew-NG-Notes/blob/master/andrewng-p-4-convolutional-neural-network.md/)












