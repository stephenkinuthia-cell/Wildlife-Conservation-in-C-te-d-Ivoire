# 🌿 Wildlife Conservation in Côte d'Ivoire

## 📜 Project Overview
This project focuses on applying deep learning to wildlife conservation efforts in Côte d'Ivoire. Leveraging a real-world dataset from a data science competition on [DrivenData.org](https://www.drivendata.org/), I developed neural network models to classify wildlife captured by camera traps. This work contributes to automating animal recognition and monitoring in protected habitats.

Using PyTorch, I explored image preprocessing, tensor manipulation, binary classification, and built a convolutional neural network (CNN) for multiclass classification. By the end of the project, I was able to generate accurate predictions for eight wildlife categories and format them for competition submission.

---

## 📚 Lessons

### ⬆️ Lesson 1.1: Image as Data

**Summary:**
I explored how images are stored as data and manipulated tensors using PyTorch. I also downloaded and visualized the wildlife dataset to understand the structure and features of the images.

**Objectives:**
- Explore tensor attributes: shape, data type, and device
- Perform slicing and mathematical operations on tensors
- Load and decompress the image dataset
- Use PIL to load images and explore color channels

**New Terms:**
- **Tensor**: A multi-dimensional array used to represent data in PyTorch.
- **Attribute**: A property of an object in Python (e.g., `tensor.shape`).
- **Class**: A Python blueprint for creating objects.
- **Color Channel**: Component of an image that holds intensity for red, green, or blue.
- **Method**: A function defined within a class.

---

### ⬆️ Lesson 1.2: Fix My Code

**Summary:**
In this debugging-focused lesson, I learned to read and interpret Python tracebacks. These stack traces are critical for identifying and fixing coding errors effectively.

**Objectives:**
- Understand what Python tracebacks are and how they help in debugging
- Trace the source and type of exceptions in Python
- Improve coding efficiency by locating and resolving errors accurately

**New Terms:**
- **Traceback**: A detailed error report showing the execution path leading to an exception.
- **Exception**: An error that disrupts the normal flow of program execution.
- **Stack Trace**: A report of the active stack frames at a certain point in time during program execution.

---

### ⬆️ Lesson 1.3: Binary Classification

**Summary:**
Here, I built my first neural network to perform binary classification on the wildlife dataset—determining whether an image contains a hog or not. The model was trained in PyTorch and saved for future use.

**Objectives:**
- Convert grayscale images to RGB
- Resize and standardize images with a transformation pipeline
- Build and train a simple feedforward neural network
- Save the trained model to disk

**New Terms:**
- **Binary Classification**: Classifying inputs into one of two classes.
- **Activation Function**: A function that introduces non-linearity in a neural network (e.g., ReLU).
- **Backpropagation**: The algorithm for updating weights in a neural network by propagating error backwards.
- **Cross-Entropy**: A loss function often used for classification tasks.
- **Epoch**: One full pass through the training dataset.
- **Layers**: Different levels in a neural network (input, hidden, output).
- **Logits**: Raw model outputs before applying activation functions like softmax.
- **Optimizer**: An algorithm (e.g., SGD, Adam) that adjusts model weights to minimize loss.

---

### ⬆️ Lesson 1.4: Multiclass Classification

**Summary:**
I extended the binary classifier to a multiclass Convolutional Neural Network (CNN) to identify eight possible image classes. This model provided competition-ready predictions.

**Objectives:**
- Load a multiclass image dataset
- Normalize images to enhance model performance
- Build and train a CNN suitable for image classification
- Format predictions according to competition submission standards

**New Terms:**
- **Multiclass Classification**: Predicting one label from more than two possible categories.
- **Normalize**: Scaling input values (usually pixel values) to a standard range.
- **Convolution**: A mathematical operation that filters input data using a kernel to detect patterns.
- **Max Pooling**: Downsampling technique to reduce spatial dimensions and computational load.
- **CNN (Convolutional Neural Network)**: A specialized neural network designed for processing grid-like data, such as images.

---

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📩 Contact
**Stephen Kinuthia**  
📧 Email: [kinuthiastephen94@gmail.com](mailto:kinuthiastephen94@gmail.com)  
🌐 GitHub: [github.com/stephenkinuthia-cell](https://github.com/stephenkinuthia-cell)

Feel free to connect for collaboration, feedback, or discussions related to deep learning and computer vision projects.

