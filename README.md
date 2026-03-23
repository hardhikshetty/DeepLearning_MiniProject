# Deep Learning Mini-Project

A practical deep learning project implementing **CNNs, Transfer Learning (ResNet18), RNN/LSTM/GRU, and GANs** for tasks in image classification, sentiment analysis, and image generation.  

This repository provides clean, modular code for training, evaluation, and visualization of results, designed for hands-on learning and experimentation.


## Features

- **CNN on CIFAR-10:** Image classification with convolutional layers, pooling, and dropout.  
- **Transfer Learning (ResNet18):** Fine-tune a pretrained ResNet18 on a subset of CIFAR-10.  
- **RNN / LSTM / GRU on IMDB:** Sentiment classification on movie reviews using sequential models.  
- **GANs (MNIST & Fashion-MNIST):** Generate realistic handwritten digits and fashion images

## Repository Structure
DL_PROJECT/
├── CNN/
│ ├── simplecnn.py
│ ├── traincnn.py
│ └── transfermodel.py
├── RNN/
│ ├── rnnmodels.py
│ └── trainrnn.py
├── GAN/
│ ├── ganmodels.py
│ └── traingan.py
└── OUTPUTS/
├── plots/
└── generated_images/

##Requirements

- Python 3.10+  
- PyTorch, Torchvision, Torchtext  
- NumPy, Matplotlib, Seaborn  
- scikit-learn  

Install dependencies via pip:

pip install torch torchvision torchtext numpy matplotlib seaborn scikit-learn

-->How to Run:
CNN:
cd CNN
python traincnn.py

Transfer Learning:
cd CNN
python transfermodel.py

RNN / LSTM / GRU:
cd RNN
python trainrnn.py

GANs:
cd GAN
python traingan.py

Results:
Model	Dataset	Accuracy :
CNN	CIFAR-10- ~73%
ResNet18	CIFAR-10- ~87%
LSTM	IMDB-	~86%
GRU	IMDB-	~85%

GAN (MNIST)	MNIST	Progressive improvement in image quality
GAN (Fashion)	Fashion-MNIST	Progressive improvement in image quality
Training loss and confusion matrices are saved in OUTPUTS/plots.
GAN-generated images are saved in OUTPUTS/generated_images.

 Key Highlights:
 
Modular code for different deep learning architectures.
Includes preprocessing, training loops, evaluation, and visualization.
Designed for educational use, showcasing deep learning workflows.
Efficient training with fewer epochs while maintaining good accuracy.

Author

Hardhik Shetty
USN: NNM23IS075
NMAM Institute of Technology
