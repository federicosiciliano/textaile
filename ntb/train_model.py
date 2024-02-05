#!/usr/bin/env python
# coding: utf-8

# # Text - AI - le
# 
# Description...
# 
# **The goal is to ...**
# 
# Description...
# 
# *Researchers*: Ludovica Schaerf, Federico Siciliano

# ## Preparation phase
# 
# 
# 1.   Installation from GitHub
# 2.   Connection to Google Drive
# 3.   Paths definition and imports
# 
# There are 2 possibilities for using this notebook:
# 
# *   Run locally
# 
# In this case we have 2 chances:
# 
# 1.   Download through pip install from GitHub, then import it using import keyword and use it.
# 2.   Download the code locally from GitHub and from the notebook define the paths to these sources (code to be adjusted).
# 
# *   Run in Colab
# 
# 1.   Download through pip install from GitHub, then import it using import keyword and use it (HERE).
# 2.    Download the code locally from GitHub into the src folder in Google Drive and from the notebook define the paths to these sources (code to be adjusted).
# 
# 
# 
# 

# ### Connection to Google Drive
# The code is a Python snippet designed to connect Google Drive to a Google Colab notebook. It checks if connect_to_drive is set to True, and if so, it imports the necessary library and mounts Google Drive to the /content/gdrive directory using Google Colab's drive.mount() function. The user will be prompted to authorize the connection through a popup window.

# In[1]:


# Set connect_to_drive to True if you want to connect to Google Drive.
connect_to_drive = False

# If connect_to_drive is True, connect to Google Drive using the google.colab library.
# This will prompt a popup window where you'll need to authorize access to your Google Drive.
if connect_to_drive:
    from google.colab import drive

    # Mount the Google Drive to the '/content/gdrive' directory in the Colab environment.
    # The 'force_remount=True' parameter ensures that if the Drive is already mounted,
    # it will be remounted to refresh the connection.
    drive.mount('/content/gdrive', force_remount=True)


# ### Installation of packages
# This code block installs Python packages and libraries using the pip package manager. It checks the value of the connect_to_drive variable, and if it is True, the code assumes that the user wants to install the specified packages (otherwise commands assumed done via console manually). Here's a breakdown of the installation process:
# 
# 1.   pip install git+https://github.com/siciliano-diag/easy_lightning.git@rec_utils: This command installs Python packages called data_utils, exp_utils and torch_utils.
# 2.   pip install pytorch_lightning: Finally, this command installs the pytorch_lightning library, which is a popular and easy-to-use PyTorch wrapper that simplifies the process of training deep learning models.
# 
# Overall, this code is meant to set up the required dependencies for a specific project, and by setting connect_to_drive to True, the user can conveniently install these packages in their environment. Note that the installation process might take a few moments to complete depending on the network speed and the size of the packages.

# In[2]:


if connect_to_drive:
    #Install FS code
    get_ipython().system('pip install git+https://github.com/siciliano-diag/easy_lightning.git@rec_utils')

    get_ipython().system('pip install pytorch_lightning')


# ### Imports

# The script imports the following libraries and modules:
# 
# - numpy (as np): Used for numerical computations, particularly array manipulation and mathematical operations.
# - pandas (as pd): Used for data manipulation and analysis, providing powerful data structures and tools.
# - matplotlib.pyplot (as plt): A sub-library of Matplotlib, used for creating various types of data visualizations, such as plots and charts.
# - os: Allows interaction with the operating system, such as working with files, directories, and environment variables.
# - sys: Provides access to system-specific parameters and functions, enabling control over the Python runtime environment.
# - torch: The core library for PyTorch, which is a widely used deep learning framework in Python for building and training neural networks.
# - The two commented-out import statements, from copy import deepcopy and import pickle, suggest that these functionalities might have been used in the past but are currently not being utilized in the script.

# In[3]:


# Put all imports here

# Import NumPy library, commonly used for numerical computations and array manipulation.
import numpy as np

# Import pandas library, widely used for data manipulation and analysis.
import pandas as pd

# Import matplotlib.pyplot from Matplotlib, used for creating data visualizations and plotting.
import matplotlib.pyplot as plt

# The following imports are commented out and not currently in use, but they are left for reference:
# #from copy import deepcopy: This import statement would bring in the 'deepcopy' function from the 'copy' module,
# which allows creating a deep copy of objects to avoid modifying the original data accidentally.

# #import pickle: This import statement would allow working with the pickle module, which is used for
# serializing and deserializing Python objects, i.e., converting objects to a byte stream and vice versa.

# Import os module, used for interacting with the operating system, such as managing files and directories.
import os

# Import sys module, used to access system-specific parameters and functions.
import sys

# Import torch library, which is the main library for working with PyTorch, a popular deep learning framework.
import torch
import pytorch_lightning as pl


# ### Definition of paths
# This is to define the paths to store data, configurations, plots, models and results.
# 
# Locally if Google Drive is not connected.

# In[4]:


# Define the project folder path and set it to the parent directory of the current location.
project_folder = "../" # Used if the notebook is run locally to define the right paths

# If connect_to_drive is True, update the project_folder to point to the specific folder in Google Drive.
if connect_to_drive:
    project_folder = "/content/gdrive/Shareddrives/TextAIle" #Name of Shared Drive folder
    #project_folder = "/content/gdrive/MyDrive/<MyDriveName>" #Name of MyDrive folder

# The cfg_folder will contain hyperparameter configurations.
# It is located inside the project_folder.
cfg_folder = os.path.join(project_folder, "cfg")

# The data_folder will contain raw and preprocessed data.
# It is also located inside the project_folder.
data_folder = os.path.join(project_folder, "data")

# The raw_data_folder will contain the raw data.
# It is a subfolder within the data_folder.
raw_data_folder = os.path.join(data_folder, "raw")

# The processed_data_folder will contain the preprocessed data.
# It is another subfolder within the data_folder.
processed_data_folder = os.path.join(data_folder, "processed")

# The source_folder will contain all essential source code.
# It is located inside the project_folder.
source_folder = os.path.join(project_folder, "src")

# The out_folder will contain all outputs, such as models, results, plots, etc.
# It is also located inside the project_folder.
out_folder = os.path.join(project_folder, "out")


# ### Packages:
# 
# data_utils: A package that provides utilities and functions for working with data, possibly for data preprocessing, augmentation, or data loading.
# 
# exp_utils: A package that contains utilities and tools for managing and organizing machine learning experiments, such as logging experiment results and managing experiment configurations.
# 
# torch_utils: A package that likely provides utility functions and classes for working with PyTorch, a popular deep learning library.
# 
# pytorch_lightning: A separate library that simplifies the process of training PyTorch models by abstracting away boilerplate code and providing useful features for distributed training, GPU acceleration, and more.

# In[5]:


# Importing all packages after GitHub download
import data_utils, exp_utils, torch_utils


# ### Import local code from the Drive

# This code snippet deals with importing modules from a custom source folder, rather than installing packages via pip install from external sources like GitHub as before.
# 
# *Differences from installing via pip install from GitHub:*
# 
#   When you use pip install to install packages from GitHub or any other source, it installs the package and its dependencies globally or within a virtual environment. The installed package becomes part of Python's standard search path, and you can import the package from anywhere in your code without explicitly manipulating sys.path. The installed package can also be accessed by other projects or scripts running in the same Python environment.
# 
#   On the other hand, the code in the provided snippet deals with importing modules from a custom source folder that might not be part of the global Python path. It allows importing specific modules from the project_folder without a formal installation step. This approach can be useful during development when you want to work with local code changes and test them without the need for package installation and updates. However, it may require more manual management and is typically used for custom development purposes rather than using pre-packaged libraries from external sources.

# In[6]:


# To import from src of Google drive a utils library (now not present)

# Attach the source folder to the start of sys.path
sys.path.insert(0, project_folder)
print(sys.path)  # View the path and verify

# Import from src directory: to be used if we want to import utilitis from src folder
# from src import utils

# Change the current working directory to source_folder
os.chdir(source_folder)

from src import utils


# ## Main
# 1.   Load configuration
# 2.   Data preparation
# 3.   Model
# 4.   Experiment check
# 5.   Training
# 6.   Test
# 7.   Save experiment

# ### Load configuration
# The code loads a configuration using the 'load_configuration' function from the 'exp_utils.cfg' module. The loaded configuration object is stored in the 'cfg' variable.
# 
# *An experiment with a name ("experiment") can be used for different configurations, when a change is not affecting the configuration parameters but it is a code-based change, then create a new experiment name*.

# In[7]:


# Load the configuration using the 'load_configuration' function from the 'exp_utils.cfg' module
cfg = exp_utils.cfg.load_configuration()

cfg  # Display the loaded configuration object


# ### Data Preparation and Label Noise Injection
# 
# This code segment carries out the following steps:
# 
# 1. **Configuration Update:** It updates the "data_folder" configuration in the `cfg` dictionary with the value of the 'processed_data_folder' variable.
# 
# 2. **Data Loading:** The code loads data using the configuration settings from `'cfg["data"]'`.
# 
# 3. **Train Data Shape:** It displays the shape of the 'train_x' data array, representing the input training images.
# 
# 4. **Data Loader Preparation:** After loading data and potentially injecting label noise, the code prepares data loaders for training. These loaders are created based on the loaded data and loader parameters from 'cfg["model"]["loader_params"]'.
# 
# 5. **Example Image Display:** The code extracts an example image with a shape of (3, 32, 32) from the training data. The example can be changed by modifying the `example_index` variable. It also displays the image in a (32, 32, 3) format using `matplotlib.pyplot.imshow()`, with the title showing "3x32x32 Image" and the axes turned off for better visualization.
# 

# In[8]:


# Update the "data_folder" configuration value with the 'data_folder' variable
cfg["data"]["data_folder"] = processed_data_folder

# Load data using the configuration settings from 'cfg["data"]'
data = data_utils.data.simple_load_data(**cfg["data"])

data["y"] = data["y"].to_numpy()
data = data_utils.utils.sort_by_column(data, column_dict = {"y":0})
data = data_utils.utils.separate_columns(data, separate_keys = {"y": ["image_id"]}, column_ids = {"image_id": [0]})
data["y"] = data["y"].astype(np.float32)
data["y"] = np.nan_to_num(data['y'])

data = data_utils.utils.transpose(data, swap_dict = {"x": [0,3,1,2]}) # move channel dimension from the end to the second position

concept_ranges = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                           [360,100,100,360,100,100,360,100,100,360,100,100,360,100,100,1,1,1,1,1,1]])
# concept_ranges = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#                            [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]])

data = data_utils.utils.min_max_scale(data, scale_dict={"x": (0,255),
                                                        "y": concept_ranges})

#data = data_utils.utils.mean_std_scale(data, scale_dict={"x": [np.array([[123.68, 116.779, 103.939]]), np.array([[58.393, 57.12, 57.375]])]})

# For each color, get the bin index
all_colors = data["y"][:,[[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14]]].reshape(-1,3)
distr, bins = np.histogramdd(all_colors, bins=np.array([[0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01 ],
                                                        [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01 ],
                                                        [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01 ]]), density=True)

bin_idx = np.digitize(data["y"][:,[[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13,14]]], bins=np.array([0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01 ]))-1

weights = np.zeros(len(bin_idx))
for i in range(len(bin_idx)):
    all_distr = []
    for r,g,b in bin_idx[i]:
        all_distr.append(distr[r,g,b])
    weights[i] = 1/np.prod(all_distr)**(1/5)
data["weights"] = weights

data = data_utils.data.split_data(data, **cfg["data"])

# Print the shape of the 'train_x' data and 'test_x' data
print("Shape of train data: ", data['train_x'].shape)
print("Shape of val data: ", data['val_x'].shape)
print("Shape of test data: ", data['test_x'].shape)


# In[9]:


# Prepare data loaders for training using the loaded data and specified loader parameters
loaders = torch_utils.preparation.prepare_data_loaders(data, **cfg["model"]["loader_params"], split_keys={"train": ["train_x", "train_y", "train_weights"],"val": ["val_x", "val_y", "val_weights"],"test": ["test_x", "test_y", "test_weights"] })

# Extract the example image with shape (3, 32, 32)
# example_index = 0  # You can change this to view different examples
# #original_one_hot_label = data['train_y'][example_index]

# # Print the shape of the example image (optional)
# #print("Shape of the printable image:", example_image_3x32x32.shape)

# # Display the 3x32x32 image
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plt.imshow(data['train_x'][example_index].transpose(1,2,0))  # Display as (32, 32, 3)
# plt.title("3x32x32 Image")
# plt.axis('off')

# # Show both displayed images
# plt.tight_layout()
# plt.show()


# In[10]:


# for x,y in loaders["test"]: break


# In[11]:


# # Extract the example image with shape (3, 32, 32)
# example_index = 12  # You can change this to view different examples

# # Display the 3x32x32 image
# fig, axs = plt.subplots(1, 6, figsize=(18, 3))
# axs[0].imshow(x[example_index].detach().numpy().transpose(1,2,0))  # Display as (32, 32, 3)
# axs[0].set_title("3x32x32 Image")
# axs[0].axis('off')

# # Display colors in y
# for i in range(5):
#     axs[i+1].imshow([[tuple(y[example_index].detach().numpy()[i*3:i*3+3])]])
#     axs[i+1].set_title(f"Color {i+1}")
#     axs[i+1].axis('off')
    
# plt.tight_layout()
# plt.show()


# ### Model parameters
# This code snippet manages the configuration and initialization of the neural network model:
# 
# The first line updates the "in_channels" value in the model configuration. This update ensures that the number of input channels matches the number of channels in the training images. It aligns the model's input layer with the image data's channel dimension.
# 
# The second line updates the "out_features" value in the model configuration. This update ensures that the number of output features (classes) matches the number of classes in the training labels. It ensures the model's output layer aligns with the classes in the data.

# In[12]:


# Model type 1

# Update the number of input channels in the model configuration based on the number of channels in the training data
cfg["model"]["encoder"]["in_channels"] = cfg["model"]["decoder"]["in_channels"] = data["train_x"].shape[1]
cfg["model"]["encoder"]["out_features"] = cfg["model"]["decoder"]["out_features"] = cfg["model"]["embedding_size"] + data["train_y"].shape[1]

# Update the number of output features (classes) in the model configuration based on the number of features in the training labels

for example_img, example_concepts, example_weight in loaders['train']: break

encoder_module = torch_utils.model.get_torchvision_model(**cfg["model.encoder"])
decoder_module = torch_utils.model.get_torchvision_model_as_decoder(example_img[:1], **cfg["model.decoder"]) #creates sort of an inverted model

model_params = cfg["model"].copy()
model_params["encoder"] = encoder_module
model_params["decoder"] = decoder_module
model_params["concept_size"] = data["train_y"].shape[1]
#model_params["encoder_features"] = encoder_module.classifier[1].out_features #efficientnet
model_params["encoder_features"] = encoder_module.fc.out_features #RESNET
model_params["decoder_features"] = decoder_module[0].in_features
main_module = utils.TextAIle(**model_params)

# for param in encoder_module.parameters():
#     param.requires_grad = False


# In[13]:


# # Model type 2: custom Unet

# model_params = cfg["model"].copy()
# model_params["n_channels"] = data["train_x"].shape[1]
# model_params["concept_size"] = data["train_y"].shape[1]
# #main_module = utils.TextAIleUNet(**model_params)
# main_module = utils.CustomTextAIle(**model_params)

# # for param in encoder_module.parameters():
# #     param.requires_grad = False


# In[14]:


# # Only Resnet

# cfg["model"]["encoder"]["in_channels"] = data["train_x"].shape[1]
# cfg["model"]["encoder"]["out_features"] = data["train_y"].shape[1]

# main_module = torch_utils.model.get_torchvision_model(**cfg["model.encoder"])


# In[15]:


# If we want to visualize the complete structure
main_module


# In[16]:


# # If we want to visualize shapes per layer
# # inp = example_img[:1] #encoder_module(example_img[:1])
# # print("ENCODER")
# # for layer_name, layer in encoder_module.named_children():
# #     print(layer_name, layer)
# #     inp = layer(inp)
# #     print(inp.shape)
# # print(inp.shape)

# inp = encoder_module(example_img[:1])

# print("DECODER")
# for layer_name, layer in decoder_module.named_children():
#     print(layer_name, layer)
#     inp = layer(inp)
#     print(inp.shape)
# print(inp.shape)


# In[17]:


# # Visualize original image, reconstructed image, true and predicted concepts
# example_rec_img, example_pred_concepts, example_emb = main_module(example_img)
# fig, ax = plt.subplots(len(example_img), 4, figsize=(12, 3 * len(example_img)))
# for i, (img, rec_img, true_concepts, pred_concepts) in enumerate(zip(example_img, example_rec_img, example_concepts, example_pred_concepts)):
#     ax[i, 0].imshow(img.detach().cpu().numpy().transpose(1,2,0), )
#     ax[i, 0].set_title("Original")
#     ax[i, 1].imshow(rec_img.detach().cpu().numpy().transpose(1,2,0))
#     ax[i, 1].set_title("Reconstructed")
#     ax[i, 2].bar(np.arange(len(true_concepts)), true_concepts.detach().cpu().numpy())
#     ax[i, 2].set_title("True Concepts")
#     ax[i, 3].bar(np.arange(len(pred_concepts)), pred_concepts.detach().cpu().numpy())
#     ax[i, 3].set_title("Predicted Concepts")
# # !!! HERE THE MODEL IS NOT TRAINED YET, SO THE PREDICTION IS RANDOM !!!


# ### Check experiment
# 
# Check if experiment has already been done, if so, load its ID

# In[18]:


#for _ in cfg.sweep("model.loss.__weight__"):
exp_found, experiment_id = exp_utils.exp.get_set_experiment_id(cfg)
print("Experiment already found:", exp_found, "----> The experiment id is:", experiment_id)


# ### Training
# The training procedure emphasizing how it is necessary to place the objects within the cfg configuration and then use them to create the trainer, using a new variable to retain both the YAML configuration with values and the one with the objects in place.

# Here we are going to define:
# 
# 
# 1.   Callbacks
# 
#   *   Early Stopping
#   *   Model Checkpoint
# 
# 
# 2.   Logger
#   *   CSV Logger
# 
# Then we are going to define, according to the actual configuration:
# 
# 
# 1.   Loss Function
# 
# 
# 2.   Optimizer
# 
# After, let's create the final model set up and start training it!

# In[19]:


trainer_params = torch_utils.preparation.prepare_experiment_id(cfg["model"]["trainer_params"], experiment_id)

# Prepare callbacks and logger using the prepared trainer_params
trainer_params["callbacks"] = torch_utils.preparation.prepare_callbacks(trainer_params)
trainer_params["logger"] = torch_utils.preparation.prepare_logger(trainer_params)

# Prepare the trainer using the prepared trainer_params
trainer = torch_utils.preparation.prepare_trainer(**trainer_params)

model_params = cfg["model"].copy()

model_params["loss"] = torch_utils.preparation.prepare_loss(cfg["model"]["loss"], utils)

# Prepare the optimizer using configuration from cfg
model_params["optimizer"] = torch_utils.preparation.prepare_optimizer(**cfg["model"]["optimizer"])

# Prepare the metrics using configuration from cfg
#metrics = torch_utils.preparation.prepare_metrics(cfg["model"]["metrics"])
model_params["metrics"] = {}

# Create the model using main_module, loss, and optimizer
model = torch_utils.process.create_model(main_module, **model_params)


# ### Train, test and save
# Train, test and save multiple ResNet models with different sizes (parameter k) as presented in the paper.
# 
# **Experiment check**
# 
# ???
# 
# Each experiment is now saved in the folder "experiment" with a specific id, each experiment there is different w.r.t. the configuration, when a new folder is created it means a change in the code is done, not affecting only the configuration.
# 
# **Testing**
# 
# Test phase to obtain the metrics needed for evaluations.
# 
# 
# **Save Experiment**
# 
# Save the experiment id with an hash in the out/exp/experiment where a file with all ids is created avoiding repetitions.

# In[20]:


#5082296 #5157000


# In[21]:


# Train the model using the prepared trainer, model, and data loaders
torch_utils.process.train_model(trainer, model, loaders)

# # Test the trained model
# #test_trained_model(trainer, model, loaders)
torch_utils.process.test_model(trainer, model, loaders)

# Save experiment and print the current configuration
#save_experiment_and_print_config(cfg)
exp_utils.exp.save_experiment(cfg)

# Print completion message
print("An execution with a model of ResNet is completed")
print("######################################################################")
print()


# In[ ]:


# for x,y,w in loaders["test"]:
#     break

# model.eval()
# pred_y = torch.sigmoid(model(x))

# import colorsys

# # Extract the example image with shape (3, 32, 32)
# for example_index in range(12):  # You can change this to view different examples

#     # Display the 3x32x32 image
#     fig, axs = plt.subplots(2, 6, figsize=(18, 3))
#     axs[0,0].imshow(x[example_index].detach().numpy().transpose(1,2,0))  # Display as (32, 32, 3)
#     axs[0,0].set_title("3x32x32 Image")
#     axs[0,0].axis('off')

#     # Display colors in y
#     for i in range(5):
#         axs[0,i+1].imshow([[colorsys.hsv_to_rgb(*tuple(y[example_index].detach().numpy()[i*3:i*3+3]))]])
#         axs[0,i+1].set_title(f"Color {i+1}")
#         axs[0,i+1].axis('off')

#     # Display colors in pred_y
#     for i in range(5):
#         axs[1,i+1].imshow([[colorsys.hsv_to_rgb(*tuple(pred_y[example_index].detach().numpy()[i*3:i*3+3]))]])
#         axs[1,i+1].set_title(f"Color {i+1}")
#         axs[1,i+1].axis('off')
        
#     plt.tight_layout()
#     plt.show()


# In[ ]:


# # Visualize original image, reconstructed image, true and predicted concepts

# for split_name, split_loaders in loaders.items():
#     print(split_name)
#     for img, true_concepts, in split_loaders:
#         fig, ax = plt.subplots(len(img), 4, figsize=(12, 3 * len(img)))
#         rec_img, pred_concepts, emb = model(img)
#         for i, (img, rec_img, true_concepts, pred_concepts) in enumerate(zip(img, rec_img, true_concepts, pred_concepts)):
#             ax[i, 0].imshow(img.detach().cpu().numpy().transpose(1,2,0))
#             ax[i, 0].set_title("Original")
#             ax[i, 1].imshow(rec_img.detach().cpu().numpy().transpose(1,2,0))
#             ax[i, 1].set_title("Reconstructed")
#             ax[i, 2].bar(np.arange(len(true_concepts)), true_concepts.detach().cpu().numpy())
#             ax[i, 2].set_title("True Concepts")
#             ax[i, 3].bar(np.arange(len(pred_concepts)), pred_concepts.detach().cpu().numpy())
#             ax[i, 3].set_title("Predicted Concepts")
#         plt.tight_layout()
#         plt.show()


# In[ ]:




