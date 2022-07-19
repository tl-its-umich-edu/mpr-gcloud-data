# mpr-gcloud-data 
# ReadMe & Documentation for MWrite Review Classifications using PeerBERT

This collection of scripts run on GCloud on Vertex AI. The Trainer parts run on a User-Managed notebook, while the Predictor part runs on a Managed Notebook. 
(A Managed Notebook might work for the Trainer part as well, but it is untested and could be prone to specific limitations.)


## Before you begin

The Predictor code is one single notebook called **mpr-peerbert-predictor.ipynb** that must run in a Managed Notebook to allow for scheduling. 

All other files are for the Trainer side code, with the keys ones being:
- mpr-research-trainer.ipynb
- tokenizerMaker.ipynb
- task.py (located in src/trainer)
- jobRunner.ipynb
Other files like **pyproject.toml** and **setup.cfg** are also required by **jobRunner.ipynb** to build the trainer code to use on Vertex AI, but are supplementary.

The code is heavily inspired by documentation from Vertex AI here: 
https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/community-content/pytorch_text_classification_using_vertex_sdk_and_gcloud/pytorch-text-classification-vertex-ai-train-tune-deploy.ipynb

In general, you will also need a good understanding of PyTorch, HuggingFace, and similar Deep Learning concepts for working with the trainer code to change and develop the models. 
The code uses a custom Dataset class, as well as a custom tokenizer trained using M-Write data. 
Read the research paper the models are built off on: https://dl.acm.org/doi/fullHtml/10.1145/3506860.3506892

The project used in GCloud is **MWrite Research** with a project ID of *mwrite-a835*.
You will need access to this project, most likely through *gcp-mwrite@umich.edu*. 

While *us-east* is the closest datacenter to the UoM, *us-central* provides a lot more services and needed configurtaions to run the project. While certain buckets and notebooks can be run across many regions, it is more efficient to stick to just one for consistency. *us-central* hence is the primary region of choice for most tasks and services used. Some buckets currently are multi-region but should be moved to a single region in the future.

---

## Setup for the Predictor Code:

Only one notebook is needed to run predictions: **mpr-peerbert-predictor.ipynb**

You will need to set up a Managed Notebook in Vertex AI for this.

Use region *us-central1 (Iowa)* when setting up this notebook.

This does not have to run on a powerful configuration, but it will make debugging faster when manually running the notebook. I recomend atleast an n1-standard with 4 CPU cores, and atleast a Tesla T4. I recommend going with much more power when scheduling the notebook using the Notebook Executor in the left panel, so that these predictions happen quickly. THe code scales up well with the resources avaible. 

The notebook has the provision to be adjusted via hyperparameters through a CLI, but that is not used here. This for future use cases.

Note: You could run the code as a user-managed notebook but you will lose the ability to schedule it.

---

## Setup for the Trainer Code:

These notebooks and scripts are only if you need to redeploy a model, or train new models on new data or model structures. Look at the User-Managed Notebook **mpr-research-predictor.ipynb** for handling predictions.

Use region *us-central1 (Iowa)* when setting up this notebook.

To set up a new User-Managed Notebook (this is a misnomer, you can have multiple notebooks in this, it's more of an environment)
Go to Vertex AI -> Workbench -> User-Managed Notebook -> New Notebook

Or, go to *mpr-peerbert-pipeline* for the orginal version of this notebook.

If opening a new notebook, you MUST make sure the environment is set to Python with nothing else installed.
You must install PyTorch and the required libraries manually afterwards, as there is a bug that prevents the GPU from being detected correctly, and the code will not work.
Install needed libraries after setup and all should work.

Allocate enough CPU and GPU resources in set up. I recomend atleast an n1-standard with 4 CPU cores, and atleast a Tesla T4. The more resources, the faster the training, as the training pipeline scales up nicely. 

Once inside the Jupyter Lab environment, retrieve the files from Github as needed. There are three notebooks, and one Python script:

1. mpr-research-trainer.ipynb

    This is the notebook to use when you are designing and building you own models. Run the first code block by uncommenting all the lines the first time you set up your workspace. This will install all the libraries needed for you to run the code. Run this block first!
    There is a provison to set hyperparameters in a CLI fashion. That won't work in a Notebook, so you must set that on your own as you build your models. Look at the Config class to see the variables that are configurable. Currently, most variable should require no modifciations, the code is designed to run out of the box. 
    
2. tokenizerMaker.ipynb

    This notebook will let you make custom tokenizers if for some reason you need to make new ones not avaiable in thw mor-research-tokenizers bucket already. You'll have to retrain the whole model from scratch. Stick to the default.
    
3. task.py

    Located in src/trainer/, task.py is basically a script version of mpr-research-trainer.ipynb that is configured by CLI arguments used in jobRunner.ipynb to run the training job on the cloud. The functions are same between this and the notebook, only the main function is slightly different. This is used to build the app that runs the Training Job.
    
4. jobRunner.ipynb

    This notebook sets up and runs the actual training Job for your models. Use this once you are satisfied with your model settings and performance and are ready to deploy. Remember that you will need to train and deploy two models for each tier level of predictions seperately (Or use a loop to iterate through each level like in the script).
    
---

## Under Construction - Buckets and needed files

tokenizerBucketName: str = 'mpr-research-tokenizers'
modelBucketName: str = 'mpr-research-models'
dataBucketName: str = 'mwrite-data-bucket-1'

predResultsBucketName: str = 'mpr-research-prediction-results'

---

## Under Construction - BigQuery connections

dataTableID: str = 'mwrite-a835.mpr_research_uploaded_dataset.course-upload-data'
timestampTableID: str = 'mwrite-a835.mpr_research_uploaded_dataset.course-upload-timestamp'
predictTableID: str = 'mwrite-a835.mpr_research_predicted_dataset.predicted-data'

---

## Under Construction - Overview of Pipeline

---

## Misc. Notes about pipeline
   
The following are notes about some specific quirks made to model training behaviour that significantly deviate from the code used to obtain the results in the paper. These are considerably significant in terms of how model performance is impacted. 

About the training data used for Tier 2 Predictions:
    The data is highly imbalanced for the the Tier 2 Predictions in the current version of the data used. To train a model, it is important to have data be balanced across labels. This gets trickier across multilabel problems such as in the case for Tier 2. A manual truncation of data is performed here in this case to help the model getting better information. In some cases, less data in better quality is much better than more data that provides no new information. 
    
About the special metric computation for Tier 2 Predictions:
    While a simple MCC score is used for Tier 1 predictions, Tier 2 model training leverages a modified version of the MCC score across three lables to get a single score. This is a highly weighted scoring system being applied to account for the skewed nature of the data, and a need to still have a good score regardless. Adjusting this function can let you optimize the model to your liking on how well it will perform. 
    
    
About the overall Trainer Pipeline:
    I do not like how the current pipeline works, since it does not leverage the true Pipeline implementations Vertex AI has to offer. This is because the target problem is very unique. These models are attempting to perform two rounds of predictions on single statement:
    Tier 1: A 3-class single-label classification task,
    Tier 2: A 3-class multi-label classification task that only occurs if a certain class is predicted in the first Tier.
This needs two unique models to be designed from two seperate sets of training data. And they must be executed serially, due to the dependency of the second tier on the first. 
Currently, Vertex AI does not support having two models at the endpoint for a task like this. They do plan to provison for this, but this a niche usecase.

In fact, there is no need to even make a task.py file and make a Job because of the current limitations of Vertex AI. The trainer notebook can save the model to buckets as well, which the predictor notebook can use. I have still kept it in for if in some point in the future Vertex AI expands support for multi-model endpoints, the basic buildings needed for the pipeline will be there to use. Similar case for the proof-of-concept single prediction handler class in the predictor notebook. It has no role now, but in the future it might.

