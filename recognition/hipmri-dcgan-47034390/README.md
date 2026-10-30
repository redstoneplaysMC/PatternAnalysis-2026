# Hip MRI image generation using DCGAN
Christopher Zhang - s4703439

# Problem
<!-- SSIM means: "For each generated image, how structurally similar is it to the single most similar real image?" -->

<!-- What it tells you

FID              Distribution similarity
SSIM             Structural/pixel similarity
Generated images Visual quality
G/D losses       Training dynamics
Generation time  Computational performance
GPU memory       Resource requirements 

Investigate: General Engineering & Research Guidelines
1. Pipeline Setup & Training Stability: Implement appropriate loss formulations (e.g., Wasserstein   tance with gradient penalty, 
discrete codebook quantization, or latent diffusion schedules). Address training stability and gradient hygiene.
2. Baseline Implementation & Comparison: Implement an initial generative baseline (e.g., standard DCGAN or basic autoencoder) to 
establish a benchmark. Compare your chosen model
from the table below against this baseline on sample clarity, diversity, and fidelity. 
3. Quantitative Benchmarking: Compute Structural Similarity Index (SSIM > 0.6) across synthesized cohorts against reference 
anatomical slices. Measure distribution fidelity (FID, feature coverage, or latent distance metrics). 
Profile GPU memory, training stability, and generation runtime.
4. Qualitative Autopsy & Memorization Audit: Display uncurated sample grids across diverse random seeds (avoid cherry-picking). 
For style-based or diffusion models, provide a 2D t-SNE or UMAP embedding plot comparing real and synthetic distributions. 
Conduct a failure autopsy on 3–5 flawed generations (checkerboard artifacts, distorted tissue margins, mode drops) 
and audit nearest training neighbors to disprove patient memorization. -->


Generartive models allow synthesizing realistic synthetic cohorts without comprimising on patient anonymity. To do this, DC-GAN ()

We are first starting with DCGAN. Use the HipMRI 2D slices (`keras_slices_data`)
Reasonably clear image with SSIM > 0.6.

The purpose of this assignment is to synthesize artificial HipMRI data for medical training and imaging purposes.


Segmentation dataset isn't required; since we are running a generative model.

Modules.py should contain the code for both? perhaps the basic encoder can be in a separate python file.

<!-- This gives you a nice experimental question:
How does a VAE compare with a DCGAN for generating anatomically plausible 2D hip MRI slices?
You could then use the same dataset and preprocessing for both.
For example:    
    │      Evaluation    │
    │ SSIM               │
    │ FID                │
    │ Diversity          │
    │ Coverage           │
    │ Visual quality     │
    │ Failure analysis   │
    │ Memorization audit │
-->

<!-- What to include in your report
- Uncurated grids from multiple seeds.
- A distribution embedding if applicable to your model.
- Three to five genuine failure cases with explanations.
- A nearest-neighbour comparison grid and similarity statistics.
- A conclusion discussing diversity, anatomical quality, possible memorization, and limitations
-->

<!-- `modules.py`: containing the source code of the components for your model. Each component must be implemented as a class/function 
    in pytorch, or TF/Keras. (I will probably be using pytorch). The model should not depend on numpy, in any way unless otherwise 
    approved by the teaching staff.
`dataset.py`: Data loader for loading and pre-processing of your data, including leakage-free train/validations/test splitting 
    functions and data augmentations.
`train.py` : Code for training, validating testing and saving of the model. Perhaps test on the CPU node first with a 
    lower-power model. This module should be imported from `modules.py` and the data loader shold be imported from `dataset.py`. 
    Ensure that losses and metrixs are plotted during training.
`predict.py` Show an example usage of the trained model. It should load the saved model weights, run inference on test cases, 
    print out any results, and provide visualizations were applicable (e.g. prediction overlays, generated samples and heatmaps). 
    Numpy is allowed for visualization or loading of the data.
`README.md` Document the project, working principles, feasibility review, experiments and instructions. -->

# Model Descripton
In this project, there are four classes that will be output by the model. To understand how the models work, we must first understand the properties of the dataset.
<!-- All preprocessing is performed in dataset.py using the HipMRIDataset class, which handles NIfTI medical imaging files and prepares them for GAN training. -->
I have used the variational autoencoder and DC-GAN as my chosen models to generate the images.


## Description of the Dataset
The dataset is the Hip-MRI dataset. It is split into 8 weeks, from week 0 to week 7. The training dataset has 11460 images; there are 32 people in the training dataset; from case 4 to case 35; 3 people in the validation dataset, from case 36 to 39, and 3 people in the test dataset; case 40 to 42.

For my experiment, in order to get more controlled data, I will only be using week 4 and week 7 of the dataset; although it is possible to train on the entire model, the resulting generated data would not be able to distinguish the week data.

## DC-GAN
GANs use a generator and a discriminator. Both models must compete with each-other during the training loop; the discriminator must be able to distinguish if the images created by the generator are real. DC-gan extends this by using a deep convolutional network; 
- replace pooling layers with strided convolutions.
- use batchnorm for both the generator and discriminator.
- Remove fully connected hidden layers for deeper architectures.
- Use ReLU activation in the generator for all layers, except the output, which uses Tanh.
- Use LeakyReLU activation for all layers of the discriminator.

The architecture has the following structure:

The training process is:



## VAE Baseline
The VAE is divided into an encoder and a decoder. They have the following architectures:


# Feasibility Review 
(From S3, 1 page)
Intended User: Medical students who need synthetic Hip MRI data so that they can test while maintaining patient privacy.

DIFFICULTY: Normal.

Chosen model(s): Variatonal autoencoder, DCGAN. Purpose of the models is to generate synthetic images; this report serves to compare the two models.

Feasibility: The model was first tested on the CIFAR-10 Dataset, batch-size 32 and 30 epochs. It was able to generate synthetic data with 30 epochs; 20 epochs isn’t sufficient enough. The CIFAR10 Dataset is 32x32; whereas the HipMRI Dataset is a bit more. This was compared with the basic auto-encoder. Once these models worked, a similar model was tested on the HIPMRI dataset on the CPU and was able to run successfully.

Verification is done by splitting the dataset into test, validation and training vectors; this was done already in the supplied dataset.

Risks: The model doesn’t generate adequately detailed synthetic data without excessive training time. The model generates, but it falls into model collapse due to insufficient data variability. 


# Reproducibility
## Dependencies required, inc. exact versions.
The following is the list of python dependencies required to run this project.
-   torch
-   numpy
-   random
-   pyYaml

For the dataset:
-   tqdm
-   typing
-   nibabel
-   re
To ensure reproducible results, in config.yaml, set the seed to 42.

To run the DC-GAN model, execute the following commands to train/predict:
```bash
py -m train
py -m predict
```

Execute the following commands to run the baseline model:
```bash
py -m baseline.train
py -m baseline.predict_with_pca
py -m baseline.predict
```

## Pre-processing
The data is pre-processed in `dataset.py`. First, the images are stored in the nii.gz format; they must be extracted and converted into a Tensor.

## Inputs, outputs, and visualizations.
Example inputs and outputs go here.
The algorithm takes in...
The algorithm should ouptut...

This is shown in the following table.

| Option A | Option B | Option C |
| -------- | -------- | -------- |
|          |          |          |
|          |          |          |


## Visualization of the algorithm and training plots
The model was trained for 50 Epochs on the RANGPUR cluster.

# Investigation of the Open Research Dilemma
<!-- ????? What does this mean? -->
– Quantitative benchmarking of your chosen model against your implemented baseline model under identical evaluation splits.
– Resource profiling table documenting computational efficiency (peak GPU VRAM, parameter count, and inference runtime/latency).
– In-depth qualitative error autopsies analyzing 3–5 representative failure cases from the dataset (diagnosing root triggers and failure modes).
– Actionable engineering recommendation and trade-off synthesis for your project manager.

# Artifical Intellegence Usage disclosure
<!-- The mandatory Artificial Intelligence Usage Disclosure subsection (see Section 6). -->
AI tools used were: ChatGPT and Github Copilot Autocomplete. The code was checked over manually, and this document was written mostly without AI.

## Tasks that the AI tools were used for:
The AI tools were used for:
- Providing inspiration for how to implement the methods.
- Code checking and Debugging.
- Supplementary Code review. 
- Cursory Feedback on the generated outputs.
- Code scaffolding.
- Writing code comments.

## AI-code auditing method
The code was generated by AI; but the expected outputs were looked at manually. Any comments were reviewed manually to ensure that they were correct.

A lot of the code was inspired by [2], where this user did the same project, but using W-GAN instead of DC-GAN, and didn't do a comparison model. However, I did not copy the code from that PR; just used it as inpsiration.

<!-- # Responsible Use of AI and Mandatory Disclosure -->
<!-- Generative AI models are permitted to assist your learning and implementation workflows. However,
the expectations of engineering professionalism apply:
1. Verification Responsibility: 
    You are 100% personally responsible for every line of code, loss formula, and metric in your submission. 
    Blaming an AI model for an incorrect calculation or data leak is unacceptable.
2. Mandatory AI Disclosure Section: In your README.md, you must include a dedicated subsection
    titled “Artificial Intelligence Usage Disclosure” documenting:
    - Which AI tools were utilized (e.g., GitHub Copilot for auto-complete, ChatGPT/Claude for debugging syntax, Gemini for concept lookup).
    - What specific tasks AI assisted with (e.g., scaffolding data loading functions, refactoring plotting scripts).
    - How you audited, verified, and empirically tested the AI-suggested code.
3. Prohibited Misconduct: Copying code from fellow students, reusing previous years’ pull requests,
or committing unverified AI hallucinations. -->

## References
<!-- Put references here. Use the APA Format.-->
<!-- Reference any websites used to do the assignment. -->
[1] Radford, A., Metz, L., & Chintala, S. (2016). Unsupervised representation learning with deep convolutional generative adversarial networks. arXiv. https://arxiv.org/abs/1511.06434

[2] samuel2003-coder. (2025). *PatternAnalysis2025* [recognition]. GitHub. https://github.com/samuel2003-coder/PatternAnalysis-2025/tree/topic-recognition

https://en.wikipedia.org/wiki/Wasserstein_metric

https://lilianweng.github.io/posts/2017-08-20-gan/#what-is-wasserstein-distance
