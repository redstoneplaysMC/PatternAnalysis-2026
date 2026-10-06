# MODEL NAME / Title
Christopher Zhang - s4703439

# Problem
<!-- Brain segmentation is a crucial and critical task within the medical areas. Difficulties in separating gray and white matters (Dénes-Fazakas et al., 2025) or identifying abnormal regions containing tumor (Lin et al., 2021). These tasks require not only high accuracy but also a very time-consuming process. Thus, the DeepLearning model of Improved UNet is introduced to help improve precision and fasten the process while ensuring everyone's safety.   -->
(1 Paragraph.)

Comparing basic autoencoder to DCGAN.
OK. i think i might want to do some VAE based approach, then compare it with DC-GAN for hip MRI image generation.

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
DCGAN. Use the HipMRI 2D slices (`keras_slices_data`)
Reasonably clear image with SSIM > 0.6.

Generartive models allow synthesizing realistic synthetic cohorts without comprimising on patient anonymity. The purpose of this assignment is to synthesize artificial
HipMRI data for medical training and imaging purposes. To do this, the 

`modules.py`: containing the source code of the components for your model. Each componetn must be imlemented as a class/fucntion in pytorch, or TF/Keras. (I will probably be using pytorch). The modle shold not depend on numpy, in any way unless otherwise approved by the teaching staff.
`dataset.py`: Data loader for loading and pre-processing of your data, including leakage-free train/validations/test splitting ufnctions and data augmentations.
`train.py` : Code for training, validating testing and saving of the model. Perhaps test on the CPU node first with a lower-power model. This module should be imported from `modules.py` and the data loader shold be imported from `dataset.py`. Ensure that losses and metrixs are plotted during training.
`predict.py` Show an example usage of the trained model. It should load the saved model weights, run inference on test cases, print out any results, and provide visualizations were applicable (e.g. prediction overlays, generated samples and heatmaps). Numpy is allowed for visualization or loading of the data.
`README.md` Document the project, working principles, feasibility review, experiments and instructions.

# Model Descripton
<!-- Standard UNet has trouble in dealing with long-range dependencies, blurred boundaries, and low-contrast environment (Al Qurri & Almekkawy, 2023). The improved UNet consist of an encoder-decoder architecture with residual blocks and skip connections. The encoder extract hierarchical features, while the decoder upsamples them to construct a segmentation map. Normalization and dropout are used to improve training and prevent overfitting. In this project, there are four classes that will be output by the model.  -->
<!-- Description of the model, and how it works -->
The models I will be using are a basic autoencoder and the DC-GAN model. DC-GAN will be used here since it should hopefully produce better results.

<!-- All preprocessing is performed in dataset.py using the HipMRIDataset class, which handles NIfTI medical imaging files and prepares them for GAN training. -->

# Feasibility Review 
(From S3, 1 page)

Is this feasible? Write a 1-page MD for this.


# Reproducibility
## Dependencies required, inc. exact versions.
The following is the list of python dependencies required to run this project.
-   ABCD
-   ABCD
-   ABCD
-   ABCD
-   ABCD

To ensure reproducible results, use the following seeds:
```
torch.random_seed()
np.random_seed()
random.seed()
torch.cuda.manual_seed_all()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

## Pre-processing
Preprocessing... The images were noramlized between 0 and 1 to...
Justification of training, validation and testing splits of the data.


## Inputs, outputs, and visualizations.
The model takes inputs from nii.gz files.
Example inputs and outputs go here.
The algorithm takes in...
The algorithm should ouptut...

The dice similarity scores were...
This is shown in the following table.

| Option A | Option B | Option C |
| -------- | -------- | -------- |
|          |          |          |
|          |          |          |


## Visualization of the algorithm and training plots
The training process was visualized and is shown in the following plots.
<!-- Paste model output here, and training plots. -->

# Investigation of the Open Research Dilemma
<!-- ????? What does this mean? -->
– Quantitative benchmarking of your chosen model against your implemented baseline model under identical evaluation splits.
– Resource profiling table documenting computational efficiency (peak GPU VRAM, parameter count, and inference runtime/latency).
– In-depth qualitative error autopsies analyzing 3–5 representative failure cases from the dataset (diagnosing root triggers and failure modes).
– Actionable engineering recommendation and trade-off synthesis for your project manager.


<!-- The mandatory Artificial Intelligence Usage Disclosure subsection (see Section 6). -->
# Responsible Use of AI and Mandatory Disclosure
Generative AI models are permitted to assist your learning and implementation workflows. However,
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
or committing unverified AI hallucinations.

<!-- Put the actual AI Disclosure here. -->

# References
<!-- Put references here. Use the APA Format.-->
<!-- Reference any websites used to do the assignment. -->
REFERENCES GO HERE.
[10] DCGAN, radford et al, [2015]
https://en.wikipedia.org/wiki/Wasserstein_metric