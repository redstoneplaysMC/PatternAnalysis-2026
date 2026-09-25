# MODEL NAME / Title
Christopher Zhang - s4703439

# Problem
<!-- Brain segmentation is a crucial and critical task within the medical areas. Difficulties in separating gray and white matters (Dénes-Fazakas et al., 2025) or identifying abnormal regions containing tumor (Lin et al., 2021). These tasks require not only high accuracy but also a very time-consuming process. Thus, the DeepLearning model of Improved UNet is introduced to help improve precision and fasten the process while ensuring everyone's safety.   -->
(1 Paragraph.)
Various recognition tasks solved in deep learning frameworks.
Tasks may include:

    Image Segmentation
    Object detection
    Graph node classification
    Image super resolution
    Disease classification
    Generative modelling with StyleGAN and Stable Diffusion

# Model Descripton
<!-- Standard UNet has trouble in dealing with long-range dependencies, blurred boundaries, and low-contrast environment (Al Qurri & Almekkawy, 2023). The improved UNet consist of an encoder-decoder architecture with residual blocks and skip connections. The encoder extract hierarchical features, while the decoder upsamples them to construct a segmentation map. Normalization and dropout are used to improve training and prevent overfitting. In this project, there are four classes that will be output by the model.  -->
Description of the model, and how it works

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

The random seed was...

## Pre-processing
Preprocessing... The images were noramlized between 0 and 1 to...
Justification of training, validation and testing splits of the data.


## Inputs and ouptuts
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