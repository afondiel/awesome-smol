# Awesome Smol Models  

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)  

**Awesome Smol Models** is a curated list of small, lightweight AI models, tools, and resources for domains like **language**, **audio**, **vision**, and **multimodal tasks**. These models are designed for edge devices, resource-constrained environments, and rapid prototyping.

Inspiration: [awesome-tensorflow-lite](https://github.com/margaretmz/awesome-tensorflow-lite).  

### **What are Smol Models?**

<img alt="smol ecosystem" width="800" height="621" src="https://pbs.twimg.com/media/GbPSEtWbgAAwGb2?format=png&name=large" />

(Source: [Link](https://x.com/LoubnaBenAllal1/status/1852055587275895188))

Smol models are AI models optimized for efficiency, offering:  
- **Lightweight Design:** Minimal memory usage.  
- **Fast Inference:** High performance on limited hardware.  
- **Accessibility:** Ideal for mobile, IoT, and edge deployment.  
- **Versatility:** Support across text, audio, vision, and more.  

### **Goal**  
- Highlight community-driven advancements in lightweight AI.  
- Provide a central hub for small model resources, tools, and benchmarks.  
- Promote faster adoption for real-world applications.  

### **Contribute to the List**  
Your contributions are warmly welcome! Submit a pull request (PR) following the [contribution guidelines](CONTRIBUTING.md).  

## **Table of Contents**  
- [Language Models](#language-models)  
- [Audio Models](#audio-models)  
- [Vision Models](#vision-models)  
- [Multimodal Models](#multimodal-models)  
- [Pretrained Models Hub](#pretrained-models-hub)  
- [Tools and Frameworks](#tools-and-frameworks)
- [AI Frontier solution at the Edge](#ai-frontier-solution-at-the-edge)
- [AI Latest News & Releases](#ai-latest-news--releases)  
- [Resources](#resources)  

## **Language Models**

| **Model**                 | **Task**            | **Platform**       | **References**                                                 |  
|----------------------------|---------------------|--------------------|-----------------------------------------------------------------|  
| SmolLM                    | General NLP        | Edge, Desktop     | [Hugging Face](https://huggingface.co/smolai/smollm-1.7b)       |  
| Zamba2-7B                 | Text Understanding | Mobile, Edge      | [Hugging Face](https://huggingface.co/smolai/zamba2-7b)         |  
| EuroLLM-1.7B              | Multilingual NLP   | Mobile, IoT       | [Hugging Face](https://huggingface.co/smolai/eurollm-1.7b)      |  
| Mistral-Small-Instruct-2409| Instruction Tasks  | Edge, Mobile      | [Hugging Face](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) |  
| Ministral-8B-Instruct-2410| Instruction Tasks  | Edge, Desktop     | [Hugging Face](https://huggingface.co/microsoft/phi-2)          |  
| TinyLlama                 | Conversational AI  | Edge, IoT         | [Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) |  
| Phi-3                     | Text Generation    | Mobile, Edge      | [Hugging Face](https://huggingface.co/microsoft/phi-2)          |  
| Gemma 2                   | Multilingual NLP   | Mobile, Desktop   | [Hugging Face](https://huggingface.co/google/gemma-2b)          |  

## **Audio Models**

| **Model**         | **Task**            | **Platform**         | **References**                                                  |  
|--------------------|--------------------|----------------------|------------------------------------------------------------------|  
| Whisper Small     | Speech Recognition | Edge, Desktop        | [Hugging Face](https://huggingface.co/openai/whisper-small)      |  
| Audio-Mamba (AuM) | Audio Processing   | Edge, IoT            | [GitHub](https://github.com/kaistmm/Audio-Mamba-AuM)            |  
| MusicGen          | Music Generation   | Edge, Desktop        | [GitHub](https://github.com/facebook/MusicGen)                  |  
| FastSpeech2 Small | Text-to-Speech     | Mobile, Edge         | [Hugging Face](https://huggingface.co/facebook/fastspeech2-small) |  
| HiFi-GAN Mini     | Audio Enhancement  | Mobile, IoT          | [GitHub](https://github.com/jik876/hifi-gan)                    |  
| MatchboxNet Small | Keyword Spotting   | Edge, IoT            | [Hugging Face](https://huggingface.co/speechbrain/matchboxnet-small) |  

## **Vision Models**  

| **Model**                | **Task**             | **Platform**         | **References**                                                  |  
|---------------------------|----------------------|----------------------|------------------------------------------------------------------|  
| MobileNet V3 Small       | Image Classification | Mobile, Edge         | [TensorFlow](https://www.tensorflow.org/lite/models/imagenet)    |  
| EfficientNet-Lite Small  | Image Classification | Mobile, IoT          | [GitHub](https://github.com/google/automl/tree/master/efficientnet) |  
| YOLOv5 Nano              | Object Detection     | Edge, IoT            | [GitHub](https://github.com/ultralytics/yolov5)                 |  
| DeepLab Lite Small       | Image Segmentation   | Mobile, Edge         | [GitHub](https://github.com/tensorflow/models/tree/master/research/deeplab) |  
| MobileUNet               | Image Segmentation   | Mobile, IoT          | [GitHub](https://github.com/zhixuhao/unet)                      |  
| Vision Transformer Small | Vision Tasks        | Mobile, Edge         | [Hugging Face](https://huggingface.co/models?filter=vit)         |  

## **Multimodal Models**

| **Model**        | **Task**              | **Platform**         | **References**                                                  |  
|-------------------|-----------------------|----------------------|------------------------------------------------------------------|  
| Mini-DALL-E      | Text-to-Image         | Mobile, Desktop      | [GitHub](https://github.com/borisdayma/dalle-mini)              |  
| TinyCLIP         | Vision-Language       | Edge, IoT            | [Hugging Face](https://huggingface.co/openai/clip)              |  
| Mini-ALIGN       | Vision-Language       | Mobile, Edge         | [GitHub](https://github.com/google-research/align)              |  

## **Pretrained Models Hub**  
Pre-trained lightweight models ready for deployment:  
- [Hugging Face Pretrained Smol Models](https://huggingface.co/models?sort=trending&search=smol): Ready-to-deploy smol models with associated datasets, and demo apps (Spaces).  
- [Model Zoo Models Categories](https://modelzoo.co/categories): Open source deep learning code and pretrained models.
- [Kaggle Pre-trained Models](https://www.kaggle.com/models): Use and download pre-trained models for your machine learning projects.
- [Tensorflow Hub](https://www.tensorflow.org/hub): A repository of trained machine learning models.
- [Pytorch Hub](https://pytorch.org/hub/): Discover and publish models to a pre-trained model repository designed for research exploration.  

## **Other Related Resources**  
- [edge-ai - @crespum](https://github.com/crespum/edge-ai)  
- [awesome-tensorflow-lite - @margaretmz](https://github.com/margaretmz/awesome-tensorflow-lite)
- [Smol Vision - @merveenoyan](https://github.com/merveenoyan/smol-vision)  

## **Tools and Frameworks**
- [LiteRT - formerly TensorFlow Lite](https://ai.google.dev/edge/litert): Lightweight model deployment for Android.
- [CoreML](https://developer.apple.com/documentation/coreml): Apple’s ML framework for iOS.
- [ExecuTorch](https://pytorch.org/executorch-overview): Pytorch/Meta end-to-end solution for enabling on-device inference.    
- [ONNX Runtime](https://onnxruntime.ai/): Efficient inference engine.  
- [OpenVINO](https://github.com/openvinotoolkit/openvino): Toolkit for optimizing and deploying deep learning models.

## **AI Frontier solution at the Edge**
- [Google AI Edge](https://ai.google.dev/edge)
- [AWS IoT for the Edge](https://aws.amazon.com/iot/solutions/iot-edge/)
- [Azure IoT Edge - Build the intelligent edge](https://azure.microsoft.com/en-us/products/iot-edge/)
- [Qualcomm On-Device AI Solutions](https://www.qualcomm.com/edgeofpossible)
- [Meta - Pytorch Edge](https://pytorch.org/edge)
- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt-getting-started)
- [Edge Impulse](https://docs.edgeimpulse.com/docs)
- [Edge AI + Vision Alliance](https://www.edge-ai-vision.com/)
- [Edge AI Foundation](https://www.edgeaifoundation.org/)

## **AI News & Announcements**
- 2024/09/25 [Meta - Llama 3.2: Revolutionizing edge AI and vision with open, customizable models](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)

## **Resources**

### Practical Guides
- [Edge-AI core concepts for all levels](https://github.com/afondiel/computer-science-notebook/tree/master/core/systems/edge-computing/edge-ai/concepts/)
- [ML Optimization Resources](https://github.com/afondiel/computer-science-notebook/tree/master/core/ai-ml/ml-notes/model-optimization)
  
### Crash-Courses & Moocs
- [Introduction to On-Device AI - Qualcomm](https://github.com/afondiel/Introduction-to-On-Device-AI-DLAI)
- [Introduction to edge AI - Edge Impulse](https://docs.edgeimpulse.com/docs/concepts/edge-ai/intro-to-edge-ai)

### **Blogs**
- [Why Small Language Models (SLMs) Are The Next Big Thing In AI - Forbes (2024/11/25)](https://www.forbes.com/sites/deandebiase/2024/11/25/why-small-language-models-are-the-next-big-thing-in-ai/)
- [Optimizing Generative AI for Edge Devices](https://www.qualcomm.com/news/onq/2023/12/optimizing-generative-ai-for-edge-devices)
- [Deploying ML Models on The Edge - @Microsoft](https://conferences.oreilly.com/artificial-intelligence/ai-eu-2019/cdn.oreillystatic.com/en/assets/1/event/299/Deploying%20machine%20learning%20models%20on%20the%20edge%20Presentation.pdf)  
- [Fine-Tuning Small Language Models - Kili](https://kili-technology.com/large-language-models-llms/a-guide-to-using-small-language-models#fine-tuning-small-language-models)
- [AI on the edge: latest insights and trends @Qualcomm](https://www.qualcomm.com/news/onq/2023/09/ai-on-the-edge-the-latest-on-device-ai-insights-and-trends)
- [Small is the new big: the rise of small language models - Capgemini](https://www.capgemini.com/be-en/insights/expert-perspectives/small-is-the-new-big-the-rise-of-small-language-models/)
- [The 5 leading small language models of 2024: Phi 3, Llama 3, and more - DSDojo](https://datasciencedojo.com/blog/small-language-models-phi-3/)
- [7 Steps to Running a Small Language Model on a Local CPU](https://www.kdnuggets.com/7-steps-to-running-a-small-language-model-on-a-local-cpu)

### **Deep Dive Podcasts**
- [SmolLM2 Released: A Series (0.1B, 0.3B, and 1.7B) of Small Language Models for OnDevice Applications](https://www.youtube.com/watch?v=7J0PAffn-QU)

### **Books**  
- [Machine Learning Systems - Vijay Janapa Reddi / Harvard (online & interactive book)](https://mlsysbook.ai/)  
- [Edge-AI Books Collection - @cs-books](https://github.com/afondiel/cs-books/edge)
