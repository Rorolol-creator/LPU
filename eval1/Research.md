# Algorithms

It can be seen that to answer classification problems, the mainstream solution
still is deep learning and CNN.

## Deep Learning

- ResNet: Solves the "vanishing gradient"* problem that can be found in deep
  neural networks by skipping connections. This lets models be trained with
  hundreds of layers without degrading performance.

- EfficientNet: Scales network depth, width, and resolution mathematically
  rather than arbitrarily. This makes it one of the most parameter-efficient
  architectures available, achieving high accuracy with less computation.

- VGG: Uniform stack of 3×3 convolutional layers followed by fully connected
  layers. A bit outdated, however, their simple architecture makes them easy to
  understand and adapt.

- Inception: Application of multiple filter sizes in parallel within the same
  layer and concatenate the results. This lets the network capture features at
  different scales simultaneously.


## Transformers

- Vision Transformer: adapts the Transformer architecture (famous in NLP) to
  images by splitting them into fixed-size patches and treating each patch as a
  "token." Self-attention mechanisms then relate every patch to every other,
  allowing the model to capture long-range dependencies across the whole image.

- Swin Transformer: ViT refined. Computes attention within windows and shifts
  those windows between layers to allow cross-window information to flow. This
  hierarchical structure makes it more efficient and better suited to
  high-resolution images.

## Mix

- SIFT + SVM: SIFT detects keypoints and SVM classifies.
- HOG + Random Forest: HOG finds edge directions to simplify the image and
  Random Forest classifies.

## Fine-Grained Visual classification (refining pipeline or module)

- Bilinear CNN: Uses two parallel CNN feature extractors whose outputs are
  combined at each location. This captures pairwise feature interactions across
  the image.

- Part-based model: Detects specific anatomical landmarks or regions and
  classifies based on those local features. By focusing on parts rather than the
  whole image, it improves accuracy on visually similar species.

- Attention mechanisms: Focuses on most discriminative areas. Used of top of
  other CNN / Transformers.

*The vanishing gradient problem is the problem of greatly diverging gradient
magnitudes between earlier and later layers encountered when training neural
networks with backpropagation.
