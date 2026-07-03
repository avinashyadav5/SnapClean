# SnapClean: Near-Duplicate Image Detection using Siamese Vision Transformers

> A deep learning-based image retrieval system that detects **near-duplicate images** by learning **semantic similarity** instead of relying on pixel-level comparison.

## Overview

As the amount of multimedia data continues to grow, duplicate and near-duplicate images occupy unnecessary storage, reduce retrieval efficiency, and introduce redundancy in datasets. Traditional approaches such as perceptual hashing (pHash), difference hashing (dHash), SIFT, and ORB primarily depend on handcrafted features or pixel-level similarity, making them less robust to transformations like cropping, compression, illumination changes, and viewpoint variations.

SnapClean addresses this challenge by combining **Vision Transformers (ViT)**, **Siamese Neural Networks**, **Deep Hashing**, and **FAISS** to learn semantic image representations and perform efficient large-scale similarity retrieval.

---

## Problem Statement

Traditional duplicate image detection methods often fail when images undergo transformations such as:

- Cropping
- Rotation
- Brightness changes
- Compression
- Scaling
- Different viewpoints

Although these images appear different at the pixel level, humans still recognize them as representing the same content.

The goal of this project is to identify such **near-duplicate images** by understanding their semantic content rather than comparing raw pixels.

---

## Key Features

- Semantic similarity learning using Vision Transformers
- Siamese Neural Network architecture with shared weights
- Metric learning using Triplet Loss
- Deep feature embeddings
- Deep Hashing for compact binary representations
- FAISS-based large-scale similarity search
- Robust against common image transformations
- Scalable image retrieval pipeline

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Deep Learning | PyTorch |
| Backbone | Vision Transformer (ViT) |
| Similarity Learning | Siamese Neural Network |
| Loss Function | Triplet Margin Loss |
| Similarity Search | FAISS |
| Image Processing | torchvision, PIL |
| Visualization | Matplotlib |
| Evaluation | Scikit-learn |

---

# System Architecture

```
                Dataset
                   │
                   ▼
          Image Preprocessing
                   │
                   ▼
           Triplet Generation
     (Anchor, Positive, Negative)
                   │
                   ▼
      Shared Vision Transformer
                   │
                   ▼
          Feature Embeddings
                   │
                   ▼
           Triplet Loss Training
                   │
                   ▼
      Similar Embeddings Cluster
                   │
                   ▼
            Deep Hashing
                   │
                   ▼
             FAISS Index
                   │
                   ▼
      Top-K Similar Image Retrieval
```

---

# Workflow

### 1. Dataset Preparation

The model is trained on an Airbnb image dataset containing multiple room categories such as:

- Bedroom
- Kitchen
- Living Room
- Exterior
- Staircase

The dataset contains significant variations in lighting, viewpoint, interior design, and layout.

---

### 2. Image Preprocessing

Each image undergoes:

- Resize (224×224)
- Normalization
- Data Cleaning
- Augmentation

This ensures consistent input for the Vision Transformer.

---

### 3. Triplet Generation

Training samples are generated as:

```
Anchor
Positive
Negative
```

Example:

```
Anchor   → Original Bedroom

Positive → Cropped Bedroom

Negative → Kitchen
```

Triplets enable the model to learn relative similarity rather than simple binary classification.

---

### 4. Vision Transformer

Each image passes through the same Vision Transformer.

ViT:

- Splits image into patches
- Converts patches into embeddings
- Applies self-attention
- Learns global relationships
- Produces semantic feature embeddings

---

### 5. Siamese Neural Network

All three images share the same Vision Transformer weights.

This ensures:

- identical feature extraction
- common embedding space
- meaningful distance comparison

---

### 6. Metric Learning

Triplet Loss optimizes the embedding space by:

- Pulling similar image embeddings closer
- Pushing dissimilar embeddings farther apart

The network learns semantic similarity rather than pixel similarity.

---

### 7. Deep Hashing

Feature embeddings are converted into compact binary hash codes.

Advantages:

- Reduced storage
- Faster retrieval
- Efficient indexing

---

### 8. FAISS Retrieval

All embeddings are indexed using FAISS.

When a user uploads a new image:

```
Query Image
        │
        ▼
Vision Transformer
        │
        ▼
Embedding
        │
        ▼
FAISS Search
        │
        ▼
Top-K Similar Images
```

---

# Model Performance

| Metric | Score |
|---------|------:|
| Accuracy | **90.9%** |
| Precision | **99.0%** |
| Recall | **91.0%** |
| F1 Score | **95.0%** |

---

# Why Vision Transformer?

Unlike CNNs that focus primarily on local receptive fields, Vision Transformers use self-attention to model global relationships between different image regions.

Benefits:

- Better semantic understanding
- Robust feature extraction
- Improved similarity learning
- Better handling of transformed images

---

# Why Siamese Neural Networks?

Siamese Networks are designed specifically for similarity learning.

Instead of predicting image classes, they learn an embedding space where:

- Similar images are close together
- Different images are far apart

This makes them ideal for image retrieval applications.

---

# Why Triplet Loss?

Triplet Loss learns from:

- Anchor
- Positive
- Negative

instead of only image pairs.

This produces a more discriminative embedding space, leading to better retrieval performance.

---

# Why Deep Hashing?

Deep Hashing converts continuous feature vectors into compact binary codes.

Benefits:

- Low memory consumption
- Fast comparison
- Scalable retrieval
- Suitable for million-scale datasets

---

# Why FAISS?

Searching every image sequentially is computationally expensive.

FAISS provides:

- Fast nearest-neighbor search
- Efficient indexing
- Large-scale similarity retrieval
- Low-latency inference

---

# Applications

- Duplicate Image Detection
- Dataset Deduplication
- Cloud Storage Optimization
- Digital Asset Management
- Reverse Image Search
- E-commerce Image Search
- Social Media Content Filtering
- Multimedia Retrieval Systems

---

# Future Improvements

- Hard Negative Mining
- Attention-guided Feature Fusion
- Approximate FAISS Indexes (IVF, HNSW)
- Multi-modal Retrieval
- Distributed FAISS
- Cross-domain Image Retrieval
- Mobile Deployment

---

# Results

The proposed framework successfully learns semantic image representations using Vision Transformers and Siamese Networks, enabling robust near-duplicate detection under real-world transformations. By integrating Deep Hashing with FAISS, the system achieves scalable and efficient image retrieval while maintaining high accuracy.

---

## Author

**Avinash Yadav**
