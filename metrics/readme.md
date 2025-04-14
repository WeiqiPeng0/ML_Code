# 📏 Distance and Similarity Metrics

| Metric | Formula | Typical Use Cases | Pros | Cons |
|--------|---------|--------------------|------|------|
| **Euclidean Distance** | \( d(a, b) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2} \) | K-NN, K-Means, clustering in continuous spaces | Intuitive, widely used, works well in low dimensions | Suffers from the curse of dimensionality |
| **Manhattan Distance (L1)** | \( d(a, b) = \sum_{i=1}^{n} |a_i - b_i| \) | Grid-based systems (e.g., robotics), sparse data | Simpler and more robust in high dimensions | Less sensitive to large differences |
| **Cosine Similarity** | \( \text{cos\_sim}(a, b) = \frac{a \cdot b}{\|a\| \|b\|} \) | Text similarity, NLP embeddings, recommendation systems | Ignores magnitude, focuses on direction, great for high-dimensional sparse vectors | Can be undefined for zero vectors |
| **Dot Product** | \( \text{dot}(a, b) = \sum_{i=1}^{n} a_i \cdot b_i \) | Neural networks, attention mechanisms, similarity scoring | Simple and fast, used in ML internals | Not scale-invariant; sensitive to magnitude |
| **Minkowski Distance** | \( d(a, b) = \left( \sum_{i=1}^{n} |a_i - b_i|^p \right)^{1/p} \) | Generalized distance metric for tuning between L1 and L2 | Flexible (p = 1: L1, p = 2: L2) | Hard to interpret for non-integer p |
| **Chebyshev Distance** | \( d(a, b) = \max_{i} |a_i - b_i| \) | Chessboard movement, anomaly detection | Easy to compute, reflects max deviation | Ignores all other dimensions except the largest |
| **Jaccard Similarity** | \( \text{Jaccard}(A, B) = \frac{|A \cap B|}{|A \cup B|} \) | Set similarity, tag matching, recommendation systems | Captures overlap well, good for binary data | Only works on sets or binary vectors |
| **Hamming Distance** | \( d(a, b) = \sum_{i=1}^{n} \mathbb{1}_{a_i \neq b_i} \) | Error correction codes, binary string comparison | Works on strings, bit vectors | Not meaningful for real-valued data |
| **Correlation (Pearson)** | \( r(a, b) = \frac{\sum (a_i - \bar{a})(b_i - \bar{b})}{\sqrt{\sum (a_i - \bar{a})^2} \sqrt{\sum (b_i - \bar{b})^2}} \) | Feature selection, time series, statistics | Measures linear relationship, scale-invariant | Only captures linear correlation |
