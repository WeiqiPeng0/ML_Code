# ML_Code
## 🔁 Cross-Entropy Loss Summary

### 🟩 Binary Cross-Entropy (BCE)

#### 🔸 Original Loss:
$$
L = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]
$$

#### 🔹 Derivative w.r.t. prediction $\( \hat{y} \)$:
$$
\frac{\partial L}{\partial \hat{y}} = -\left[ \frac{y}{\hat{y}} - \frac{1 - y}{1 - \hat{y}} \right]
$$

#### 🔹 Derivative w.r.t. logits \( z \) (where \( $\hat{y} = \sigma(z)$ \)):
$$
\frac{\partial L}{\partial z} = \hat{y} - y
$$


---

### 🟦 Multi-Class Cross-Entropy (CCE)

#### 🔸 Original Loss (for a single sample):
$$
L = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)
$$

#### 🔹 Derivative w.r.t. prediction \( $\hat{y}_k$ \):
$$
\frac{\partial L}{\partial \hat{y}_k} = -\frac{y_k}{\hat{y}_k}
$$

#### 🔹 Derivative w.r.t. logits \( $z_k$ \) (where \( $\hat{y}_k = \text{softmax}(z_k)$ \)):
$$
\frac{\partial L}{\partial z_k} = \hat{y}_k - y_k
$$


