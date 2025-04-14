### Linear Regression
MSE Loss:
$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2
$$
vectroized form:
$$
J(\theta) = \frac{1}{2m} (X\theta - y)^\top (X\theta - y)
$$


Gradient Descent Rule:
$$
\theta := \theta - \alpha \cdot \frac{1}{m} X^\top (X\theta - y)
$$

