r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

Ideally the residuals are randomly scattered around zero in a narrow band, with about the same
spread for every $\hat{y}$ and no visible pattern (no curve or funnel). That means the model
captured the structure in the data and what's left is just noise. For the test set we also want the
residuals to look like the training residuals.

In the top-5 features plot the band is wide (MSE about 27) and there is a clear pattern: at both
low and high $\hat{y}$ the points drift away from zero in the same direction. So the errors aren't
random, the model is missing a nonlinear relationship (e.g. MEDV vs. LSTAT is curved in the scatter
matrix). There are also a few very large errors, mostly houses with the capped price of 50.

After adding the nonlinear features and choosing the hyperparameters with CV, the residuals are
much tighter (band of about $\pm 3$, test MSE about 11.4) and the curved pattern is gone. The test
points fall in the same band as the training points, so the model fits well and generalizes,
with only a few outliers left.
"""

part3_q2 = r"""
**Your answer:**

Adding nonlinear features lets the model fit curved relationships between the original features
and the target, which is why the error dropped so much.

1. Yes. The model is $\hat{y} = \vec{w}^\top\varphi(\vec{x})$, which is still linear in the weights
   $\vec{w}$; "linear" refers to the parameters, not the features. We solve it exactly the same way
   (same loss, same closed-form solution), just with the new features as the data matrix.

2. Not really. We can only fit functions that are combinations of the features we chose. In theory
   polynomials of high enough degree can approximate any continuous function, but the number of
   features blows up quickly (91 for degree 2, 455 for degree 3 in our case), which needs a lot more
   data and overfits easily. Also, we have to pick the features by hand.

3. In the new feature space the decision boundary $\vec{w}^\top\varphi(\vec{x}) = 0$ is still a
   hyperplane, but in the original feature space it's no longer one. It becomes a curved boundary,
   e.g. with degree-2 features in 2D it can be a circle, ellipse or parabola. This lets a linear
   classifier separate data that isn't linearly separable in the original features.
"""

part3_q3 = r"""
**Your answer:**

**1.** The joint density is 1 on the unit square, and by symmetry we can integrate over $x < y$
and multiply by 2:

$$
\mathbb{E}_{x,y}[|y-x|] = 2\int_0^1\!\!\int_0^y (y-x)\,dx\,dy = 2\int_0^1 \frac{y^2}{2}\,dy = \frac{1}{3}
$$

**2.** For a fixed $\hat{x} \in [0,1]$, split the integral at $x = \hat{x}$:

$$
\mathbb{E}_{x}[|\hat{x}-x|] = \int_0^{\hat{x}} (\hat{x}-x)\,dx + \int_{\hat{x}}^1 (x-\hat{x})\,dx
= \frac{\hat{x}^2}{2} + \frac{(1-\hat{x})^2}{2} = \hat{x}^2 - \hat{x} + \frac{1}{2}
$$

**3.** The constant $\frac{1}{2}$ doesn't depend on $\hat{x}$ (or on the generator's parameters), so
its gradient is zero. Dropping it just shifts the loss by a constant and doesn't change the
minimum or the training. The same goes for the $\frac{1}{3}$ from part 1, which only depends on the
data.
"""

# ==============

# ==============
