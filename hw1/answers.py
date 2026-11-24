r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

1. **False.** The split matters. The test set has to look like the data the model will see later,
   and both sets need to be big enough. For example, if all images of one class end up in the test
   set, or the test set has only a handful of samples, the split is useless for estimating
   performance. Usually we split randomly (or stratified by class) to avoid this.

2. **False.** The test set should only be used once, at the very end. Cross-validation is for
   choosing hyperparameters, so it's done only on the training data (split into folds). If the test
   set is used during CV, it influences our choices and the final test score is too optimistic.

3. **True** (with a small caveat). The average validation score over the folds is our estimate of
   how well a model with these hyperparameters generalizes, and that's how we compare between
   settings. Since we then pick the best setting, this estimate ends up a bit optimistic, which is
   why we still report the final result on the test set.

4. **False.** Adding noise to the labels corrupts the ground truth, so we can't use it to measure
   anything. To check robustness we would perturb the *inputs* (e.g. add noise to the images) and
   see whether the predictions stay the same. Label noise can act as a kind of regularization during
   training, but that's improving the model, not validating it.
"""

part1_q2 = r"""
**Your answer:**

No. By choosing $\lambda$ according to the test results, he effectively used the test set for
training (model selection). The test set is no longer independent of the model, so the score he
gets on it is too optimistic and doesn't tell him how well the model really generalizes.

He should have split off a validation set from the training data (or used k-fold CV on the
training set) to choose $\lambda$, and only then evaluated the final model once on the test set.
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**

Each term of the loss is $\max(0, \Delta + \vec{w}_j^\top\vec{x}_i - \vec{w}_{y_i}^\top\vec{x}_i)$.
With $\Delta < 0$, a wrong class is not penalized even if its score is *higher* than the correct
class's score, as long as it's higher by less than $|\Delta|$. So the loss no longer asks for the
correct class to win.

Worse, $\boldsymbol{W} = 0$ becomes a perfect solution: all scores are 0, so every hinge term is
$\max(0, \Delta) = 0$, and the regularization term is also 0. The total loss is 0, which is the
global minimum, so training would push the weights towards zero and the model wouldn't learn
anything useful. That's why $\Delta$ has to be positive (its exact value doesn't matter much,
since $\boldsymbol{W}$ can just scale up).
"""

part2_q2 = r"""
**Your answer:**

The model learns one template per class. The score of a class is the dot product of its weight
image with the input, so each weight image shows which pixels should be bright (positive weights)
and which should be dark (negative weights) for that digit. In the visualization you can roughly
see the shape of each digit, e.g. a ring for 0 with a dark spot in the middle, and a vertical stroke
for 1. Basically it's matching the input against an "average" digit of each class.

This also explains the errors. A digit that is written in an unusual way (shifted, slanted, thick
or thin) doesn't overlap well with its own template and can overlap more with another one. In the
examples we see a 5 classified as 6, a 4 as 6, a 3 as 2 and a 7 as 9. These pairs share many pixels
(the bottom loop of 5 and 6, the top of 7 and 9), and a linear model can't look at the details of
the shape, only at the pixel overlap with the template.
"""

part2_q3 = r"""
**Your answer:**

1. **Good.** The training loss drops quickly in the first few epochs and then keeps going down
   slowly and smoothly until it levels off. With a learning rate that's too low the loss would go
   down slowly and almost linearly, and it would still be far from converging after the same
   number of epochs. With a learning rate that's too high the loss would jump around (or even
   blow up) instead of decreasing smoothly, because the steps overshoot the minimum.

2. **Slightly overfitted.** The training accuracy (around 92.5%) ends up a bit above the
   validation accuracy (around 91.3%), and the gap slowly grows in the later epochs while the
   validation accuracy stays flat. The gap is small, so it's only slightly overfitted. It's not
   underfitted, since both accuracies are high and the training loss has basically converged.
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
