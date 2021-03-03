### Softmax classifier

It turns out that the SVM is one of two commonly seen classifiers. The other popular choice is the **Softmax classifier**, which has a different loss function. If you've heard of the binary Logistic Regression classifier before, the Softmax classifier is its generalization to multiple classes. Unlike the SVM which treats the outputs \\(f(x_i,W)\\) as (uncalibrated and possibly difficult to interpret) scores for each class, the Softmax classifier gives a slightly more intuitive output (normalized class probabilities) and also has a probabilistic interpretation that we will describe shortly. In the Softmax classifier, the function mapping \\(f(x_i; W) =  W x_i\\) stays unchanged, but we now interpret these scores as the unnormalized log probabilities for each class and replace the *hinge loss* with a **cross-entropy loss** that has the form:

$$
L_i = -\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right) \hspace{0.5in} \text{or equivalently} \hspace{0.5in} L_i = -f_{y_i} + \log\sum_j e^{f_j}
$$

where we are using the notation \\(f_j\\) to mean the j-th element of the vector of class scores \\(f\\). As before, the full loss for the dataset is the mean of \\(L_i\\) over all training examples together with a regularization term \\(R(W)\\). The function \\(f_j(z) = \frac{e^{z_j}}{\sum_k e^{z_k}} \\) is called the **softmax function**: It takes a vector of arbitrary real-valued scores (in \\(z\\)) and squashes it to a vector of values between zero and one that sum to one. The full cross-entropy loss that involves the softmax function might look scary if you're seeing it for the first time but it is relatively easy to motivate.

**Information theory view**. The *cross-entropy* between a "true" distribution \\(p\\) and an estimated distribution \\(q\\) is defined as:

$$
H(p,q) = - \sum_x p(x) \log q(x)
$$

The Softmax classifier is hence minimizing the cross-entropy between the estimated class probabilities ( \\(q = e^{f_{y_i}}  / \sum_j e^{f_j} \\) as seen above) and the "true" distribution, which in this interpretation is the distribution where all probability mass is on the correct class (i.e. \\(p = [0, \ldots 1, \ldots, 0]\\) contains a single 1 at the \\(y_i\\) -th position.). Moreover, since the cross-entropy can be written in terms of entropy and the Kullback-Leibler divergence as \\(H(p,q) = H(p) + D_{KL}(p\|\|q)\\), and the entropy of the delta function \\(p\\) is zero, this is also equivalent to minimizing the KL divergence between the two distributions (a measure of distance). In other words, the cross-entropy objective *wants* the predicted distribution to have all of its mass on the correct answer.

**Probabilistic interpretation**. Looking at the expression, we see that

$$
P(y_i \mid x_i; W) = \frac{e^{f_{y_i}}}{\sum_j e^{f_j} }
$$

can be interpreted as the (normalized) probability assigned to the correct label \\(y_i\\) given the image \\(x_i\\) and parameterized by \\(W\\). To see this, remember that the Softmax classifier interprets the scores inside the output vector \\(f\\) as the unnormalized log probabilities. Exponentiating these quantities therefore gives the (unnormalized) probabilities, and the division performs the normalization so that the probabilities sum to one. In the probabilistic interpretation, we are therefore minimizing the negative log likelihood of the correct class, which can be interpreted as performing *Maximum Likelihood Estimation* (MLE). A nice feature of this view is that we can now also interpret the regularization term \\(R(W)\\) in the full loss function as coming from a Gaussian prior over the weight matrix \\(W\\), where instead of MLE we are performing the *Maximum a posteriori* (MAP) estimation. We mention these interpretations to help your intuitions, but the full details of this derivation are beyond the scope of this class.

**Practical issues: Numeric stability**. When you're writing code for computing the Softmax function in practice, the intermediate terms \\(e^{f_{y_i}}\\) and \\(\sum_j e^{f_j}\\) may be very large due to the exponentials. Dividing large numbers can be numerically unstable, so it is important to use a normalization trick. Notice that if we multiply the top and bottom of the fraction by a constant \\(C\\) and push it into the sum, we get the following (mathematically equivalent) expression:

$$
\frac{e^{f_{y_i}}}{\sum_j e^{f_j}}
= \frac{Ce^{f_{y_i}}}{C\sum_j e^{f_j}}
= \frac{e^{f_{y_i} + \log C}}{\sum_j e^{f_j + \log C}}
$$

We are free to choose the value of \\(C\\). This will not change any of the results, but we can use this value to improve the numerical stability of the computation. A common choice for \\(C\\) is to set \\(\log C = -\max_j f_j \\). This simply states that we should shift the values inside the vector \\(f\\) so that the highest value is zero. In code:

```python
f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
p = np.exp(f) / np.sum(np.exp(f)) # Bad: Numeric problem, potential blowup

# instead: first shift the values of f so that the highest number is 0:
f -= np.max(f) # f becomes [-666, -333, 0]
p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer

```

**Possibly confusing naming conventions**. To be precise, the *SVM classifier* uses the *hinge loss*, or also sometimes called the *max-margin loss*. The *Softmax classifier* uses the *cross-entropy loss*. The Softmax classifier gets its name from the *softmax function*, which is used to squash the raw class scores into normalized positive values that sum to one, so that the cross-entropy loss can be applied. In particular, note that technically it doesn't make sense to talk about the "softmax loss", since softmax is just the squashing function, but it is a relatively commonly used shorthand.

<a name='svmvssoftmax'></a>

### SVM vs. Softmax

A picture might help clarify the distinction between the Softmax and SVM classifiers:

<div class="fig figcenter fighighlight">
  <img src="/assets/svmvssoftmax.png">
  <div class="figcaption">Example of the difference between the SVM and Softmax classifiers for one datapoint. In both cases we compute the same score vector <b>f</b> (e.g. by matrix multiplication in this section). The difference is in the interpretation of the scores in <b>f</b>: The SVM interprets these as class scores and its loss function encourages the correct class (class 2, in blue) to have a score higher by a margin than the other class scores. The Softmax classifier instead interprets the scores as (unnormalized) log probabilities for each class and then encourages the (normalized) log probability of the correct class to be high (equivalently the negative of it to be low). The final loss for this example is 1.58 for the SVM and 1.04 (note this is 1.04 using the natural logarithm, not base 2 or base 10) for the Softmax classifier, but note that these numbers are not comparable; They are only meaningful in relation to loss computed within the same classifier and with the same data.</div>
</div>

**Softmax classifier provides "probabilities" for each class.** Unlike the SVM which computes uncalibrated and not easy to interpret scores for all classes, the Softmax classifier allows us to compute "probabilities" for all labels. For example, given an image the SVM classifier might give you scores [12.5, 0.6, -23.0] for the classes "cat", "dog" and "ship". The softmax classifier can instead compute the probabilities of the three labels as [0.9, 0.09, 0.01], which allows you to interpret its confidence in each class. The reason we put the word "probabilities" in quotes, however, is that how peaky or diffuse these probabilities are depends directly on the regularization strength \\(\lambda\\) - which you are in charge of as input to the system. For example, suppose that the unnormalized log-probabilities for some three classes come out to be [1, -2, 0]. The softmax function would then compute:

$$
[1, -2, 0] \rightarrow [e^1, e^{-2}, e^0] = [2.71, 0.14, 1] \rightarrow [0.7, 0.04, 0.26]
$$

Where the steps taken are to exponentiate and normalize to sum to one. Now, if the regularization strength \\(\lambda\\) was higher, the weights \\(W\\) would be penalized more and this would lead to smaller weights. For example, suppose that the weights became one half smaller ([0.5, -1, 0]). The softmax would now compute:

$$
[0.5, -1, 0] \rightarrow [e^{0.5}, e^{-1}, e^0] = [1.65, 0.37, 1] \rightarrow [0.55, 0.12, 0.33]
$$

where the probabilites are now more diffuse. Moreover, in the limit where the weights go towards tiny numbers due to very strong regularization strength \\(\lambda\\), the output probabilities would be near uniform. Hence, the probabilities computed by the Softmax classifier are better thought of as confidences where, similar to the SVM, the ordering of the scores is interpretable, but the absolute numbers (or their differences) technically are not.

**In practice, SVM and Softmax are usually comparable.** The performance difference between the SVM and Softmax are usually very small, and different people will have different opinions on which classifier works better. Compared to the Softmax classifier, the SVM is a more *local* objective, which could be thought of either as a bug or a feature. Consider an example that achieves the scores [10, -2, 3] and where the first class is correct. An SVM (e.g. with desired margin of \\(\Delta = 1\\)) will see that the correct class already has a score higher than the margin compared to the other classes and it will compute loss of zero. The SVM does not care about the details of the individual scores: if they were instead [10, -100, -100] or [10, 9, 9] the SVM would be indifferent since the margin of 1 is satisfied and hence the loss is zero. However, these scenarios are not equivalent to a Softmax classifier, which would accumulate a much higher loss for the scores [10, 9, 9] than for [10, -100, -100]. In other words, the Softmax classifier is never fully happy with the scores it produces: the correct class could always have a higher probability and the incorrect classes always a lower probability and the loss would always get better. However, the SVM is happy once the margins are satisfied and it does not micromanage the exact scores beyond this constraint. This can intuitively be thought of as a feature: For example, a car classifier which is likely spending most of its "effort" on the difficult problem of separating cars from trucks should not be influenced by the frog examples, which it already assigns very low scores to, and which likely cluster around a completely different side of the data cloud.

<a name='webdemo'></a>

### Interactive web demo

<div class="fig figcenter fighighlight">
  <a href="http://vision.stanford.edu/teaching/cs231n/linear-classify-demo" style="text-decoration:none;">
    <img src="/assets/classifydemo.jpeg">
  </a>
  <div class="figcaption">We have written an interactive web demo to help your intuitions with linear classifiers. The demo visualizes the loss functions discussed in this section using a toy 3-way classification on 2D data. The demo also jumps ahead a bit and performs the optimization, which we will discuss in full detail in the next section.
  </div>
</div>


<a name='summary'></a>

### Summary

In summary,

- We defined a **score function** from image pixels to class scores (in this section, a linear function that depends on weights **W** and biases **b**).
- Unlike kNN classifier, the advantage of this **parametric approach** is that once we learn the parameters we can discard the training data. Additionally, the prediction for a new test image is fast since it requires a single matrix multiplication with **W**, not an exhaustive comparison to every single training example.
- We introduced the **bias trick**, which allows us to fold the bias vector into the weight matrix for convenience of only having to keep track of one parameter matrix.
- We defined a **loss function** (we introduced two commonly used losses for linear classifiers: the **SVM** and the **Softmax**) that measures how compatible a given set of parameters is with respect to the ground truth labels in the training dataset. We also saw that the loss function was defined in such way that making good predictions on the training data is equivalent to having a small loss.

We now saw one way to take a dataset of images and map each one to class scores based on a set of parameters, and we saw two examples of loss functions that we can use to measure the quality of the predictions. But how do we efficiently determine the parameters that give the best (lowest) loss? This process is *optimization*, and it is the topic of the next section.

<a name='furtherreading'></a>

### Further Reading

These readings are optional and contain pointers of interest.

- [Deep Learning using Linear Support Vector Machines](https://arxiv.org/abs/1306.0239) from Charlie Tang 2013 presents some results claiming that the L2SVM outperforms Softmax.
