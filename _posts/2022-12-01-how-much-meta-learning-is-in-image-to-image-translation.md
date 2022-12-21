---
layout: distill
title: How much meta-learning is in image-to-image translation
description: [Your blog's abstract - a short description of what your blog is about]
date: 2022-12-01
htmlwidgets: true

# anonymize when submitting 
authors:
  - name: Anonymous 

# do not fill this in until your post is accepted and you're publishing your camera-ready post!
# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton 

# must be the exact same name as your blogpost
bibliography: 2022-12-01-distill-example.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: The problem of classification with class-imbalances
---


At the last ICLR conference Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> tested how well CNN architectures transfer the invariance to nuisance transformations between classes of a  classification task. 
- Allan Zhou, Fahim Tajwar, Alexander Robey, Tom Knowles, George J. Pappas, Hamed Hassani, Chelsea Finn [ICLR, 2022] [Do Deep Networks Transfer Invariances Across Classes?]<d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> 

In other words, does a CNN transfer the information that "apples are apples no matter how you rotate them" to images of plums and pears?

The result: not very well. But Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> found they could use an image-to-image translation method DBLP:conf/eccv/HuangLBK18 to learn invariance indirectly. The viability of this method suggests an interesting connection between meta-learning and image-to-image translation.

In this blog post I will:
1. define classification in a class-imbalanced setting.
2. explain how Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> measured invariance transfer.
3. give an overview of generative invariance transfer.
4. clarify the connection to meta-learning.
5. discuss the similarities between image-to-image translation and contemporary meta-learning methods.

Let's go!

## The problem of classification with class-imbalances
First, we need to get acquainted with class-imbalanced classification. We can formalize general classification problems as estimating a distribution

$$\begin{equation}
    \hat{P}_w(y = j|x),
\end{equation}$$

where $ j $ is a class label, $ x $ is a training sample, for instance, an image, and $ y $ is our model's prediction of the image's class. During the training process, we select the model weights $ w $ to minimize the difference between the model and the actual distribution for each example in our training set. Academic datasets often have an equal number of examples for each class. In the real world, this is not always the case. Class-imbalanced classification is classification on class-imbalanced data, meaning the frequencies of class labels $ j $ differ significantly. 

Of course, it is harder for a neural net to learn to correctly recognize the classes with few examples than the classes with many. Still, we might want to perform well on all classes, regardless of size. If we have, for instance, a dataset classifying different types of skin tumors, the overwhelming majority of examples will be benign. However, we should be *especially* good at spotting the rare, malignant ones. Tools designed for regular classification tasks can mislead here: If 99.9% of skin samples were benign and we used the regular accuracy metric, a classifier simply returning "benign" for *any* input would achieve a seemingly great accuracy of 99.9%. There are several more expressive performance measures in such a setting. Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> use a balanced test set, i.e., with equal frequencies of all labels $ j $. 

## 2 Measuring invariance transfer
Transformations are alterations to the data. Nuisance transformations in an image classification setting are alterations that do not affect the class labels of data. A model is invariant to a nuisance transformation if it can successfully ignore the transformation when predicting a label.

We can formally define a nuisance transformation as a distribution over transformation functions:

$$\begin{equation}
    T(\cdot |x)
\end{equation}$$

<img src="{{ site.url }}/public/images/2020-09-01-how-much-meta-learning/TRANSFORM_DIST.svg" alt="an example of a transformation distribution" style="height:450px;display: block;margin-left: auto;margin-right: auto;"/>

A nuisance transformation might be a distribution over rotation matrices of different angles or lighting transformations of different exposure values. At any rate, nuisance transformations have, by definition, no impact on class labels $ y $, only on data $ x $. A perfectly transformation-invariant classifier would completely ignore them, i.e.,

$$\begin{equation}
    \hat{P}_w(y = j|x) = \hat{P}_w(y = j|x'), \; x' \sim T(\cdot |x)
\end{equation}$$

Because this equality will not hold for all classifiers, we need a way to *quantify* how much worse they perform given transformed data. The most straightforward solution is to compare the two distributions using the [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence). Since measuring the true KL divergence would require an infinite amount of test samples, we calculate its empirical expectation:

$$\begin{equation}
    eKLD(\hat{P}_w) = \mathbb{E}_{x \sim \mathbb{P}_{train}, x' \sim T(\cdot|x)} [D_{KL}(\hat{P}_w(y = j|x) || \hat{P}_w(y = j|x'))]
\end{equation}$$

The KL divergence is zero if the two distributions are identical and greater than zero otherwise. *The higher the expected KL-divergence, the more 
 the applied transformation impacts the network's predictions.*

<img alt="expected KL-divergence by class-size" src="{{ site.url }}/public/images/2020-09-01-how-much-meta-learning/EKLD.svg" style="height:450px;display: block;margin-left: auto;margin-right: auto;"/>

Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> run several classification experiments using common CNN architectures and class-imbalanced datasets. They find that class size strongly predicts invariance: Nuisance transformations impact the classification results more strongly on classes with few examples (see figure above). They conclude that CNNs cannot transfer invariance to nuisance transformations between classes.

## 3 Generative Invariance Transfer

To transfer invariance to nuisance transformations to classes with few examples, Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> use an interesting method: They train a Multimodal Unsupervised image-to-image Translation (MUNIT) network DBLP:conf/eccv/HuangLBK18 on the class-imbalanced dataset.

MUNIT networks perform image-to-image translation, meaning they translate an image from one domain, say, pictures of leopards, into another domain, say, pictures of house cats. The resulting translated image should simultaneously look like the image of a real house cat and resemble the original leopard image. For example, if the depicted leopard closes its eyes, the translated image should contain a house cat with closed eyes: Since eye-state is a feature of both domains a good translator will not alter it. On the other hand, a leopard's fur is yellow and spotted, while a house cat can be white, black, grey, or brown. Therefore, to make the translated images indiscernible from real house cats, the translator has to produce house cats in all these colors.

<img alt="an overview of how MUNITs auto-encoding process" src="{{ site.url }}/public/images/2020-09-01-how-much-meta-learning/MUNIT_ENCODING.svg" style="height:450px;display: block;margin-left: auto;margin-right: auto;"/>

MUNIT networks learn to perform translations by correctly distinguishing the *domain-agnostic* features (eye-state) and the *domain-specific* features (distribution of fur color). They embed an image into two latent spaces: A content space encoding the domain-agnostic and a style space encoding the domain-specific features (see figure above).

To transform a leopard into a house cat, we encode the leopard into a content and a style code, throw away the leopard-specific style code, randomly select a cat-specific style code and assemble a similar-looking house cat photo by combining the leopard's content code with the randomly drawn cat style code (see figure below). 

<img alt="an overview of how MUNITs translation process" src="{{ site.url }}/public/images/2020-09-01-how-much-meta-learning/MUNIT_TRANSLATION.svg" style="height:450px;display: block;margin-left: auto;margin-right: auto;"/>

Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite>  modify this process. They do not use MUNIT to transfer images between domains but within a *single domain*. The MUNIT network exchanges the style code of the image with another style code *of the same domain*.  In the example of house cats, this might mean translating a grey house cat into a black one. The difficulty in this single-domain application of MUNIT is to decompose *domain-agnostic content features* from *domain-specific style features* so that the translated images still look like they are from the same domain: In the house cat domain, fur color is a valid style feature, since every house cat has a fur color. However, if the domain contained house cats *and apples*, fur color is not a valid style feature, since we then might generate an apple with black fur. This example illustrates that all valid style features must be exchangeable between *any two images* of the domain.

On a dataset with nuisance transformations, it turns out that the nuisance transformations themselves are valid style features: The result of randomly rotating an image cannot be discerned as artificial when images of all classes, *cats and apples*, were previously randomly rotated.

Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> found that training a MUNIT network on a class-imbalanced dataset with nuisance transformations results in a decomposition of class and transformations. The style latent space of the MUNIT network approximates the transformation distribution $ T(\cdot &#124; x) $. We can then use the MUNIT network to generate images with randomly drawn transformations. Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> use this method to add data to classes with few examples. While still unable to transfer invariance to $ T(\cdot &#124;x) $ between classes, the CNN can now learn it for each class separately using the data generated by MUNIT, which in turn acquired the knowledge of $ T(\cdot &#124;x) $ from the entire dataset.

The knowledge of $ T(\cdot &#124;x) $ extracted by the MUNIT is information that applies to the entire dataset, irrespective of class label - a meta-fact about the dataset. This brings us to meta-learning.


## 4 What does this have to do with meta-learning? 

It is instructive to look at one of the earliest and most prominent definitions of meta-learning to see how meta-learning relates to all discussed previously. In the 1998 book "Learning to learn" Sebastian Thrun & Lorien Pratt define an algorithm as capable of "Learning to learn" if it improves its performance in proportion to the number of tasks it is exposed to:

>an algorithm is said to learn to learn if its performance at each task improves with experience and with the number of tasks. Put differently, a learning algorithm whose performance does not depend on the number of learning tasks, which hence would not benefit from the presence of other learning tasks, is not said to learn to learn. [[Thrun and Pratt]](https://link.springer.com/chapter/10.1007/978-1-4615-5529-2_1).

At first glance, this does not have much to do with our problem as we are transferring invariance between classes of a *single task*. However, it is not necessarily clear that we are performing a single task: Instead of viewing the problem as one multi-classification task, we can also view it as a series of binary classification tasks of the type $ P(y=j_i&#124;x) \forall i \in J $, where J is a set of classes, which the CNN executes jointly. If we use the second interpretation, a meta-learning algorithm should be able to transfer information, for instance, invariance to nuisance transformations, between classes and get better with each class we add to the dataset. Thrun and Pratt continue:

>For an algorithm to fit this definition some kind of *transfer* must occur between multiple tasks that must have a positive impact on expected task-performance [[Thrun and Pratt]](https://link.springer.com/chapter/10.1007/978-1-4615-5529-2_1).

We can see from [Zhou et al.'s [2022]](https://arxiv.org/abs/2203.09739) results that CNNs do not fulfill this criterium. They do not, or not fully, transfer the invariance to nuisance transformations between classes. State-of-the-art meta-learning algorithms do fulfill it at least for some types of transformations. Surprisingly, so do the very differently functioning MUNIT networks. So are MUNIT networks meta-learners? We are finally ready to discuss this question:

## 5 How much meta-learning is in image-to-image translation?

To see how well MUNIT fits the definition of meta-learning, let's define meta-learning more concretely. Contemporary neural-network-based meta-learners are defined in terms of a learning procedure: An outer training loop with a set of trainable parameters iterates over tasks in a  distribution of tasks. Formally a task is comprised of a dataset and a loss function $ \mathcal{T} = \\\{ \mathcal{D}, \mathcal{L} \\\} $. In an inner loop, a learning algorithm is instantiated for each such task based on the outer loop's parameters. It is trained on a training set (*meta-training*) and tested on a validation set (*meta-validation*). The loss on this validation set is then used to update the outer loop's parameters. In this task-centered view of meta-learning, we can express the objective function as

$$\begin{equation}
\underset{\omega}{\mathrm{min}} \; \mathbb{E}_{\mathcal{T} \sim p(\mathcal{T})} \; \mathcal{L}(\mathcal{D}, \omega), 
\end{equation}$$

where $ \omega $ is parameters trained exclusively on the meta-level, i.e., the *meta-knowledge* learnable from the task distribution [[Hospedales et al., 2020]](https://arxiv.org/pdf/2004.05439.pdf).

This *meta-knowledge* is what the meta-learner accumulates and transfers across the tasks in Thrun and Pratt's definition above. Collecting meta-knowledge allows the meta-learner to improve its expected task-performance with the number of tasks. The meta-knowledge in [Zhou et al.'s [2022]](https://arxiv.org/abs/2203.09739) class-imbalanced classification problem is the invariance to the nuisance transformations as the transformations are identical and need to be ignored in all classes. By creating additional transformed samples, the MUNIT network makes the meta-knowledge learnable for the CNN. Contemporary meta-learners, such as the MAML-framework [[Finn et al., 2017]](http://proceedings.mlr.press/v70/finn17a/finn17a.pdf), would, however, also be able to extract and use this meta-knowledge. In this respect, both approaches function similarly.

The task-centered view of meta-learning brings us to a related issue: A meta-learner must discern and decompose task-specific knowledge from meta-knowledge. Contemporary meta-learners decompose meta-knowledge through the different objectives of their inner and outer loops and their respective loss terms. They store meta-knowledge in the outer loop's parameter set $ \omega $ but must not learn task-specific information there. Any unlearned meta-features lead to slower adaptation, negatively impacting performance, *meta-underfitting*. On the other hand, any learned task-specific features will not generalize to unseen tasks in the distribution, thus also negatively impacting performance, *meta-overfitting*.

We recall that, similarly, MUNIT DBLP:conf/eccv/HuangLBK18 decomposes domain-specific style information and domain-agnostic content information. Applied to two domains, leopards and house cats, a MUNIT network will encode the domain-agnostic information, e.g., posture, scale, background, in its content latent space, and the domain-specific information, e.g., how a cat's hair looks, in its style latent space. If the MUNIT network encoded the domain-agnostic information in the style latent space, the resulting image would not appear to be a good translation since the style information is discarded and replaced. A closed-eyed leopard might be turned into a staring cat. If the MUNIT network encoded the domain-specific transformation in the content latent space, the network would have difficulty translating between domains. A house cat might have a leopard's fur.

Both meta-learning and multi-domain unsupervised image-to-image translation are thus learning problems that require a separation of the general from the specific. This is even visible when comparing their formalizations as optimization problems.

[Francheschi et al. [2018]](https://proceedings.mlr.press/v80/franceschi18a/franceschi18a.pdf) show that all contemporary neural-network-based meta-learning approaches can be expressed as bi-level optimization problems. We can formally write the optimization objective of a general meta-learner as:

<div class="blue_highlight_mx-e">
$$
\begin{equation}
   \omega^{*} = \underset{\omega}{\mathrm{argmin}} \sum_{i=1}^{M} \mathcal{L}^{meta}(\theta^{* \; (i)}(\omega), D^{val}_i),
   \end{equation}
$$
</div>

where M describes the number of tasks in a batch, $\mathcal{L}^{meta}$ is the meta-loss function, and $ D^{val}_i $ is the meta-validation set of the task $ i $. $\omega$ represents the parameters exclusively updated in the outer loop. $ \theta^{* \; (i)} $ represents an inner loop learning a task that we can formally express as a sub-objective constraining the primary objective

<div class="yellow_highlight_mx-e">
$$\begin{equation}
   s.t. \; \theta^{* \; (i)} = \underset{\theta}{\mathrm{argmin}} \; \mathcal{L^{task}}(\theta, \omega, D^{tr}_i),
\end{equation}$$
</div>

where $ \theta $ are the model parameters updated in the inner loop, $ \mathcal{L}^{task} $ is the loss function by which they are updated and $ D^{tr}_i $ is the training set of the task $ i $ [[Hospedales et al., 2020]](https://arxiv.org/pdf/2004.05439.pdf).

It turns out that the loss functions of MUNIT can be similarly decomposed:

<img alt="a visualization of MUNIT's GAN loss" src="{{ site.url }}/public/images/2020-09-01-how-much-meta-learning/MUNIT_LOSS.svg" style="height:450px;display: block;margin-left: auto;margin-right: auto;"/>

MUNIT's loss function consists of two adversarial (GAN) [[Goodfellow et al., 2014]](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf) loss terms (see figure above) with several auxiliary reconstruction loss terms. To keep the notation simple, we combine all reconstruction terms into a joined reconstruction loss $ \mathcal{L}_{recon}(\theta_c, \theta_s) $, where $ \theta_c $ are the parameters of the *content* encoding/decoding networks and $ \theta_s $ are the parameters of the *style* encoding/decoding networks. We will only look at one of the two GAN losses in detail since they are symmetric and one is discarded entirely when MUNIT is used on a single domain in the fashion of Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite>.

MUNIT's GAN loss term is

$$
\begin{equation}
\mathcal{L}^{x_{2}}_{GAN}(\theta_d, \theta_c, \theta_s) = \mathbb{E}_{c_{1} \sim p(c_{1}), s_{2} \sim p(s_{2})} \left[ \log (1 -D_ {2} (G_{2} (c_{1}, s_{2}, \theta_c, \theta_s), \theta_d)) \right] + \mathbb{E}_{x_{2} \sim p(x_{2})}  \left[ \log(D_{2} (x_{2}, \theta_d)) \right],
\end{equation}$$

where the $ \theta_d $ represents the parameters of the discriminator network, $p(x_2)$ is the data of the second domain, $ c_1 $ is the content embedding of an image from the first domain to be translated. $ s_2 $ is a random style code of the second domain. $ D_2 $ is the discriminator of the second domain, and $ G_2 $ is its generator. MUNIT's full objective function is:

$$\begin{equation}
        \underset{\theta_c, \theta_s}{\mathrm{argmin}} \; \underset{\theta_d}{\mathrm{argmax}} \;\mathbb{E}_{c_{1} \sim p(c_{1}), s_{2} \sim p(s_{2})} \left[ \log (1 -D_ {2} (G_{2} (c_{1}, s_{2}, \theta_c, \theta_s), \theta_d)) \right]
    \\ + \mathbb{E}_{x_{2} \sim p(x_{2})}  \left[ \log(D_{2} (x_{2}, \theta_d)) \right], + \; \mathcal{L}^{x_{1}}_{GAN}(\theta_d, \theta_c, \theta_s) 
    \\+  \mathcal{L}_{recon}(\theta_c, \theta_s)
\end{equation}$$

(compare DBLP:conf/eccv/HuangLBK18, [[Goodfellow et al., 2014]](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf)).
We can reformulate this into a bi-level optimization problem by extracting a minimization problem describing the update of the generative networks.
We also drop the second GAN loss term as it is not relevant to our analysis. 
<div class="blue_highlight_mx-e">

$$\begin{equation}
    \omega^{*} = \{ \theta_c^*, \theta_s^* \} = \underset{\theta_c, \theta_s}{\mathrm{argmin}} \; \mathbb{E}_{c_{1} \sim p(c_{1}), s_{2} \sim p(s_{2})} \left[ \log (1 -D_ {2} (G_{2} (c_{1}, s_{2}, \theta_c, \theta_s), \theta_d^{*})) \right]
    \end{equation}$$
</div>

$$\begin{equation}
    \\+  \mathcal{L}_{recon}(\theta_c, \theta_s),
\end{equation}$$

We then add a single constraint, a subsidiary maximization problem for the discriminator function:

<div class="yellow_highlight_mx-e">
$$\begin{equation}
   s.t. \; \theta_d^{*} = \underset{\theta_d}{\mathrm{argmax}} \; \mathbb{E}_{c_{1} \sim p(c_{1}), s_{2} \sim p(s_{2})} \left[ \log (1 -D_ {2} (G_{2} (c_{1}, s_{2}, \theta_c, \theta_s), \theta_d)) \right] 
    + \mathbb{E}_{x_{2} \sim p(x_{2})}  \left[ \log(D_{2} (x_{2}, \theta_d)) \right]
    \end{equation}$$
</div>


Interestingly, this bi-level view does not only resemble a meta-learning procedure as expressed above, but the bi-level optimization also facilitates a similar effect. The constraint of maximizing the discriminator's performance punishes style information encoded as content information: Artifacts of the original domain in the translated image that stem from this false decomposition are detected by the discriminator. This is similar to *meta-overfitting*, which meta-learning prevents via its outer optimization loop.

However, the two procedures differ in the way the inner and outer loop parameters affect each other during training. In GAN training, both sets of parameters have a direct mutual influence, i.e., the discriminator function impacts the gradient of the generator function and vice versa. In a meta-learner, the outer loop directly impacts the inner loop. The impact of the inner loop on the outer loop via the meta-validation loss, meanwhile, is indirect. In the case of MAML, the loss is propagated back through the entire learning procedure of the inner loop.

There are also more general reasons not to classify MUNIT as a meta-learner: 
Although MUNIT does improve its ability to extract the transformation distribution $ T(\cdot |x) $ in a single-domain classification setting when further classes are added, it is not itself a classification algorithm. One could speculate that a classification procedure could be integrated using its content space and additional loss terms. Since the content space does not contain any transformation information as a result of the decomposition process, it would even be transformation-invariant. However, these ideas are not extendable to the multi-domain setting.

Examining its primary purpose as an image-to-image translation architecture, we cannot add domains to a MUNIT architecture. MUNIT was designed as a binary translator. For multi-domain translation, many MUNIT networks need to be trained to translate between pairs of domains. Thus, it is not a meta-learner in an image-to-image translation context.

In summary, using a task-centric definition of meta-learning allowed us to recognize that Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> use MUNIT to enable a CNN to learn meta-information via data augmentation. This led us to investigate the parallels between MUNIT and meta-learning algorithms such as MAML. We not only found that both algorithms face the challenge of decomposing task-specific information from meta-information, but also facilitate this through similar, opposing loss terms. 

Understanding that meta-learning shares commonalities with the seemingly unrelated field of image-to-image translation and GANs more broadly is valuable, since ideas from these fields may be applied to meta-learning. Furthermore, a more general intuition about what constitutes meta-learning prompts further research on edge cases such as invariance transfer between imbalanced classes.

## Conclusion

The ICLR paper ["Do Deep Networks Transfer Invariances Across Classes?"](https://arxiv.org/abs/2203.09739) shows that image-to-image translation methods can be used to learn and apply nuisance transformations, enabling a CNN to become invariant to them via data augmentation. This blog post argues that this is a meta-learning setting. In the view of this author, Zhou et al. [2022] <d-cite key="DBLP:conf/iclr/ZhouTRKPHF22"></d-cite> solve a meta-learning problem using a generative method. A closer examination reveals parallels between both types of architecture. 
Further research on the boundary of these related fields and the transfer of ideas between them appears promising. 

*In summary, learning the meta-information of image-to-image translation and meta-learning might train researchers to adapt to new tasks successfully.*
