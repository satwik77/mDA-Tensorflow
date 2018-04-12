## Marginalised Denoising Autoencoders for Nonlinear Respresentations
Tensorflow implementation of the paper [Marginalized Denoising Auto-encoders for Nonlinear Representations][main-paper] (ICML 2014). Other denoising techniques have longer training time and high computational demands. *mDA* addresses the problem by implicitly denoising the raw input via Marginalization and, thus, is effectively trained on *infinitely* many training samples without explicitly corrupting the data. There are similar approaches but they have non-linearity or latent representations stripped away. This addresses the disadvantages of those approaches, and hence is a generalization of those works.

### Requirements
 - Python 2.7
 - Tensorflow
 - NumPy

### Run
To train the demo model :
```sh
python mdA.py 
```

### Demo Results


Resulted filters of first layer during training:  
![Image Filter Gif](https://raw.githubusercontent.com/satwik77/mDA-Tensorflow/master/image-filters.gif?token=AKhAbQmuInoOJ7gkhJq9fxTwXkrh7fEQks5a2NjvwA%3D%3D)  
The filters are continuously improving and learning specialized feature extractors.

### References
 - Chen, Minmin, et al. "Marginalized denoising auto-encoders for nonlinear representations." International Conference on Machine Learning. 2014. [[Paper]][main-paper]
 - Vincent, Pascal, et al. "Extracting and composing robust features with denoising autoencoders." Proceedings of the 25th international conference on Machine learning. ACM, 2008. [[Paper]][da]


   [main-paper]: <http://www.cse.wustl.edu/~mchen/papers/deepmsda.pdf>
   [da]: <http://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf>
