# neural-style-transfer

## Introduction

Neural Style Transfer is the process of assembling the content of one image and style of another image. This way, it is possible to transfer the style of an image to a particular content image. So we can transfer the style of a Van Gogh painting to a night sky image like the ones below. It is easy to see the effects but how does it work?

## How does it work?

We do not actually transfer the style to the content image, we generate a noisy image and transfer the style and content to it. The style and content loss are calculated by the difference between certain feature maps and noisy image. As we know CNN layers uses filters and execute the convolution process on an image and generate feature maps for each filter. These feature maps are patterns that the filters extract and the CNNs with the help of pooling layers extract lower level to higher level patterns as the layers proceed.  

There was no need to train a CNN model from scratch since we only need to extract feature maps from the given image. That's why I only used VGG19 model, since the original paper used VGG16 model.

### Transfering content 

Content feature maps lies in higher level CNN layers since those contain the higher level patterns like objects within the image, we do not need the tiny details. Thus I have chosen the second layer of fifth block in the VGG net, since it is the last layer of convolution structure of VGG. And only thing to do to transfer this content to the noisy image is the computation of mean squared error loss of each pixel, then compute the gradient noisy image respect to the content feature map and applying it to the noisy image. 


<p align="center">
  <img src="https://user-images.githubusercontent.com/77073029/221670671-4845382b-0941-487f-ba8b-f545ea0880db.gif" />
</p>


### Transfering style 

Transfering style is a bit more tricky, because style is embedded at both the low level and high level in the image. So the feature maps of each block has an effect on the style. I have used one layer from each convolution block from VGG net. However, this time the mean squared error are computed with the gram matrix of feature maps rather than feature maps themselves. Gram matrix is the matrix multiplication of a matrix's transpose with itself. 


<p align="center">
  <img src="https://user-images.githubusercontent.com/77073029/221694413-b95c628b-ff28-40b5-9390-339cf29abc42.gif" />
</p>


## Experimentation

Since this is a machine learning solution, there is going to be hyperparameter tuning. The hyperparameters are the weights of the content and style layers. From the paper, we know that the content weight must have much higher weight than style weight. But there was a problem, the generated images had high shifts between pixels. To fix this, people have introduced total variation loss which is the difference between neighboring pixels. Therefore at last I had three different hyperparameters: content, style and total variation weight. 

The experimentations can be seen below:

![0_image](https://user-images.githubusercontent.com/77073029/215739543-8cf82d88-471e-44d8-8925-aa2315fccf51.png)

![1_image](https://user-images.githubusercontent.com/77073029/215739617-51a5caf9-2750-49bf-9217-576271ea83fa.png)

![2_image](https://user-images.githubusercontent.com/77073029/215739624-3671f4c3-47bf-42e6-b8f6-e9e02bd8ee40.png)

I have utilized tensorflow hub's neural style transfer for comparison. It can easily be seen from the result that there is still so much more to do.

![pt_generated_image](https://user-images.githubusercontent.com/77073029/215739748-2f375552-2ede-4d8f-9151-435c9a5cd59f.png)

## References

The original neural style transfer paper:

https://arxiv.org/abs/1508.06576

More detailed look on the paper:

https://arxiv.org/abs/1701.01036

Brilliant explanation by Aleksa Gordic:

https://www.youtube.com/watch?v=S78LQebx6jo&list=PLBoQnSflObcmbfshq9oNs41vODgXG-608
