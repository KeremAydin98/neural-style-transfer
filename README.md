# neural-style-transfer

Neural Style Transfer is the process of assembling the content of one image and style of another image. This way, it is possible to transfer the style of an image to a particular content image. So we can transfer the style of a Van Gogh painting to a night sky image like the ones below. It is easy to see the effects but how does it work?

We do not actually transfer the style to the content image, we generate a noisy image and transfer the style and content to it. The style and content loss are calculated by the difference between certain feature maps and noisy image. As we know CNN layers uses filters and execute the convolution process on an image and generate feature maps for each filter. These feature maps are patterns that the filters extract and the CNNs with the help of pooling layers extract lower level to higher level patterns as the layers proceed.  

![0_image](https://user-images.githubusercontent.com/77073029/215739543-8cf82d88-471e-44d8-8925-aa2315fccf51.png)

![1_image](https://user-images.githubusercontent.com/77073029/215739617-51a5caf9-2750-49bf-9217-576271ea83fa.png)

![2_image](https://user-images.githubusercontent.com/77073029/215739624-3671f4c3-47bf-42e6-b8f6-e9e02bd8ee40.png)

![pt_generated_image](https://user-images.githubusercontent.com/77073029/215739748-2f375552-2ede-4d8f-9151-435c9a5cd59f.png)
