# Fourier and Edge Loss for GAN-Based Single Image Super Resolution

This repository is part of the work I did during my year at Osaka University
with the exchange program with FrontierLab@OsakaU. It was also presented at 
MIRU 2018 in Sapporo on August 8th during the poster session.
The work was done with the supervision and valuable help of Matsushita Yasuyuki
and Michael Waechter.

This repositery contains the script for training different type of Super 
Resolution algorithm. A script for generating High resolution images from a 
pretrained network. Pretrained Networks. The Texture Data set used for training.
The poster and the extended abstract presented at MIRU are also available.

The Idea of this contribution is to guide the GAN convergence using additional losses that focus on aspects the adversarial and Mean Squared Error (MSE) loss struggle. The Loss that showed success for that tasks were a Sobel Loss (1st order derivative of the image) and a frequency spectrum loss.
The architecture comes for the most part from the article SRGAN by C. Ledig with a few improvements from the general advances of the machine learning community. 