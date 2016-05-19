# How to Run
$ python2 mirror.py

You can muck around with config.json if you want to do something special.

# pixelqueer
Gender-bends your face, with support for both MTF and FTM, using an algorithm based on eigenfaces. Useful for trans folks who are curious about how they might look after facial feminization surgery (FFS.)

# Eigenfaces
Eigenfaces form a vector space of facial features. Each eigenface is a basis vector created using Principal Component Analysis (PCA) of a collection of facial images.
Let X be a matrix where each row is a face image, encoded as a vector of pixel intensities.
M is the eigendecomposition of (X^T)X, i.e. the auto-correlation matrix.
Each vector of M is a prinicpal component of the facial images, and its corresponding eigenvalues tell us how important each eigenface is.

Projecting a facial image onto each eigenface gives a vector representing how much of each eigenface is necessary to reconstruct the original image. That is,
Mx = a,
where M is the matrix of male eigenfaces.

For a given image g, we can measure the error of eigenface reconstruction by:
E(M, g) = ||M^(-1).M.g - g||

# Male and Female Eigenfaces
Let X be the collection of female faces, and Y be the collection of male faces.
Correspondingly, let F be the top few eigenfaces of X and let M be the top few eigenfaces of Y.
Lowercase letters x and y are used to refer to representatives of X and Y respectively.

Since male and female facial features are different, we expect E(M, x) > E(F, x), since F will exclusively have eigenfaces which correspond to female features, and vice-versa.

# Male to Female
Let T be the matrix which minimizes
|(F^T) . T . M . x - x| over all x in X.
(i.e. least squares minimization.)

x is a female input image. First, M is applied to it, to project x into the vector space of MALE eigenfaces. E(M, x) would be expected to be high. But if we had a way of mapping the projection onto the vector space of FEMALE eigenface, we'd have a better basis vector that we know COULD get closer to the image (i.e. E(??M, x) < E(M, x).)

So, let's call that matrix T (for trans! :D). Then we'd be able to use the female basis vectors for reconstruction via having T transform the male projection onto the female vector space, using T.

# Usage
Marshal ColorFERET data:
python2 convert\_colorferet.py ../../colorferet ../../faces

Learn:
python2 pixelqueer.py  --source ../../faces --maxEigenfaces 32 --maxImages 128 learn brain/eig\_32\_img\_128.mgs

Run:
python2 pixelqueer.py --direction mtf --image ../../faces/Male/00001\_930831\_fa\_a.ppm run brain/eig\_32\_img\_128.mgs
