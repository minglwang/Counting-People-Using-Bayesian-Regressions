# Counting-People-Using-Bayesian-Regressions

This project aims to estimate the number of people in images extracted from a pedestrian video. The full video is available on the website [SVCL](http://www.svcl.ucsd.edu/projects/peoplecnt/). 

The procedures of counting people in the image is illustrated below [see this paper](http://visal.cs.cityu.edu.hk/static/pubs/journal/tip12-pedcount.pdf).
<p align ="center">
<img src=https://user-images.githubusercontent.com/45757826/57583937-0fe4ff80-74d6-11e9-9113-41a099988f56.png>
</p>


# Table of Content
- [Feature Extraction](#feature-extraction)
- [Bayesian Regression](#bayesian-regression)
- [Results](#results)
- [How To Use](#how-to-use)

# Feature Extraction

#### motion segmentation
The first step of the system is to segment the scene into the crowd subcomponents of interest. This is accomplished by first using a mixture of dynamic textures
[41] to segment the crowd into subcomponents of distinct motion flow. The video is represented as a collection of spatiotemporal patches, which are modeled as independent samples from a mixture of dynamic textures. The mixture model is learned with the expectation-maximization algorithm. Video locations are then scanned sequentially; a patch is extracted at each location and assigned to the mixture component of the largest posterior probability. Full implementation details are available in 
[DTM](http://www.svcl.ucsd.edu/publications/journal/2008/pami/pami08-dytexmix.pdf).


#### low-rank feature extraction
The following low-rank features are calculated:
- Area: number of pixels in the segment.
- Perimeter: number of pixels on the segment perimeter.
- Perimeter-area ratio: ratio between the segment perimeter and area.
- Blob count: number of connected components, with more than 10 pixels, in the segment.
- Edge length: number of edge pixels in the segment.
- Minkowski dimension: fractal dimension of the internal edges, which estimates the degree of “space-filling” of the edges .
- Homogeneity: the texture smoothness, <img src="https://tex.s2cms.ru/svg/%5Cinline%20%5Csum_%7Bi%2Cj%7D%20p(i%2Cj%20%5Cmid%20%5Ctheta)%2F(1%2B%5Cvert%20i-j%5Cvert)" alt="\inline \sum_{i,j} p(i,j \mid \theta)/(1+\vert i-j\vert)" />.
- Energy: the total sum-squared energy, <img src="https://tex.s2cms.ru/svg/%5Cinline%20%5Csum_%7Bi%2Cj%7D%20p(i%2Cj%20%5Cmid%20%5Ctheta)%5E2" alt="\inline \sum_{i,j} p(i,j \mid \theta)^2" />.
- Entropy: the randomness of the texture distribution, <img src="https://tex.s2cms.ru/svg/%5Cinline%20%5Csum_%7Bi%2Cj%7D%20p(i%2Cj%20%5Cmid%20%5Ctheta)%20%5Clog%20p(i%2Cj%20%5Cmid%20%5Ctheta)" alt="\inline \sum_{i,j} p(i,j \mid \theta) \log p(i,j \mid \theta)" />.

[Back To The Top](#counting-people-ssing-bayesian-regressions)


# Bayesian Regression

#### Bayesian linear regression
Consider a standard linear regression problem, 

<p align = "center">

<img src="https://tex.s2cms.ru/svg/y%20%3D%20%5CPhi%5E%7BT%7D%5Ctheta%2B%5Cepsilon" alt="y = \Phi^{T}\theta+\epsilon" />

</p>

where <img src="https://tex.s2cms.ru/svg/%5Ctheta%20%3D%20%5B%5Ctheta_%7B1%7D%2C%5Ccdots%2C%5Ctheta_%7BD%7D%5D%5E%7BT%7D" alt="\theta = [\theta_{1},\cdots,\theta_{D}]^{T}" /> is the parameter vector, <img src="https://tex.s2cms.ru/svg/y%20%3D%20%5By_%7B1%7D%2C%20%5Ccdots%2C%20y_%7Bn%7D%5D%5E%7BT%7D%20" alt="y = [y_{1}, \cdots, y_{n}]^{T} " /> is the vector of outputs, <img src="https://tex.s2cms.ru/svg/%20%5C%7Bx_%7B1%7D%2C%20%5Ccdots%2C%20x_%7Bn%7D%5C%7D" alt=" \{x_{1}, \cdots, x_{n}\}" /> are the set of corresponding inputs, <img src="https://tex.s2cms.ru/svg/%5Cphi(x_%7Bi%7D)" alt="\phi(x_{i})" /> is a feature transformation, with 

<p align = "center">
<img src="https://tex.s2cms.ru/svg/%20%5CPhi%20%3D%20%5B%5Cphi(x_%7B1%7D)%2C%5Ccdots%2C%20%5Cphi(x_%7Bn%7D)%5D" alt=" \Phi = [\phi(x_{1}),\cdots, \phi(x_{n})]" />
</p>

and <img src="https://tex.s2cms.ru/svg/%5Cespilon%20%3D%5C%7B%5Cepsilon_%7B1%7D%2C%5Ccdots%2C%5Cepsilon_%7Bn%7D%5C%7D%20" alt="\espilon =\{\epsilon_{1},\cdots,\epsilon_{n}\} " /> is a normal random process, i.e. <img src="https://tex.s2cms.ru/svg/%5Cepsilon%20%5Csim%20%5Cmatcal%7BN%7D(0%2C%5CSigma)" alt="\epsilon \sim \matcal{N}(0,\Sigma)" />, with some covariance matrix <img src="https://tex.s2cms.ru/svg/%5CSigma" alt="\Sigma" />.

Now consider the parameter vector <img src="https://tex.s2cms.ru/svg/%5Ctheta" alt="\theta" /> is Gaussian distributed (which known as a Gaussian prior in Bayesian framework)

<p align = "center">
<img src="https://tex.s2cms.ru/svg/%20p(%5Ctheta)%20%3D%20%5Cmathcal%7BN%7D(0%2C%5CGamma)" alt=" p(\theta) = \mathcal{N}(0,\Gamma)" />
</p>

where <img src="https://tex.s2cms.ru/svg/%5CGamma" alt="\Gamma" /> is the covariance matrix. 

Given a training set <img src="https://tex.s2cms.ru/svg/%5Cmathcal%7BD%7D%3D%20%5C%7B(x_1%2Cy_%7B1%7D)%2C%20%5Ccdots%2C%20(x_%7Bn%7D%2C%20y_%7Bn%7D)%5C%7D" alt="\mathcal{D}= \{(x_1,y_{1}), \cdots, (x_{n}, y_{n})\}" />, the posterior distribution is

<p align = "center">

<img src="https://tex.s2cms.ru/svg/p(%5Ctheta%5Cmid%20%5Cmathcal%7BD%7D)%20%3D%20%5Cmathcal%7BN%7D(%5Ctheta%5Cmid%20%5Chat%7B%5Cmu%7D_%7B%5Ctheta%7D%2C%20%5Chat%7B%5CSigma%7D_%7B%5Ctheta%7D)%2C%20" alt="p(\theta\mid \mathcal{D}) = \mathcal{N}(\theta\mid \hat{\mu}_{\theta}, \hat{\Sigma}_{\theta}), " />
</p>

where the posterior mean and covariance

<p align = "center">
<img src="https://tex.s2cms.ru/svg/%5Chat%7B%5Cmu%7D_%7B%5Ctheta%7D%20%3D%20(%5CGamma%5E%7B-1%7D%2B%20%5CPhi%5CSigma%5E%7B-1%7D%5CPhi%5E%7BT%7D%20)%5E%7B-1%7D%5CPhi%5CSigma%5E%7B-1%7Dy%20%5C%5C%0A%5Chat%7B%5CSigma%7D_%7B%5Ctheta%7D%20%3D%20(%5CGamma%5E%7B-1%7D%2B%5CPhi%5CSigma%5E%7B-1%7D%5CPhi%5E%7BT%7D)%5E%7B-1%7D." alt="\hat{\mu}_{\theta} = (\Gamma^{-1}+ \Phi\Sigma^{-1}\Phi^{T} )^{-1}\Phi\Sigma^{-1}y \\
\hat{\Sigma}_{\theta} = (\Gamma^{-1}+\Phi\Sigma^{-1}\Phi^{T})^{-1}." />
</p>

[Back To The Top](#counting-people-ssing-bayesian-regressions)

##### Comparion
For comparison, we have
- maximum a posterior estimate: <img src="https://tex.s2cms.ru/svg/%5Chat%7B%5Ctheta%7D_%7BMAP%7D%20%3D%5Cunderset%7B%5Ctheta%7D%7B%5Carg%20%5Cmax%7D%20%5C%3Ap(%5Ctheta%5Cmid%20%5Cmathcal%7BD%7D)%20%5Cquad%20%5CRightarrow%20%5Cquad%20%5Chat%7B%5Ctheta%7D_%7BMAP%7D%20%3D%5Chat%7B%5Cmu%7D_%7B%5Ctheta%7D" alt="\hat{\theta}_{MAP} =\underset{\theta}{\arg \max} \:p(\theta\mid \mathcal{D}) \quad \Rightarrow \quad \hat{\theta}_{MAP} =\hat{\mu}_{\theta}" /> 
- least square estimate: <img src="https://tex.s2cms.ru/svg/%5Chat%7B%5Ctheta%7D_%7BLS%7D%20%3D(%5CPhi%5CPhi%5E%7BT%7D)%5E%7B-1%7D%20%5CPhi%20y" alt="\hat{\theta}_{LS} =(\Phi\Phi^{T})^{-1} \Phi y" />
- weighted least square estimate: <img src="https://tex.s2cms.ru/svg/%5Chat%7B%5Ctheta%7D_%7BWLS%7D%20%3D(%5CPhi%5CSigma%5E%7B-1%7D%5CPhi%5E%7BT%7D%20)%5E%7B-1%7D%5CPhi%5CSigma%5E%7B-1%7Dy%20" alt="\hat{\theta}_{WLS} =(\Phi\Sigma^{-1}\Phi^{T} )^{-1}\Phi\Sigma^{-1}y " />
- regularized least square estimate: <img src="https://tex.s2cms.ru/svg/%20%5Chat%7B%5Ctheta%7D_%7BRLS%7D%20%3D%20%5Cunderset%7B%5Ctheta%7D%7B%5Carg%20%5Cmax%7D%20%5Cleft%5CVert%20y%20-%5CPhi%5E%7BT%7D%5Ctheta%5Cright%5CVert%5E%7B2%7D%20%2B%20%5Clambda%20%5CVert%20%5Ctheta%20%5CVert_%7B2%7D%5E2%20%20%5Cquad%20%5CRightarrow%20%5Cquad%20%20%5Chat%7B%5Ctheta%7D_%7BRR%7D%20%3D%20(%5CPhi%5CPhi%5E%7BT%7D%2B%5Clambda%20I)%5E%7B-1%7D%5CPhi%20y" alt=" \hat{\theta}_{RLS} = \underset{\theta}{\arg \max} \left\Vert y -\Phi^{T}\theta\right\Vert^{2} + \lambda \Vert \theta \Vert_{2}^2  \quad \Rightarrow \quad  \hat{\theta}_{RR} = (\Phi\Phi^{T}+\lambda I)^{-1}\Phi y" /> 
- LASSO: <img src="https://tex.s2cms.ru/svg/%5Chat%7B%5Ctheta%7D_%7BLASSO%7D%20%3D%20%5Cunderset%7B%5Ctheta%7D%7B%5Carg%20%5Cmax%7D%20%5Cleft%5CVert%20y%20-%5CPhi%5E%7BT%7D%5Ctheta%5Cright%5CVert%5E%7B2%7D%20%2B%5Clambda%20%5CVert%20%5Ctheta%20%5CVert_%7B1%7D%20" alt="\hat{\theta}_{LASSO} = \underset{\theta}{\arg \max} \left\Vert y -\Phi^{T}\theta\right\Vert^{2} +\lambda \Vert \theta \Vert_{1} " />

In addition, the predictive distribution for a novel input <img src="https://tex.s2cms.ru/svg/x_%7B%5Cstar%7D" alt="x_{\star}" />

<p align = "center">
<img src="https://tex.s2cms.ru/svg/f_%7B%5Cstar%7D%20%3D%20%5Cmathcal%7BN%7D(f_%7B%5Cstar%7D%5Cmid%20%5Chat%7B%5Cmu%7D_%7B%5Cstar%7D%2C%20%5Chat%7B%5Csigma%7D_%7B%5Cstar%7D%5E2)%5C%5C%0A%5Chat%7B%5Cmu%7D_%7B%5Cstar%7D%20%3D%20%5Cphi(x_%7B%5Cstar%7D)%5E%7BT%7D%5Chat%7B%5Cmu%7D_%7B%5Ctheta%7D%5C%5C%0A%5Chat%7B%5Csigma%7D_%7B%5Cstar%7D%5E2%20%3D%20%5Cphi(x_%7B%5Cstar%7D)%5E%7BT%7D%5Chat%7B%5CSigma%7D_%7B%5Ctheta%7D%5Cphi(x_%7B%5Cstar%7D)" alt="f_{\star} = \mathcal{N}(f_{\star}\mid \hat{\mu}_{\star}, \hat{\sigma}_{\star}^2)\\
\hat{\mu}_{\star} = \phi(x_{\star})^{T}\hat{\mu}_{\theta}\\
\hat{\sigma}_{\star}^2 = \phi(x_{\star})^{T}\hat{\Sigma}_{\theta}\phi(x_{\star})" />
</p>

Then the predictive distribution of <img src="https://tex.s2cms.ru/svg/y_%7B%5Cstar%7D" alt="y_{\star}" /> can be obtained as

<p align = "center">
<img src="https://tex.s2cms.ru/svg/%20p(y_%7B%5Cstar%7D%5Cmid%20x_%7B%5Cstar%7D%2C%5Cmathcal%7BD%7D)%20%3D%5Cint%20p(y_%7B%5Cstar%7D%5Cmid%20x_%7B%5Cstar%7D%2C%5Ctheta)p(%5Ctheta%5Cmid%5Cmathcal%7BD%7D)d%5Ctheta%20%3D%20%5Cmathcal%7BN%7D(y_%7B%5Cstar%7D%20%5Cmid%20%5Chat%7B%5Cmu%7D_%7B%5Cstar%7D%2C%5Chat%7B%5Csigma%7D_%7B%5Cstar%7D%5E2%2B%5CSigma%5E2)." alt=" p(y_{\star}\mid x_{\star},\mathcal{D}) =\int p(y_{\star}\mid x_{\star},\theta)p(\theta\mid\mathcal{D})d\theta = \mathcal{N}(y_{\star} \mid \hat{\mu}_{\star},\hat{\sigma}_{\star}^2+\Sigma^2)." />
</p>

Using the matrix inverse identity, we can rewrite the predictive mean and variance as

<p align = "center">
<img src="https://tex.s2cms.ru/svg/%5Chat%7B%5Cmu%7D_%7B%5Cstar%7D%20%3D%20%5Cphi_%7B%5Cstar%7D%5E%7BT%7D%5CGamma%20%5CPhi(%5CPhi%5E%7BT%7D%5CGamma%5CPhi%2B%5Csigma%5E%7B2%7DI)%5E%7B-1%7Dy%5C%5C%0A%5Chat%7B%5Csigma%7D_%7B%5Cstar%7D%5E2%20%3D%20%5Cphi_%7B%5Cstar%7D%5E%7BT%7D%5CGamma%5Cphi_%7B%5Cstar%7D-%20%5Cphi_%7B%5Cstar%7D%5E%7BT%7D%5CGamma%20%5CPhi(%5CPhi%5E%7BT%7D%5CGamma%5CPhi%2B%5Csigma%5E%7B2%7DI)%5E%7B-1%7D%5CPhi%20%5CGamma%20%5Cphi_%7B%5Cstar%7D%0A" alt="\hat{\mu}_{\star} = \phi_{\star}^{T}\Gamma \Phi(\Phi^{T}\Gamma\Phi+\sigma^{2}I)^{-1}y\\
\hat{\sigma}_{\star}^2 = \phi_{\star}^{T}\Gamma\phi_{\star}- \phi_{\star}^{T}\Gamma \Phi(\Phi^{T}\Gamma\Phi+\sigma^{2}I)^{-1}\Phi \Gamma \phi_{\star}
" />
</p>

[Back To The Top](#counting-people-ssing-bayesian-regressions)

#### Gaussian process


The "kernel trick" can be applied to the predictive distribution of <img src="https://tex.s2cms.ru/svg/%20f_%7B%5Cstar%7D" alt=" f_{\star}" />, by defining <img src="https://tex.s2cms.ru/svg/k(x_%7Bi%7D%2Cx_%7Bj%7D)%3D%20%5Cphi(x_%7Bi%7D)%5E%7BT%7D%5CGamma%20%5Cphi(x_j)%20" alt="k(x_{i},x_{j})= \phi(x_{i})^{T}\Gamma \phi(x_j) " />, yielding

<p align = "center">
<img src="https://tex.s2cms.ru/svg/%5Chat%7B%5Cmu%7D_%7B%5Cstar%7D%20%3D%20k_%7B%5Cstar%7D%5E%7BT%7D(K%2B%5Csigma%5E%7B2%7DI)%5E%7B-1%7Dy%5C%5C%0A%5Chat%7B%5Csigma%7D_%7B%5Cstar%7D%5E2%20%3D%20k_%7B%5Cstar%5Cstar%7D-%20%5Ck_%7B%5Cstar%7D%5E%7BT%7D(K%2B%5Csigma%5E%7B2%7DI)%5E%7B-1%7Dk_%7B%5Cstar%7D%0A" alt="\hat{\mu}_{\star} = k_{\star}^{T}(K+\sigma^{2}I)^{-1}y\\
\hat{\sigma}_{\star}^2 = k_{\star\star}- \k_{\star}^{T}(K+\sigma^{2}I)^{-1}k_{\star}
" />
</p>

where <img src="https://tex.s2cms.ru/svg/K%3D%5Bk(x_%7Bi%7D%2Cx_%7Bj%7D)%5D_%7Bij%7D%20" alt="K=[k(x_{i},x_{j})]_{ij} " /> is the kernel matrix, <img src="https://tex.s2cms.ru/svg/k_%7B%5Cstar%7D%20%3D%20%5Bk(x_%7B%5Cstar%7D%2Cx_i)%5D_%7Bi%7D%2C%20k_%7B%5Cstar%5Cstar%7D%20%3D%20k(x_%7B%5Cstar%7D%2Cx_%7B%5Cstar%7D)%20" alt="k_{\star} = [k(x_{\star},x_i)]_{i}, k_{\star\star} = k(x_{\star},x_{\star}) " />.

Define <img src="https://tex.s2cms.ru/svg/z%20%3D%20(K%2B%5Csigma%5E2)%5E%7B-1%7Dy" alt="z = (K+\sigma^2)^{-1}y" />, the predictive mean 

<p align = "center">
<img src="https://tex.s2cms.ru/svg/%5Chat%7B%5Cmu%7D(x_%7B%5Cstar%7D)%20%3D%20%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20z_%7Bi%7D%20k(x_%7B%5Cstar%7D%2Cx_%7Bi%7D)%20" alt="\hat{\mu}(x_{\star}) = \sum_{i=1}^{n} z_{i} k(x_{\star},x_{i}) " />
</p>

where <img src="https://tex.s2cms.ru/svg/z_i" alt="z_i" /> is the ith element of <img src="https://tex.s2cms.ru/svg/z" alt="z" />. Hence the regressed function is a linear combination of kernel functions.

[Back To The Top](#counting-people-ssing-bayesian-regressions)


# Results

By applying the linear [methods](#comparion), we define

<p align = "center">
<img src="https://tex.s2cms.ru/svg/%5Cphi(x)%20%3A%3D%20%5Bx_%7B1%7D%2C%5Ccdots%2Cx_%7B9%7D%5D." alt="\phi(x) := [x_{1},\cdots,x_{9}]." />
</p>

where <img src="https://tex.s2cms.ru/svg/x_%7B1%7D" alt="x_{1}" /> -- <img src="https://tex.s2cms.ru/svg/x_%7B9%7D" alt="x_{9}" /> are the low-dimension features.

In this case, we set <img src="https://tex.s2cms.ru/svg/%5CGamma%20%3D%205" alt="\Gamma = 5" /> for Bayesian linear regression.


<p align ="center">
<img src=https://user-images.githubusercontent.com/45757826/57603465-128d3680-7562-11e9-8567-0a7eaeb53cd1.png>
</p>

Figure 1. The test results of different methods:least squares (LS), Regularized LS(RLS), LASSO, Robust Regression (RR), Bayesian linear regression (BR), Gaussian process (GP).

[Back To The Top](#counting-people-ssing-bayesian-regressions)

# How To Use
The MATLAB functions are contained in the repository:

- LS.m -- least square

```Matlab
 % trainx and trainy are the training data sets, phix is can be the polynomial form of [x1,...,x9], 
LS(trainx,trainy,phix);
```

- RLS.m -- regularized LS or Ridge regression

```Matlab
lamd=0.5;
[RLS_yhat,RLS_theta]=RLS(trainx,trainy,lamd,phix);
```
- LASSO.m -- least absolute selection and shrinkage operator
```Matlab
[LASSO_yhat,LASSO_theta]=LASSO(trainx,trainy,lamd,phix);
```
- RR.m -- Robust regression

- BR.m -- Bayesian linear regression

```Matlab
[mean_theta,cov_theta]=BR(gamma,sigma,trainx,trainy,phix);
```
[Back To The Top](#counting-people-ssing-bayesian-regressions)



