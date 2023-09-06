# Recommender System Algorithms

Recommender system algorithms implemented in Python. It includes four different recommendation algorithms: Matrix Factorization model, User-Based Collaborative Filtering, Item-Based Collaborative Filtering, and Dimensionality Reduction Collaborative Filtering.

## Install required packages:

```bash
conda install numpy pandas scikit-learn scipy
```

## Algorithms

- [Matrix Factorization](#matrix-factorization)
- [User-Based Collaborative Filtering](#user-based-collaborative-filtering)
- [Item-Based Collaborative Filtering](#item-based-collaborative-filtering)
- [Dimensionality Reduction Collaborative Filtering](#dimensionality-reduction-collaborative-filtering)


### Matrix Factorization

The Matrix Factorization model is a collaborative filtering technique used for recommender systems. This algorithm uses the idea of machine learning to learn latent factors which captures the underlying patterns of user-item interactions.

1. **Initialization:** The model takes as input a $m \times n$ user-item rating matrix $R$ where rows represent users, columns represent items, and the values are user ratings.
2. **Factorization:** The user-item rating matrix $R$ is factorized into a lower-dimensional $m \times k$ user matrix $U$, and an $n \times k$ item matrix $V$, where $k \ll \min\{m, n\}$. These matrices will be initialized with random values.
$$R \approx UV^T$$

3. **Training:** The goal of optimazation is to minize error J:
$$J = \frac{1}{2} ||R - UV^T||^2$$
Since not all items are rated in the matrix R, we only consider the items already been rated. For each observed rating in the training data, the model computes a predicted rating by taking the dot product of the corresponding user $u_i$ and item $v_j$ latent factor vectors.The predicted rating of user $i$ to item $j$ is denoted as $\hat{r}_{ij}$.

$$\hat{r_{ij}} = \sum_{i,j}^{k} u_{i} \cdot v_{j}$$

Loss function:
$$J = \frac{1}{2} \sum_{i,j} \left( r_{ij} - \sum_{i,j}^{k} u_{i} \cdot v_{j} \right)^2$$

Calculate the Gradient for $u$ and $v$:
$$\frac{\partial J}{\partial u_{i}} = \sum_{i,j} (r_{ij} - \sum_{i,j}^{k} u_{i} \cdot v_{j}) \cdot (-v_{jk}) = \sum_{i,j} e_{ij} \cdot (-v_{jk})$$

$$\frac{\partial J}{\partial v_{j}} = \sum_{i,j} (r_{ij} - \sum_{i,j}^{k} u_{i} \cdot v_{j}) \cdot (-u_{ik}) = \sum_{i,j} e_{ij} \cdot (-u_{jk})$$

Update $u$ and $v$ with learning rate $\alpha$:
$$u_{ik} := u_{ik} + \alpha \sum_{i,j} e_{ij} \cdot (-v_{jk})$$
$$v_{jk} := v_{jk} + \alpha \sum_{i,j} e_{ij} \cdot (-u_{ik})$$

4. **Evaluation:**  The model's performance is evaluated using metrics such as Root Mean Square Error (RMSE) and Mean Absolute Error (MAE) by comparing predicted ratings to actual ratings.
$$\text{RMSE} = \sqrt{\frac{1}{n} (R - \hat{R})^2}$$
$$\text{MAE} = \frac{1}{n} |R - \hat{R}|$$

## User-Based Collaborative Filtering
User-Based Collaborative Filtering computes recommendations based on user similarity.  then recommends items that those similar users have rated highly.
1. Calculate **Pearson Similarity** to find users with similar tastes to the target user, this is based on the items they have both rated.

$${Similarity}(u, v) = \frac{\sum_{i}(r_{ui} - \mu_u)(r_{vi} - \mu_v)}{\sqrt{\sum_{i}(r_{ui} - \mu_u)^2} \sqrt{\sum_{i}(r_{vi} - \mu_v)^2}}$$

2. The **k-nearest neighbors $N(u)$** for a target user $u$ are found based on the Pearson similarity scores. 

3. Predict the potential ratings. Since different users may have different rating scale, ratings are normalized first.
$$\hat{r_{ui}} = \mu_u + \frac{\sum_{v \in N(u)} \text{similarity}(u, v) \cdot (r_{vi} - \mu_v)}{\sum_{v \in N(u)} |\text{similarity}(u, v)|}$$

## Item-Based Collaborative Filtering
Item-Based Collaborative Filtering
Item-Based Collaborative Filtering focuses on item similarity. It identifies items similar to those the user has interacted with and recommends items that are related to the user's past preferences.

1. Calculate **Adjusted Cosine Similarity** between two items, $i$ and $j$, this is based on the users who have rated both items.
$${similarity}(i, j) = \frac{\sum_{u}(r_{ui} - \mu_i)(r_{uj} - \mu_j)}{\sqrt{\sum_{u}(r_{ui} - \mu_i)^2} \sqrt{\sum_{u}(r_{uj} - \mu_j)^2}}$$

2. The **k-nearest neighbors $N(i)$** for a target item $i$ are found based on the Adjusted Cosine Similarity.

3. Predict the potential ratings.
$$\hat{r_{ui}} = \frac{\sum_{j \in N(i)} \text{similarity}(i, j) \cdot r_{uj}}{\sum_{j \in N(i)} |\text{similarity}(i, j)|}$$


## Dimensionality Reduction Collaborative Filtering
Dimensionality Reduction techniques such as Singular Value Decomposition (SVD), reduce the dimensionality of the user-item interaction matrix. This can help improve recommendation quality and reduce computational complexity.
1. Fill missing entries in the ratings matrix $R$ with the mean ratings $\mu$. This results in a filled matrix $F$ of the same size as $R$.

2. Perform **Singular Value Decomposition (SVD)** on the filled matrix $F$. SVD decomposes $F$ into three matrices: $U$, $S$, and $V^T$:
$$F = U \cdot S \cdot V^T$$

- $U$ is an $M \times D$ matrix representing user latent factors.
- $S$ is a diagonal $D \times D$ matrix representing singular values in decreasing order.
- $V^T$ is a $D \times N$ matrix representing item latent factors.

To reduce the dimensionality to $D'$, where $D' < D$, select the first $D'$ columns of matrices $U$ and $S$, resulting in $U'$ and $S'$:
$$U' = U[:, :D']$$
$$S' = S[:D', :D']$$
The reduced matrix $R'$ is obtained by multiplying $U'$ and $S'$:
$$R' = U' \cdot S'$$
$R'$ represents the reduced-dimensional approximation of the original ratings matrix $R$.
