## SMILES-2026 Signal Interference Cancellation
Solution Report, Bondarenko Anna

### Reproducibility instructions

To reproduce the obtained result, just run:

```bash
python applicant_solution.py
```

### The short task description

The received signal is
$$
RX = S + I + \eta = S + F(TX) + E + \eta
$$

where $S$ is the desired signal ($N \times 4$ matrix), $TX$ is a transmitted signal ($N \times 6$ matrix), so 
$F(TX)$ is structured interference from transmission, $E$ is an external interference term that is spatially coherent, so it can be represented as a rank-1 matrix, and finally $\eta$ is noise.

We want to cancel all structured interference: TX-driven part $F(TX)$ and external rank-1 part $E$.

### TX-driven part

The baseline solution predicts the transmission interference by solving a ridge regression problem. The code constructs 
nonlinear features with lags from $TX$ and gets a matrix $X$. Then, for each channel $c$, it solves

$$
\beta_c = \arg\min_{\beta_c \in \mathbb{C}^{130}}
\left\|X\beta_c - y_c\right\|_2^2 + 10^{-6}\left\|\beta_c\right\|_2^2
$$

Equivalently,

$$
\beta_c = \left(X^*X + 10^{-6}I\right)^{-1}X^*y_c
$$

Because the scorer validates the removed signal using this method and a specific set of nonlinear TX features, we kept the TX-driven 
interference model unchanged and used the provided `fit_tx_prediction` function in our solution.

### Rank-1 Approximation

The provided baseline subtracts only the TX-driven part, which is computed with `fit_tx_prediction`. So the first natural improvement is to
add a rank-1 part for the coherent external component. The task code already has helper function `rank1_from_band_matrix`, but we
wrote our own implementation `rank1_approx` that is simpler and equivalent here.

For rank-1 approximation we use the SVD. If our matrix is $A$, the best approximation is 
$$
A \approx u_1 \sigma_1 v_1^*,
$$
where $\sigma_1$ is the largest singular value and $u_1$ and $v_1$ are corresponding vectors.
To compute them we use the spectral decomposition of the Gram matrix. It works because if $A = U \Sigma V^*$ then 
$A^*A = V \Sigma^* \Sigma V^*$ is a spectral decomposition, so we can get $v_1$ as the principal eigenvector for $A^*A$. After that 
we can compute $Av_1 = u_1 \sigma_1$ so the desired approximation is
$$
u_1 \sigma_1 v_1^* = (A v_1) v_1^*
$$

The implementation is:

```python
def rank1_approx(band_matrix):
    cov = band_matrix.conj().T @ band_matrix
    _, vecs = np.linalg.eigh(cov)
    v = vecs[:, -1]
    return (band_matrix @ v)[:, None] * v.conj()[None, :]
```

### Solution steps and experiments

1. So our first solution version was:

```python
rx_hat = baseline(tx_n, rx, helpers["fit_tx_prediction"])
band_residual = np.column_stack(
    [helpers["score_filter"](rx_hat[:, ch]) for ch in range(rx_hat.shape[1])]
)
rx_hat = rx_hat - rank1_approx(band_residual)
```

This improved the score from the baseline value of about `4.02 dB` to about
`7.01 dB`.

2. Then we tried multiplying the rank-1 component by a scalar before subtraction.
Values around `0.9` to `1.0` were very similar. The best tested value in that stage
was `0.94`, but the improvement was small:

```text
rank-1 scale 1.00: about 7.01 dB
rank-1 scale 0.94: about 7.03 dB
```

So the scaling at that step wasn't the main source of improvement.

3. Changing the Order

The next idea was to change the order of the stages. Instead of subtracting the rank-1 part after the TX fit, we did it before.
Surprisingly, it worked much better and the score went above `8 dB`.

It seems like the coherent external component is stronger and removing it first makes the later TX fit cleaner.

4. Some invalid attempts.

Some sequential orders were not valid according to the scorer. In particular,
pipelines like:

```text
TX fit -> rank-1 -> TX fit
TX fit -> TX fit -> rank-1
```

often failed the explainability or residual guard. They removed energy, but the
removed part was no longer explained well enough as the allowed combination of a
TX-driven component plus one coherent rank-1 component.

5. Two TX fits. 

Interestingly, the second TX fit helped a lot, and the version

```text
rank-1 -> TX fit -> TX fit
```

was really good. The best implementation uses three scalar coefficients:

```text
rank-1 scale = 0.85
first TX scale = 0.85
second TX scale = 0.85
```

A full second TX subtraction was too aggressive and failed validity, but a scaled second pass was stable and gave a large gain.

The final score here was already about `10.7 dB`.
Per channel:

```text
ch0: 11.76 dB
ch1:  9.03 dB
ch2: 10.93 dB
ch3: 11.14 dB
```

The main improvement came from using the rank-1 component first, and then
applying two careful TX-fit subtractions.

6. Channel-specific coefficients.

We noticed that the per-channel scores were not balanced. Channel 1 was
clearly weaker than the others, as you can see above. That gave an idea to change
coefficients (scale values) for the rank-1 component and both TX-fit passes specifically for ch1.

By searching the grid we selected such values:

```text
rank-1 scale: [0.85, 0.70, 0.85, 0.85]
TX pass 1:    [0.85, 0.65, 0.85, 0.85]
TX pass 2:    [0.85, 0.65, 0.85, 0.85]
```

This improved channel 1 without hurting the other channels. The updated score was `10.9773 dB`. Per channel:

```text
ch0: 11.76 dB
ch1: 10.08 dB
ch2: 10.93 dB
ch3: 11.14 dB
```
7. The score was already close to `11 dB`, so we decided to experiment more with TX passes and
add the third TX fit with a small coefficient. The best simple coefficient was $0.5$ and we got a final score of about `11.5 dB`.

### Final result 

We cancel structured interference from the signal by subtracting the rank-1 coherent component and three iterations of TX fit 
with different scales. This gives the final score value `11.51 dB`. 
