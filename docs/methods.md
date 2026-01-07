# Causal Forests with Fixed Effects (CFFE)

## Methods Note

This implementation is based on:

> **Kattenberg, M.A.C., Scheer, B.J., & Thiel, J.H. (2023).** *Causal Forests with Fixed Effects for Treatment Effect Heterogeneity in Difference-in-Differences.* CPB Discussion Paper, Netherlands Institute for Economic Policy Analysis.

We provide a Python implementation of their methodology, which was originally released as an R package.

### 1. Introduction

We propose a Causal Forest with Fixed Effects (CFFE) estimator for heterogeneous treatment effects in panel and difference-in-differences (DiD) settings. Standard causal forests assume independent observations and fail in the presence of unit and time fixed effects. Naively including fixed-effect dummies leads to spurious splits and invalid inference.

Our estimator modifies the causal forest algorithm to partial out fixed effects locally within each tree node, ensuring orthogonality between treatment assignment and fixed effects at every stage of tree construction.

### 2. Setup and Notation

Let:
- $i = 1, \dots, N$ index units
- $t = 1, \dots, T$ index time
- $Y_{it}$ outcome
- $D_{it}$ treatment (binary or continuous)
- $X_{it} \in \mathbb{R}^p$ covariates
- $\alpha_i$ unit fixed effects
- $\gamma_t$ time fixed effects

We assume the potential outcome model:

$$Y_{it} = \alpha_i + \gamma_t + \tau(X_{it}) D_{it} + \varepsilon_{it}$$

with $\mathbb{E}[\varepsilon_{it} \mid X_{it}, i, t] = 0$.

The target estimand is the Conditional Average Treatment Effect (CATE):

$$\tau(x) = \mathbb{E}[Y_{it}(1) - Y_{it}(0) \mid X_{it} = x]$$

### 3. Why Standard Causal Forests Fail with Fixed Effects

Standard causal forests (Athey & Imbens, 2016; Wager & Athey, 2018):
- Assume i.i.d. observations
- Choose splits by maximizing treatment effect heterogeneity
- Estimate leaf effects using local regression

In panel data:
- Outcomes are correlated within units
- Fixed effects induce strong, non-causal heterogeneity
- Tree splits align with unit/time identifiers
- Treatment effect estimates become biased and unstable

Including fixed-effect dummies does not solve this problem, because splitting is still driven by FE-induced variation.

### 4. CFFE Algorithm Overview

CFFE modifies the causal forest in two fundamental ways:
1. Node-level fixed-effect orthogonalization
2. Cluster-aware honesty and subsampling

Crucially, fixed effects are removed inside each node, not globally.

### 5. Node-Level Fixed-Effect Orthogonalization

Consider a node containing observations $S \subset \{(i,t)\}$.

We define residualized variables:

$$\tilde{Y}_{it} = Y_{it} - \hat{\alpha}_i^{(S)} - \hat{\gamma}_t^{(S)}$$

$$\tilde{D}_{it} = D_{it} - \hat{\alpha}_{D,i}^{(S)} - \hat{\gamma}_{D,t}^{(S)}$$

where fixed effects are estimated using only observations in $S$.

**Implementation:** We compute $\tilde{Y}$ and $\tilde{D}$ via alternating projections (two-way within transformation), iterating:
1. Demean by unit
2. Demean by time

This procedure is repeated until convergence (fixed small number of iterations, default 5).

### 6. Split Criterion

For a candidate split $S \to (S_L, S_R)$, we estimate treatment effects separately in each child node:

$$\hat{\tau}_L = \frac{\sum_{(i,t) \in S_L} \tilde{D}_{it} \tilde{Y}_{it}}{\sum_{(i,t) \in S_L} \tilde{D}_{it}^2}$$

$$\hat{\tau}_R = \frac{\sum_{(i,t) \in S_R} \tilde{D}_{it} \tilde{Y}_{it}}{\sum_{(i,t) \in S_R} \tilde{D}_{it}^2}$$

The split score is:

$$\Delta(S_L, S_R) = \frac{|S_L| |S_R|}{(|S_L| + |S_R|)^2} (\hat{\tau}_L - \hat{\tau}_R)^2$$

This criterion directly targets heterogeneity in treatment effects, not outcome variance.

### 7. Leaf Estimation

For any terminal node $S$, the treatment effect estimate is:

$$\hat{\tau}_S = \frac{\sum_{(i,t) \in S} \tilde{D}_{it} \tilde{Y}_{it}}{\sum_{(i,t) \in S} \tilde{D}_{it}^2}$$

This corresponds to a within-node fixed-effects regression of $Y$ on $D$.

### 8. Honesty and Subsampling

To ensure valid inference:
- Each tree is grown using an honest sample split:
  - One subsample for splitting
  - One for estimation
- Subsampling is clustered at the unit level

This preserves dependence within units while enabling asymptotic normality.

### 9. Forest Aggregation

Let $\hat{\tau}_b(x)$ denote the estimate from tree $b$. The forest estimate is:

$$\hat{\tau}(x) = \frac{1}{B} \sum_{b=1}^{B} \hat{\tau}_b(x)$$

### 10. Inference

We use half-sample variance estimation:

$$\hat{V}(x) = \frac{1}{B/2} \sum_{b=1}^{B/2} (\hat{\tau}_b(x) - \bar{\tau}(x))^2$$

which is valid under cluster-level sampling and honesty.

Confidence intervals are constructed using normal approximation:

$$\hat{\tau}(x) \pm z_{1-\alpha/2} \sqrt{\hat{V}(x)}$$

### 11. Relation to Existing Methods

- Reduces to standard causal forest when no fixed effects are present
- Generalizes DiD estimators to allow for heterogeneity
- Distinct from residualized or dummy-variable approaches

### 12. Implementation

The estimator is implemented in Python with:
- C++/pybind11 core for performance
- Node-level FE residualization
- Cluster-aware sampling
- Honest trees

## References

- **Kattenberg, M.A.C., Scheer, B.J., & Thiel, J.H. (2023).** Causal Forests with Fixed Effects for Treatment Effect Heterogeneity in Difference-in-Differences. *CPB Discussion Paper*. — **Primary reference for this implementation.**
- Athey, S., & Imbens, G. (2016). Recursive Partitioning for Heterogeneous Causal Effects. *PNAS*.
- Wager, S., & Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. *JASA*.
