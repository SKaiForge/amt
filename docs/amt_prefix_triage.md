# AMT Prefix Triage

## Summary

The current result does not support H1 yet, but it also does not show that one AMT token lacks capacity. Too few training examples contribute to the failure, while the experiment is also confounded by its split, label distribution, initialization, encoder, projector size, and scoring.

The observed PyTorch-heavy predictions are consistent with majority-class or pretrained-prior collapse rather than reliable use of the supplied memory token.

## Findings

### 1. The train/test split is not a paraphrase split

The proposed experiment trains on `_00` examples and tests on `_01` examples. The generator selects the memory with:

```python
primary = ml_memories[query_number % len(ml_memories)]
```

Consequently, `_00` and `_01` frequently use different postulates and answers instead of holding the postulate fixed and changing only the query wording. In 8 of 10 smoke users, the target changes between training and testing.

Observed label distributions:

```text
Training:
PyTorch       3
scikit-learn  3
Julia         2
Python        1
JAX           1

Testing:
PyTorch       5
TensorFlow    2
Python        1
Julia         1
scikit-learn  1
```

TensorFlow is absent from training. A constant PyTorch prediction already scores 50% on this test set.

### 2. Profiles contain competing preferences in the same scope

A user can have several memories with `scope="ml"`, while questions ask for "the" preference. The selected answer depends on shuffled memory order and `query_number`, not on a single canonical preference.

This makes the synthetic user profile internally ambiguous for the initial H1 experiment.

### 3. Qwen's prior appears to dominate the prefix

Predictions such as `PyTorch Lightning` are not benchmark labels. They suggest Qwen is responding to its pretrained association between ML frameworks and PyTorch rather than decoding the supplied AMT token.

A causal memory test must keep the query fixed and swap only the memory token.

### 4. Zero initialization initially erases all memory differences

The current projector initializes its final layer to zero:

```python
nn.init.zeros_(self.projector[-1].weight)
nn.init.zeros_(self.projector[-1].bias)
```

At initialization, every postulate maps to the same zero token:

```text
Python postulate     -> zero vector
Julia postulate      -> zero vector
TensorFlow postulate -> zero vector
```

The first projector layer also receives zero gradient on the first optimization step because the final layer weights are zero. Memory discrimination can begin only after the final layer changes.

### 5. The projector is too large for ten observations

For Qwen hidden width $D$, the two dense layers contain approximately:

$$
2D^2
$$

parameters. At $D=4096$, this is roughly 33.5 million trainable weights. Ten examples are insufficient to establish that this mapping generalizes.

### 6. The memory encoder is weak

The current encoder performs:

```text
token embeddings -> mean pooling -> MLP
```

Mean pooling discards token order and contextual structure. Similar postulates differ in only one or two tokens, while most of their pooled representation is shared:

```text
The user prefers Python for ML experiments.
The user prefers Julia for ML experiments.
```

A stronger reader-matched encoder would use frozen lower Qwen transformer layers before pooling.

### 7. The metric can be too permissive

A substring or framework-only check may mark this as correct:

```text
Gold:       PyTorch for ML experiments
Prediction: PyTorch Lightning
```

For the strict H1 claim, this should be wrong. Report at least two metrics:

- strict normalized answer accuracy
- framework-value accuracy

### 8. Four-bit frozen-Qwen backpropagation is plausible but unverified

Freezing Qwen weights does not prevent gradients from flowing through Qwen operations to the prefix embedding. BitsAndBytes layers generally support input gradients with frozen quantized weights.

However, the actual Colab setup should log dtypes, prefix norms, loss, gradient norms, and parameter changes to confirm the path works numerically.

## Discriminating Experiments

### Experiment 1: One-example overfit

Train repeatedly on one fact:

```text
Memory: User prefers Julia for ML experiments.
Query: Which framework is preferred?
Answer: Julia
```

Log:

- loss
- prefix norm
- projector parameter change
- gradient norm for each projector layer

If the model cannot overfit one example, the problem is architecture, gradient flow, quantization, or prompting rather than dataset size.

### Experiment 2: Free-vector control

Bypass the memory encoder and projector. Optimize one standalone prefix vector:

```python
memory_token = nn.Parameter(
    torch.randn(1, 1, hidden_size) * 0.02
)
```

If this succeeds while `QwenPrefixAMT` fails, frozen Qwen can consume a continuous prefix and the failure lies in the encoder/projector.

### Experiment 3: Counterfactual memory test

Keep the query identical and switch only the memory:

```text
Python memory     -> Python
Julia memory      -> Julia
JAX memory        -> JAX
TensorFlow memory -> TensorFlow
```

If predictions remain unchanged, Qwen is ignoring the prefix.

### Experiment 4: Memory ablations

Compare:

```text
correct AMT token
wrong AMT token
zero AMT token
random AMT token
no AMT token
```

H1 requires the correct memory condition to perform materially better.

### Experiment 5: Text-memory control

Give Qwen the same gold postulate as ordinary text using the identical question and answer format. If text memory fails, the prompt or generation setup is broken. If text memory succeeds and AMT fails, the problem is the latent interface.

### Experiment 6: Quantization control

Run the one-example test with identical seeds under:

```text
BF16 or FP16 frozen reader
4-bit frozen reader
```

Compare loss curves and prefix gradient norms to determine whether quantized layers transmit useful input gradients.

### Experiment 7: True paraphrase split

Hold the postulate and answer fixed, changing only the query wording:

```text
Train query: What does the user prefer for ML experiments?
Test query:  Which framework matches the user's preference?
```

Only this setup measures query-paraphrase generalization.

### Experiment 8: Data scaling

After the one-example and free-vector controls succeed, run balanced training sizes:

```text
1
10
50
100
1,000
```

Each framework value must appear equally often across training and evaluation.

## Recommended Order

1. Fix the dataset so a user has one canonical ML preference for this split.
2. Add a true paraphrase split.
3. Run one-example overfit.
4. Run the free-vector control.
5. Run correct/wrong/zero/random/no-memory ablations.
6. Replace zero initialization with an identity-preserving initialization.
7. Replace mean-only encoding with frozen lower Qwen layers.
8. Scale balanced training data.
9. Compare prefix injection with the intended cross-attention AMT architecture.

## Conclusion

The current evidence primarily shows that a zero-initialized, approximately 33.5-million-parameter prefix projector trained on ten confounded examples collapses toward Qwen's prior. It does not establish that one atomic postulate cannot fit in one latent token.

The next decisive experiment is a one-example free-vector overfit test followed by counterfactual memory swapping. Together, these isolate whether frozen Qwen can consume one continuous memory token before evaluating the postulate encoder or cross-attention design.
