# PyTorch track: contract

Every problem can be solved in **NumPy** (the original track) or **PyTorch** (this track). The user picks one with a toggle on the problem page.

Real PyTorch has no WebAssembly build, so it can't run in Pyodide. The browser runs a **NumPy-backed shim** that registers itself as `torch` and implements the subset of the API below, including reverse-mode autograd.

## Files

| What | Where |
|---|---|
| Shim (one file, NumPy only) | `public/py/torch_shim.py`, `install()` registers `torch`, `torch.nn`, `torch.nn.functional`, `torch.nn.init`, `torch.optim`, `torch.distributions` in `sys.modules` |
| Shim unit tests | `scripts/test_torch_shim.py` (compares against real torch when it's importable, otherwise against hand-computed values) |
| PyTorch tracks | `src/data/problems/pytorch/<section-file>.ts`, each exporting `Record<problemId, LanguageVariant>` (already wired into `src/data/problems/index.ts`) |
| Harness (shared by browser and CI) | `public/py/runner.py`: `run_tests(code, tests_json, fn_name, language='pytorch')` |
| Checker | `python3 scripts/validate_solutions.py problems.json [--torch shim\|real]`, also run by `npm test` (`src/__tests__/solutions.test.ts`) |

`LanguageVariant` (see `src/types/index.ts`): `{ starterCode, solution, testCases, hints?, note? }`.

## Rules for PyTorch tracks

1. **Same problem, same function name, same semantics** as the NumPy track. The problem description is shared, so don't change what is asked. If the NumPy track returns a Python float, the torch track returns a Python float too, so most `expected` strings stay identical. Tensors returned by tests are converted automatically: a tensor becomes an ndarray, and a 0-d tensor becomes a scalar via `.item()`.
2. **Tests must be expression-based**: every `input` explicitly calls the function (or the class), e.g. `softmax(torch.tensor([1.0, 2.0, 3.0])).shape`. The bare argument form `'[1, 2, 3]'` is NumPy-only. `torch` and `np` are available in test expressions.
3. **Inputs are tensors**: build them with `torch.tensor(...)`, `torch.randn(...)` etc. Randomness is seeded per test (`np.random.seed(0)` and `torch.manual_seed(0)`).
4. **Wrap booleans with `bool(...)`**, as in the NumPy track: `bool(torch.allclose(out, expected, atol=1e-4))`.
5. **Keep the hidden/visible split**: each track has at least 1 visible and at least 1 hidden test. Check values, not just shapes: compare against an inline reference or a hand-computed constant.
6. **Idiomatic torch, inside the supported API**: the solution should look like what you'd write in real PyTorch, e.g. `F.softmax(x, dim=-1)`, `nn.Linear`, `loss.backward()`. For "implement X from scratch" problems (attention, layer norm, batch norm, conv2d, cross-entropy...), the solution implements it with tensor ops and does **not** call the built-in (`F.scaled_dot_product_attention`, `F.layer_norm`, `nn.Conv2d`...). Mirror that in `note`, e.g. `"Implement with tensor ops; don't use F.layer_norm."`
7. **Starter code** imports what's needed (`import torch`, `import torch.nn as nn`, `import torch.nn.functional as F`) and has the same signature and docstring as the NumPy track, adapted to tensors. It must not pass the tests.
8. **NumPy-specific sections** (NumPy Fundamentals, Python Basics, Einsum) map to the torch equivalents: `torch.zeros/arange/linspace/eye`, tensor indexing, `torch.einsum`, `dim=` instead of `axis=`, `.view/.permute`.
9. Use only the API listed below. If you need something that isn't listed, use it anyway and list it under "API needed" in your report, so the shim can add it. Don't edit the shim yourself.

## Supported API (target surface for the shim)

**Tensor creation:** `tensor, as_tensor, from_numpy, zeros, ones, full, empty, eye, arange, linspace, rand, randn, randint, randperm, normal, zeros_like, ones_like, full_like, rand_like, randn_like, manual_seed`

**dtypes:** `float32 (default float), float64, int64 (default int), int32, bool, long, float`; `.float() .long() .int() .bool() .double() .to(dtype)`; `.dtype`

**Tensor attributes and methods:** `shape, size(dim=None), dim(), ndim, numel(), T, mT, item(), tolist(), numpy(), detach(), clone(), requires_grad, requires_grad_(), grad, grad_fn, is_leaf, backward(gradient=None), zero_() / fill_() / copy_() / normal_() / uniform_() (in-place, under no_grad), contiguous(), to(device_or_dtype) (no-op for 'cpu'), cpu(), device`

**Shape operations:** `view, reshape, flatten(start_dim, end_dim), unflatten, squeeze(dim), unsqueeze, permute, transpose, t, expand, expand_as, repeat, chunk, split, cat, stack, gather, scatter (non-autograd ok), index_select, masked_fill, where, tril, triu, flip, roll, narrow`, plus basic, advanced and boolean indexing and setitem (`x[mask] = 0`)

**Math (elementwise, broadcasting):** `+ - * / // % ** @`, unary `-`, `abs, exp, log, log1p, sqrt, rsqrt, sin, cos, tanh, sigmoid, relu, clamp(min,max), clip, sign, floor, ceil, round, maximum, minimum, pow, reciprocal, lerp`, and comparisons `== != < <= > >=` returning bool tensors, plus `logical_and/or/not, isnan, isinf, isclose, allclose, equal`

**Reductions** (with `dim` as int or tuple, and `keepdim`): `sum, mean, var(unbiased=/correction=), std, max (returns values,indices with dim), min, argmax, argmin, prod, cumsum, logsumexp, norm(p, dim), all, any, topk, sort, argsort, unique, bincount, count_nonzero`

**Linear algebra:** `matmul, mm, bmm, einsum, outer, dot, linalg.norm, linalg.inv, linalg.svd, linalg.eigh, diag, trace`

**Autograd:** `backward`, `torch.no_grad()` (context manager and decorator), `torch.enable_grad`, `torch.autograd.grad(outputs, inputs)` (optional), and `retain_grad()`. Gradients must be correct for every differentiable op above, including broadcasting reductions, indexing and `masked_fill`.

**nn:** `Module` (with `__call__`, `forward`, `parameters()`, `named_parameters()`, `children()`, `modules()`, `state_dict()`, `load_state_dict()`, `zero_grad()`, `train()`, `eval()`, `training`, auto-registration of `Parameter` and submodules, `register_buffer`), `Parameter`, `Sequential`, `ModuleList`, `ModuleDict`, `Linear`, `Bilinear` (optional), `Embedding`, `Conv1d`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`, `Flatten`, `Dropout`, `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `ReLU`, `LeakyReLU`, `GELU`, `SiLU`, `Sigmoid`, `Tanh`, `Softmax`, `Identity`, `MSELoss`, `L1Loss`, `CrossEntropyLoss` (class indices, `label_smoothing` optional), `NLLLoss`, `BCELoss`, `BCEWithLogitsLoss`, `MultiheadAttention` (batch_first, optional)

**nn.functional:** `relu, leaky_relu, gelu, silu, sigmoid, tanh, softmax, log_softmax, dropout, linear, cross_entropy, nll_loss, mse_loss, l1_loss, binary_cross_entropy, binary_cross_entropy_with_logits, kl_div, one_hot, pad, conv1d, conv2d, max_pool2d, avg_pool2d, layer_norm, batch_norm, normalize, cosine_similarity, embedding, scaled_dot_product_attention (is_causal, attn_mask)`

**nn.init:** `zeros_, ones_, constant_, normal_, uniform_, xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, calculate_gain`

**optim:** `SGD(lr, momentum, weight_decay, nesterov)`, `Adam(lr, betas, eps, weight_decay)`, `AdamW`, `RMSprop`, with `zero_grad()`, `step()`, `param_groups`, and `optim.lr_scheduler.StepLR`/`CosineAnnealingLR` (optional)

**distributions:** `Categorical(probs=|logits=)` with `.sample() .log_prob() .entropy()`, and `Normal(loc, scale)` with `.sample() .rsample() .log_prob() .entropy()`

**Not supported (fail with a clear `NotImplementedError("torch shim: X is not supported in the browser")`):** CUDA, `torch.compile`, JIT, sparse tensors, complex dtypes, `DataLoader` (a minimal `utils.data.TensorDataset` + `DataLoader(batch_size, shuffle)` is nice-to-have), and saving or loading files.

**Numerics:** float32 by default like torch (store `np.float32`), so values match real torch to about 1e-6. Integer tensors are int64. Division of ints gives float (true division). Error messages mimic torch where cheap (shape mismatch in matmul, for example).
