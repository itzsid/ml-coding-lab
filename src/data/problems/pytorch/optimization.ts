import { LanguageVariant } from '../../../types';

/**
 * PyTorch tracks for the problems in ../optimization.ts, keyed by problem id.
 * Code runs against the NumPy-backed torch shim (public/py/torch_shim.py).
 */
export const optimizationPyTorch: Record<string, LanguageVariant> = {};
