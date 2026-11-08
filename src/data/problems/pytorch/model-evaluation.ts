import { LanguageVariant } from '../../../types';

/**
 * PyTorch tracks for the problems in ../model-evaluation.ts, keyed by problem id.
 * Code runs against the NumPy-backed torch shim (public/py/torch_shim.py).
 */
export const modelEvaluationPyTorch: Record<string, LanguageVariant> = {};
