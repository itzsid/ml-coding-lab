import { LanguageVariant } from '../../../types';

/**
 * PyTorch tracks for the problems in ../data-preprocessing.ts, keyed by problem id.
 * Code runs against the NumPy-backed torch shim (public/py/torch_shim.py).
 */
export const dataPreprocessingPyTorch: Record<string, LanguageVariant> = {};
