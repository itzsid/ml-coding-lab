import { LanguageVariant } from '../../../types';

/**
 * PyTorch tracks for the problems in ../sequence-models.ts, keyed by problem id.
 * Code runs against the NumPy-backed torch shim (public/py/torch_shim.py).
 */
export const sequenceModelsPyTorch: Record<string, LanguageVariant> = {};
