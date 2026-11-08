import { LanguageVariant } from '../../../types';
import { pythonBasicsPyTorch } from './python-basics';
import { dataPreprocessingPyTorch } from './data-preprocessing';
import { supervisedLearningPyTorch } from './supervised-learning';
import { unsupervisedLearningPyTorch } from './unsupervised-learning';
import { deepLearningPyTorch } from './deep-learning';
import { modelEvaluationPyTorch } from './model-evaluation';
import { neuralNetworkPyTorch } from './neural-networks';
import { cnnPyTorch } from './cnn';
import { transformerPyTorch } from './transformers';
import { generativeModelPyTorch } from './generative-models';
import { numpyFundamentalsPyTorch } from './numpy-fundamentals';
import { einsumPyTorch } from './einsum';
import { pytorchBasicsPyTorch } from './pytorch-basics';
import { e2eImplementationPyTorch } from './e2e-implementations';
import { reinforcementLearningPyTorch } from './reinforcement-learning';

export const pytorchVariants: Record<string, LanguageVariant> = {
  ...pythonBasicsPyTorch,
  ...dataPreprocessingPyTorch,
  ...supervisedLearningPyTorch,
  ...unsupervisedLearningPyTorch,
  ...deepLearningPyTorch,
  ...modelEvaluationPyTorch,
  ...neuralNetworkPyTorch,
  ...cnnPyTorch,
  ...transformerPyTorch,
  ...generativeModelPyTorch,
  ...numpyFundamentalsPyTorch,
  ...einsumPyTorch,
  ...pytorchBasicsPyTorch,
  ...e2eImplementationPyTorch,
  ...reinforcementLearningPyTorch,
};
