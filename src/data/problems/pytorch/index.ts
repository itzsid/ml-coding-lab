import { LanguageVariant } from '../../../types';
import { numpyFundamentalsPyTorch } from './numpy-fundamentals';
import { einsumPyTorch } from './einsum';
import { pytorchBasicsPyTorch } from './pytorch-basics';
import { dataPreprocessingPyTorch } from './data-preprocessing';
import { supervisedLearningPyTorch } from './supervised-learning';
import { unsupervisedLearningPyTorch } from './unsupervised-learning';
import { modelEvaluationPyTorch } from './model-evaluation';
import { neuralNetworkPyTorch } from './neural-networks';
import { optimizationPyTorch } from './optimization';
import { cnnPyTorch } from './cnn';
import { sequenceModelsPyTorch } from './sequence-models';
import { transformerPyTorch } from './transformers';
import { llmInternalsPyTorch } from './llm-internals';
import { representationLearningPyTorch } from './representation-learning';
import { generativeModelPyTorch } from './generative-models';
import { reinforcementLearningPyTorch } from './reinforcement-learning';
import { postTrainingPyTorch } from './post-training';
import { e2eImplementationPyTorch } from './e2e-implementations';

export const pytorchVariants: Record<string, LanguageVariant> = {
  ...numpyFundamentalsPyTorch,
  ...einsumPyTorch,
  ...pytorchBasicsPyTorch,
  ...dataPreprocessingPyTorch,
  ...supervisedLearningPyTorch,
  ...unsupervisedLearningPyTorch,
  ...modelEvaluationPyTorch,
  ...neuralNetworkPyTorch,
  ...optimizationPyTorch,
  ...cnnPyTorch,
  ...sequenceModelsPyTorch,
  ...transformerPyTorch,
  ...llmInternalsPyTorch,
  ...representationLearningPyTorch,
  ...generativeModelPyTorch,
  ...reinforcementLearningPyTorch,
  ...postTrainingPyTorch,
  ...e2eImplementationPyTorch,
};
