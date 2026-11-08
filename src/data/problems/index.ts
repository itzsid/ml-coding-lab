import { Problem } from '../../types';
import { pytorchVariants } from './pytorch';
import { numpyFundamentalsProblems } from './numpy-fundamentals';
import { einsumProblems } from './einsum';
import { pytorchBasicsProblems } from './pytorch-basics';
import { dataPreprocessingProblems } from './data-preprocessing';
import { supervisedLearningProblems } from './supervised-learning';
import { unsupervisedLearningProblems } from './unsupervised-learning';
import { modelEvaluationProblems } from './model-evaluation';
import { neuralNetworkProblems } from './neural-networks';
import { optimizationProblems } from './optimization';
import { cnnProblems } from './cnn';
import { sequenceModelsProblems } from './sequence-models';
import { transformerProblems } from './transformers';
import { llmInternalsProblems } from './llm-internals';
import { representationLearningProblems } from './representation-learning';
import { generativeModelProblems } from './generative-models';
import { reinforcementLearningProblems } from './reinforcement-learning';
import { postTrainingProblems } from './post-training';
import { e2eImplementationProblems } from './e2e-implementations';

// Curriculum order.
const baseProblems: Problem[] = [
  ...numpyFundamentalsProblems,
  ...einsumProblems,
  ...pytorchBasicsProblems,
  ...dataPreprocessingProblems,
  ...supervisedLearningProblems,
  ...unsupervisedLearningProblems,
  ...modelEvaluationProblems,
  ...neuralNetworkProblems,
  ...optimizationProblems,
  ...cnnProblems,
  ...sequenceModelsProblems,
  ...transformerProblems,
  ...llmInternalsProblems,
  ...representationLearningProblems,
  ...generativeModelProblems,
  ...reinforcementLearningProblems,
  ...postTrainingProblems,
  ...e2eImplementationProblems,
];

// Attach PyTorch tracks (kept in ./pytorch so they can evolve independently).
export const allProblems: Problem[] = baseProblems.map(p =>
  pytorchVariants[p.id] ? { ...p, pytorch: pytorchVariants[p.id] } : p
);

export function getProblemById(id: string): Problem | undefined {
  return allProblems.find(p => p.id === id);
}

export function getProblemsBySection(sectionId: string): Problem[] {
  return allProblems.filter(p => p.section === sectionId);
}

export {
  numpyFundamentalsProblems,
  einsumProblems,
  pytorchBasicsProblems,
  dataPreprocessingProblems,
  supervisedLearningProblems,
  unsupervisedLearningProblems,
  modelEvaluationProblems,
  neuralNetworkProblems,
  optimizationProblems,
  cnnProblems,
  sequenceModelsProblems,
  transformerProblems,
  llmInternalsProblems,
  representationLearningProblems,
  generativeModelProblems,
  reinforcementLearningProblems,
  postTrainingProblems,
  e2eImplementationProblems,
};
