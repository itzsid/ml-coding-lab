/**
 * Old section ids that were merged into other sections. Problem URLs whose
 * section no longer matches the problem are corrected automatically on the
 * problem page; this map handles section pages.
 */
export const SECTION_ALIASES: Record<string, string> = {
  'python-basics': 'numpy-fundamentals',
  'deep-learning': 'neural-networks',
};
