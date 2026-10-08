"""
Tests for Skill Taxonomy and Skill Normalization.
"""

import unittest
from modules.nlp.skill_taxonomy import SkillTaxonomy


class TestSkillTaxonomy(unittest.TestCase):

    def test_skill_normalization_synonyms(self):
        self.assertEqual(SkillTaxonomy.normalize("ml"), "Machine Learning")
        self.assertEqual(SkillTaxonomy.normalize("ai"), "Artificial Intelligence")
        self.assertEqual(SkillTaxonomy.normalize("nlp"), "Natural Language Processing")
        self.assertEqual(SkillTaxonomy.normalize("py torch"), "PyTorch")
        self.assertEqual(SkillTaxonomy.normalize("scikit learn"), "Scikit-learn")
        self.assertEqual(SkillTaxonomy.normalize("sklearn"), "Scikit-learn")
        self.assertEqual(SkillTaxonomy.normalize("k8s"), "Kubernetes")
        self.assertEqual(SkillTaxonomy.normalize("postgres"), "PostgreSQL")

    def test_skill_categories(self):
        self.assertEqual(SkillTaxonomy.get_category("Python"), "Programming Languages")
        self.assertEqual(SkillTaxonomy.get_category("Machine Learning"), "Machine Learning & AI")
        self.assertEqual(SkillTaxonomy.get_category("Docker"), "Cloud & DevOps")
        self.assertEqual(SkillTaxonomy.get_category("PostgreSQL"), "Databases & Storage")


if __name__ == "__main__":
    unittest.main()
