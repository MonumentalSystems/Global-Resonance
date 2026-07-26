import unittest

from backend.cancer_soil_study.pipeline import _normalize_name, parse_cancer_csv


class PipelineTests(unittest.TestCase):
    def test_county_name_normalization(self):
        self.assertEqual(_normalize_name("St. Clair County"), "saintclair")
        self.assertEqual(_normalize_name("Orleans Parish"), "orleans")

    def test_cancer_export_parser(self):
        raw = b'''Incidence Rate Report for Test by County\n\n"All Cancer Sites, 2018-2022"\n\nCounty,FIPS,2023 Rural-Urban Continuum Codes([rural urban note]),"Age-Adjusted Incidence Rate([rate note]) - cases per 100,000",Lower 95% Confidence Interval,Upper 95% Confidence Interval,Average Annual Count\n"Example County(7)",19001,Rural,456.7,440.0,470.0,100\n"Suppressed County(7)",19003,Rural,*,*,*,*\n'''
        frame = parse_cancer_csv(raw, "all")
        self.assertEqual(frame.loc[0, "fips"], "19001")
        self.assertEqual(frame.loc[0, "period"], "2018-2022")
        self.assertAlmostEqual(frame.loc[0, "incidence_rate"], 456.7)
        self.assertTrue(frame.loc[1, "incidence_rate"] != frame.loc[1, "incidence_rate"])


if __name__ == "__main__":
    unittest.main()
