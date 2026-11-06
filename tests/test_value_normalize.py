"""Deterministic value normalization
(pipeline/value_normalize.py): the code side of the value
attribute judgment split.

Covered: each of the six closed value types on normal,
boundary and failure surfaces; the §11 decisions — money
never guesses a currency (incomplete, not failed) and
accepts the US$ prefix as stated USD notation, number
splits its unit and never converts, datetime stays naive
without a stated timezone and demotes a date-only surface
to date, duration is length only (a start/end pair fails),
a date keeps its stated precision (day / month / year)
instead of inventing the missing components;
relative dates resolve only against an explicit reference
date; an unparseable surface rides on a well-formed
proposed value with code_normalized False (the conversion
is then the judge's question), and fails outright without
one. Normalization never raises.
"""
from __future__ import annotations

import unittest
from datetime import date

from pipeline import value_normalize as vn


class TestDate(unittest.TestCase):

    def test_iso(self):
        r = vn.normalize_value("date", "2024-03-01")
        self.assertTrue(r["ok"])
        self.assertEqual(r["value"], "2024-03-01")
        self.assertTrue(r["code_normalized"])

    def test_month_name_forms(self):
        for surface in ("March 1, 2024", "1 March 2024",
                        "Mar. 1, 2024", "March 1st, 2024"):
            r = vn.normalize_value("date", surface)
            self.assertTrue(r["ok"], surface)
            self.assertEqual(r["value"], "2024-03-01", surface)

    def test_invalid_calendar_date_fails(self):
        r = vn.normalize_value("date", "2024-02-30")
        self.assertFalse(r["ok"])

    def test_full_date_has_day_precision(self):
        r = vn.normalize_value("date", "2024-03-01")
        self.assertEqual(r["precision"], "day")
        r = vn.normalize_value("date", "March 1, 2024")
        self.assertEqual(r["precision"], "day")

    def test_iso_month_precision(self):
        r = vn.normalize_value("date", "2008-05")
        self.assertTrue(r["ok"])
        self.assertEqual(r["value"], "2008-05")
        self.assertEqual(r["precision"], "month")

    def test_month_name_year_precision(self):
        r = vn.normalize_value("date", "May 2008")
        self.assertTrue(r["ok"])
        self.assertEqual(r["value"], "2008-05")
        self.assertEqual(r["precision"], "month")

    def test_year_only_precision(self):
        r = vn.normalize_value("date", "2008")
        self.assertTrue(r["ok"])
        self.assertEqual(r["value"], "2008")
        self.assertEqual(r["precision"], "year")

    def test_invalid_month_is_rejected(self):
        self.assertFalse(
            vn.normalize_value("date", "2024-13")["ok"])
        self.assertFalse(
            vn.normalize_value("date", "2024-00")["ok"])
        self.assertFalse(
            vn.normalize_value("date", "2024-13-01")["ok"])

    def test_partial_proposed_keeps_its_precision(self):
        r = vn.normalize_value("date", "sometime in 2008",
                               proposed="2008")
        self.assertTrue(r["ok"])
        self.assertFalse(r["code_normalized"])
        self.assertEqual(r["value"], "2008")
        self.assertEqual(r["precision"], "year")

    def test_relative_needs_reference(self):
        r = vn.normalize_value("date", "next Friday")
        self.assertFalse(r["ok"])
        r = vn.normalize_value(
            "date", "next Friday",
            reference_date=date(2024, 3, 1))  # a Friday
        self.assertTrue(r["ok"])
        self.assertEqual(r["value"], "2024-03-08")

    def test_relative_words_with_reference(self):
        ref = date(2024, 3, 1)
        self.assertEqual(
            vn.normalize_value("date", "today",
                               reference_date=ref)["value"],
            "2024-03-01")
        self.assertEqual(
            vn.normalize_value("date", "in 30 days",
                               reference_date=ref)["value"],
            "2024-03-31")

    def test_proposed_fallback_flags_conversion(self):
        r = vn.normalize_value("date", "next Friday",
                               proposed="2024-03-08")
        self.assertTrue(r["ok"])
        self.assertFalse(r["code_normalized"])
        self.assertEqual(r["value"], "2024-03-08")

    def test_proposed_malformed_still_fails(self):
        r = vn.normalize_value("date", "next Friday",
                               proposed="08/03/2024")
        self.assertFalse(r["ok"])

    def test_garbage_fails(self):
        self.assertFalse(
            vn.normalize_value("date", "whenever")["ok"])


class TestDatetime(unittest.TestCase):

    def test_iso_with_offset_kept(self):
        r = vn.normalize_value("datetime",
                               "2024-03-01T09:30:00+05:00")
        self.assertTrue(r["ok"])
        self.assertEqual(r["value"],
                         "2024-03-01T09:30:00+05:00")

    def test_zulu_canonicalized(self):
        r = vn.normalize_value("datetime", "2024-03-01T09:30:00Z")
        self.assertEqual(r["value"],
                         "2024-03-01T09:30:00+00:00")

    def test_naive_stays_naive(self):
        r = vn.normalize_value("datetime", "2024-03-01 09:30")
        self.assertTrue(r["ok"])
        self.assertEqual(r["value"], "2024-03-01T09:30:00")

    def test_month_name_with_ampm(self):
        r = vn.normalize_value("datetime",
                               "March 1, 2024 9:30 AM")
        self.assertEqual(r["value"], "2024-03-01T09:30:00")
        r = vn.normalize_value("datetime",
                               "March 1, 2024 9:30 PM")
        self.assertEqual(r["value"], "2024-03-01T21:30:00")

    def test_date_only_demotes_to_date(self):
        r = vn.normalize_value("datetime", "2024-03-01")
        self.assertTrue(r["ok"])
        self.assertEqual(r["normalized_type"], "date")
        self.assertEqual(r["value"], "2024-03-01")

    def test_bad_time_fails(self):
        self.assertFalse(vn.normalize_value(
            "datetime", "2024-03-01 25:30")["ok"])


class TestNumber(unittest.TestCase):

    def test_plain_and_grouped(self):
        self.assertEqual(
            vn.normalize_value("number", "1,250")["value"], 1250)
        self.assertEqual(
            vn.normalize_value("number", "12.5")["value"], 12.5)

    def test_unit_split_not_converted(self):
        r = vn.normalize_value("number", "100 mg")
        self.assertEqual(r["value"], 100)
        self.assertEqual(r["unit"], "mg")
        r = vn.normalize_value("number", "2.5 kg")
        self.assertEqual(r["value"], 2.5)
        self.assertEqual(r["unit"], "kg")

    def test_percent_keeps_unit(self):
        r = vn.normalize_value("number", "12%")
        self.assertEqual(r["value"], 12)
        self.assertEqual(r["unit"], "%")

    def test_scale_word_is_notation(self):
        r = vn.normalize_value("number", "1.25 million")
        self.assertEqual(r["value"], 1250000)

    def test_no_unit(self):
        r = vn.normalize_value("number", "42")
        self.assertIsNone(r["unit"])

    def test_words_fail(self):
        self.assertFalse(
            vn.normalize_value("number", "several")["ok"])


class TestMoney(unittest.TestCase):

    def test_symbol_maps(self):
        r = vn.normalize_value("money", "$500,000")
        self.assertTrue(r["ok"])
        self.assertEqual(r["value"], 500000)
        self.assertEqual(r["currency"], "USD")
        self.assertTrue(r["complete"])
        self.assertEqual(
            vn.normalize_value("money", "€250")["currency"],
            "EUR")

    def test_code_and_name_map(self):
        self.assertEqual(
            vn.normalize_value("money", "USD 500")["currency"],
            "USD")
        self.assertEqual(
            vn.normalize_value(
                "money", "500 euros")["currency"], "EUR")
        self.assertEqual(
            vn.normalize_value(
                "money", "500 pounds sterling")["currency"],
            "GBP")

    def test_no_currency_is_incomplete_not_failed(self):
        r = vn.normalize_value("money", "500,000")
        self.assertTrue(r["ok"])
        self.assertIsNone(r["currency"])
        self.assertFalse(r["complete"])

    def test_scale_word_amount(self):
        r = vn.normalize_value("money", "$1.25 million")
        self.assertEqual(r["value"], 1250000)
        self.assertEqual(r["currency"], "USD")

    def test_us_dollar_prefix(self):
        r = vn.normalize_value("money", "US$400,000.00")
        self.assertTrue(r["ok"])
        self.assertEqual(r["value"], 400000)
        self.assertEqual(r["currency"], "USD")
        self.assertTrue(r["complete"])

    def test_unit_bearing_surface_is_not_money(self):
        self.assertFalse(
            vn.normalize_value("money", "100 mg")["ok"])

    def test_proposed_with_unknown_currency_fails(self):
        r = vn.normalize_value(
            "money", "a large sum",
            proposed={"value": 10, "currency": "XXX"})
        self.assertFalse(r["ok"])


class TestDuration(unittest.TestCase):

    def test_iso_passthrough(self):
        r = vn.normalize_value("duration", "P30D")
        self.assertTrue(r["ok"])
        self.assertEqual(r["value"], "P30D")

    def test_natural_single_and_multi_unit(self):
        self.assertEqual(
            vn.normalize_value("duration", "30 days")["value"],
            "P30D")
        self.assertEqual(
            vn.normalize_value("duration", "2 years")["value"],
            "P2Y")
        self.assertEqual(
            vn.normalize_value(
                "duration", "1 year 6 months")["value"],
            "P1Y6M")
        self.assertEqual(
            vn.normalize_value("duration", "48 hours")["value"],
            "PT48H")

    def test_interval_pair_fails(self):
        r = vn.normalize_value(
            "duration", "from March 1, 2024 to June 1, 2025")
        self.assertFalse(r["ok"])

    def test_empty_iso_fails(self):
        self.assertFalse(
            vn.normalize_value("duration", "P")["ok"])


class TestIdentifier(unittest.TestCase):

    def test_whitespace_unified(self):
        r = vn.normalize_value("identifier", "  ABC   123-45 ")
        self.assertTrue(r["ok"])
        self.assertEqual(r["value"], "ABC 123-45")

    def test_empty_fails(self):
        self.assertFalse(
            vn.normalize_value("identifier", "   ")["ok"])


class TestContract(unittest.TestCase):

    def test_unknown_type_fails_closed(self):
        r = vn.normalize_value("freetext", "anything")
        self.assertFalse(r["ok"])

    def test_non_string_surface_fails(self):
        self.assertFalse(vn.normalize_value("date", 20240301)
                         ["ok"])
        self.assertFalse(vn.normalize_value("date", "")["ok"])

    def test_never_raises(self):
        for value_type in vn.VALUE_TYPES:
            for surface in ("", " ", "!!!", "2024-13-45",
                            "P1Y2M3DT4H5M6S", "1,2,3"):
                r = vn.normalize_value(value_type, surface)
                self.assertIn("ok", r)


if __name__ == "__main__":
    unittest.main()
