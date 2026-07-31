import unittest

from src.utils.material_condition_diagnostics import (
    MaterialRecord,
    build_parameter_derangement,
    rotate_material_type,
)


class MaterialConditionDiagnosticsTest(unittest.TestCase):
    def test_derangement_is_within_material_one_to_one_and_reproducible(self):
        records = [
            MaterialRecord("e0.h5", 0, 4.0, 0.1),
            MaterialRecord("e1.h5", 0, 5.0, 0.2),
            MaterialRecord("e2.h5", 0, 6.0, 0.3),
            MaterialRecord("s0.h5", 2, 4.5, 0.15),
            MaterialRecord("s1.h5", 2, 6.5, 0.35),
        ]
        first = build_parameter_derangement(records, seed=7)
        second = build_parameter_derangement(records, seed=7)
        self.assertEqual(first, second)
        for group in ({"e0.h5", "e1.h5", "e2.h5"}, {"s0.h5", "s1.h5"}):
            source = {(r.log10_e, r.nu) for r in records if r.model in group}
            assigned = {first[name] for name in group}
            self.assertEqual(source, assigned)
            for record in (r for r in records if r.model in group):
                self.assertNotEqual(first[record.model], (record.log10_e, record.nu))

    def test_rotates_supported_material_classes(self):
        self.assertEqual([rotate_material_type(i) for i in (0, 1, 2)], [1, 2, 0])

    def test_rejects_unsupported_material_class(self):
        with self.assertRaisesRegex(ValueError, "expected one of"):
            rotate_material_type(3)

    def test_rejects_unsupported_material_class_in_records(self):
        records = [
            MaterialRecord("e0.h5", 0, 4.0, 0.1),
            MaterialRecord("x0.h5", 3, 5.0, 0.2),
        ]
        with self.assertRaisesRegex(ValueError, "expected one of"):
            build_parameter_derangement(records, seed=7)

    def test_rejects_singleton_material_group(self):
        records = [MaterialRecord("e0.h5", 0, 4.0, 0.1)]
        with self.assertRaisesRegex(ValueError, "at least two records"):
            build_parameter_derangement(records, seed=7)


if __name__ == "__main__":
    unittest.main()
