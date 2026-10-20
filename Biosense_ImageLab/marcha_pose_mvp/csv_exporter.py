"""CSV serialization without reducing the precision of internal calculations."""
import csv
import math
from numbers import Integral, Real

class CsvExporter:
    @staticmethod
    def format_value(value):
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, Real):
            if not math.isfinite(value):
                return ""
            rounded = round(float(value), 1)
            return f"{0.0 if rounded == 0 else rounded:.1f}"
        return value

    def export(self, path, fields, rows):
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows({key: self.format_value(value) for key, value in row.items()} for row in rows)
        return path
