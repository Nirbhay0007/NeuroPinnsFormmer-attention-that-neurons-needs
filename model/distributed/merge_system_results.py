import json
import os
import subprocess

reports_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "detailed_pinnsformmer_reports"))

merged_dict = {}

for sys_name in ["A", "B", "C"]:
    file_path = os.path.join(reports_dir, f"system_{sys_name}_results.json")
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            merged_dict.update(data)
            print(f"[LOADED] System {sys_name} results ({len(data)} configurations)")

merged_path = os.path.join(reports_dir, "benchmark_results.json")
with open(merged_path, "w", encoding="utf-8") as f:
    json.dump(merged_dict, f, indent=4)

print(f"\n[SUCCESS] Merged all 3 system results into: {merged_path}")
print("Generating publication paper plots...")

julia_plot_cmd = ["julia", os.path.join(os.path.dirname(__file__), "..", "paper_plots.jl")]
subprocess.run(julia_plot_cmd)
