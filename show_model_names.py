import json
from imageworks.model_loader.simplified_naming import simplified_display_for_entry

with open("configs/model_registry.json") as f:
    registry = json.load(f)

print("| Underlying Model | Registry Display Name | Simplified Display Name |")
print("|------------------|----------------------|-------------------------|")
for entry in registry:
    orig = entry.get("backend_config", {}).get("model_path")
    disp = entry.get("display_name")

    class E:
        pass

    e = E()
    for k, v in entry.items():
        setattr(e, k, v)
    simp = simplified_display_for_entry(e)
    print(f"| {orig} | {disp} | {simp} |")
