"""CPU-only schedule regression test; does not initialize the training config."""
import ast
from pathlib import Path
source = Path(__file__).resolve().parents[1] / "lib/utils/static_prune.py"
tree = ast.parse(source.read_text())
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "scheduled_prune_iterations")
scope = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), scope)
schedule = scope["scheduled_prune_iterations"]
class Options(dict):
    def __getattr__(self, name):
        return self[name]
base = Options(static_prune_enabled=True, static_prune_interval=10000,
               static_prune_until_iter=45000, static_prune_cdf_threshold=.99,
               densify_until_iter=50000)
assert schedule(base, 100000) == {10000,20000,30000,40000,45000}
assert schedule(Options(base, static_prune_enabled=False), 100000) == set()
assert schedule(Options(base, static_prune_until_iter=-1), 100000) == {10000,20000,30000,40000,50000}
assert schedule(Options(base, static_prune_interval=5000, static_prune_until_iter=-1, densify_until_iter=25000),50000) == {5000,10000,15000,20000,25000}
assert schedule(Options(base, static_prune_until_iter=0),100000) == set()
for bad in [True, -2, 60000]:
    try:
        schedule(Options(base, static_prune_until_iter=bad), 100000)
    except ValueError:
        pass
    else:
        raise AssertionError(bad)
print("Schedule regression checks passed.")
