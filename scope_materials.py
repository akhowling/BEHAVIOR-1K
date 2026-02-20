from pxr import Usd, Sdf, UsdShade

src = "omnigibson/data/scenes/my_scene/scene.usdc"
dst = "omnigibson/data/scenes/my_scene/scene_scoped.usda"  # ascii for easier debugging

stage = Usd.Stage.Open(src)
root = stage.GetRootLayer()

# Ensure default prim is /World
world = stage.GetPrimAtPath("/World")
if not world.IsValid():
    raise RuntimeError("No /World prim found in USD")
stage.SetDefaultPrim(world)

# Move /_materials -> /World/_materials (so it comes along inside the reference)
if stage.GetPrimAtPath("/_materials").IsValid():
    edits = Sdf.BatchNamespaceEdit()
    edits.Add(Sdf.Path("/_materials"), Sdf.Path("/World/_materials"))
    ok = root.Apply(edits)
    if not ok:
        raise RuntimeError("Failed to move /_materials under /World")
else:
    print("No /_materials prim found (skipping move)")

# Rewrite any material binding relationship targets that point at /_materials/*
rewritten = 0
for prim in stage.Traverse():
    for rel in prim.GetRelationships():
        n = rel.GetName()
        if not n.startswith("material:binding"):
            continue
        tgts = rel.GetTargets()
        if not tgts:
            continue
        new_tgts = []
        changed = False
        for t in tgts:
            if t.pathString.startswith("/_materials/"):
                new_t = Sdf.Path("/World/_materials/" + t.pathString[len("/_materials/"):])
                new_tgts.append(new_t)
                changed = True
                rewritten += 1
            else:
                new_tgts.append(t)
        if changed:
            rel.SetTargets(new_tgts)

print("Rewritten binding targets:", rewritten)

# Export to a new file
stage.Export(dst)
print("Wrote:", dst)
