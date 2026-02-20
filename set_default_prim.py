from pxr import Usd

usd_path = "omnigibson/data/scenes/my_scene/scene.usdc"
stage = Usd.Stage.Open(usd_path)

top = [p for p in stage.GetPseudoRoot().GetChildren() if p.IsValid()]
if not top:
    raise RuntimeError("No top-level prims found.")

# If there’s a prim named World/Root/Scene, prefer it
preferred = {"world","root","scene","environment"}
chosen = None
for p in top:
    if p.GetName().lower() in preferred:
        chosen = p
        break
if chosen is None:
    chosen = top[0]

stage.SetDefaultPrim(chosen)
stage.GetRootLayer().Save()
print("Default prim set to:", chosen.GetPath())
