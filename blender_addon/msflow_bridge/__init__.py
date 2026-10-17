"""MSFlow Motion Bridge — bring MSFlow-generated motions into Blender.

The webUI (or the CLI) exports one `sample_XX.msflow.json` per generated
sample next to its npy/mp4 outputs. This add-on imports those files (file
browser, 'latest run', folder watch, or the webUI's 'Send to Blender' push
receiver), retargets them onto any armature via a bone mapping, keys a new
Action (optionally as a muted NLA clip), fits the scene duration, and can
draw the sparse constraint targets as empties.
"""
bl_info = {
    "name": "MSFlow Motion Bridge",
    "description": "Import MSFlow text/joint-controlled motions as retargeted "
    "Actions on your armature",
    "author": "MSFlow constraint webui project",
    "version": (1, 0, 0),
    "blender": (4, 2, 0),
    "category": "Animation",
    "doc_url": "https://github.com/ly-corporation/MSFlow",
}

try:
    import bpy
except ImportError:  # importing pure submodules from a plain interpreter (tests)
    bpy = None

if "io_msflow" in locals():  # reload-friendly iterative development
    import importlib

    importlib.reload(io_msflow)
    importlib.reload(mapping)
else:
    from . import io_msflow, mapping

if bpy is not None:
    if "retarget" in globals():
        importlib.reload(retarget)
        importlib.reload(server)
        importlib.reload(import_ops)
        importlib.reload(panels)
    else:
        from . import import_ops, panels, retarget, server

    from bpy.props import PointerProperty

    def register() -> None:
        for cls in import_ops.CLASSES + panels.CLASSES:
            try:
                bpy.utils.register_class(cls)
            except RuntimeError:  # already registered (reload)
                pass
        import_ops.register_scene_props()

    def unregister() -> None:
        import_ops.shutdown_timers()
        import_ops.unregister_scene_props()
        for cls in reversed(panels.CLASSES + import_ops.CLASSES):
            try:
                bpy.utils.unregister_class(cls)
            except RuntimeError:
                pass


if __name__ == "__main__":
    if bpy is not None:
        register()
