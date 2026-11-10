========================================================================
  Jampot  —  Prison Pack
  A matched set of stylised, hand-painted-look 3D environment props.
========================================================================

Thank you for picking up the pack! Everything you need is in this folder.

------------------------------------------------------------------------
WHAT'S INSIDE
------------------------------------------------------------------------
One cohesive Prison building kit — every prop shares the same palette and
finish so they drop into a scene together. Included piece families:

  Door · Window · Wall (straight / corner / floor) · Fence (straight /
  corner / gate) · Terrain (flat / slope / cliff / corner) · Furniture ·
  Bench · Light Pole · Barrels (closed / open / stack) · Crates (closed /
  open / stack)

Every piece is a separate model with a clean, upright, origin-on-the-floor
transform and a flat-colour stylised PBR material (base colour + roughness;
no texture files to wrangle — the look is in the material).

------------------------------------------------------------------------
FOLDERS
------------------------------------------------------------------------
  production-ready/
    FBX/   Every piece as .fbx (Y-up) + Prison_FullBuild.fbx (the whole kit).
           Best for Unity and Maya/Max pipelines.
    GLB/   Every piece as .glb (Y-up, materials embedded) + Prison_FullBuild.glb.
           Drop-in for Godot, Three.js, Babylon, Unreal (glTF), Sketchfab.
    Prison_FullBuild.blend
           The whole kit in one Blender file, fully self-contained.
    README.txt / LICENSE.txt
  shots/
    hero_a/b/c.png   The three marketing angles of the assembled scene.
    items/           One clean studio render of every individual piece.

------------------------------------------------------------------------
TECH SPECS
------------------------------------------------------------------------
  Style               Prison  (pack 05 of the Jampot series)
  Pieces              21 individual props
  Scale               Real-world — 1 Blender unit = 1 metre.
  Orientation         GLB & FBX: Y-up   ·   .blend: Z-up (Blender native)
  Origins             Every piece centred at the origin, base on the floor (Z=0),
                      identity transform — a true drop-in (no re-scaling or
                      rotating needed in-engine).
  Materials           One flat-colour Principled/PBR material per surface.
                      Metallic 0 except where a piece is genuinely metal;
                      the stylised look lives in the base colour + roughness.

------------------------------------------------------------------------
ENGINE NOTES
------------------------------------------------------------------------
  UNITY   Drag in the FBX (or GLB) — imports upright at scale 1 with no
          rotation. Material Standard/URP-Lit; plug Base Color + Roughness.
  UNREAL  Import the GLB (or FBX) — real-world scale, drop-in materials.
  GODOT   Drag the .glb in — materials come through automatically.
  WEB     GLB is self-contained — ideal for Three.js / Babylon / Sketchfab.

------------------------------------------------------------------------
LICENSE
------------------------------------------------------------------------
  Royalty-free commercial. Use in unlimited personal and commercial
  projects; no per-title fees or attribution required. You may NOT resell
  or redistribute the raw asset files themselves. Full terms in LICENSE.txt.

------------------------------------------------------------------------
CREDITS / CONTACT
------------------------------------------------------------------------
  Jampot.  Built procedurally in Blender.
  Questions or a bug in a mesh? Reach out on the itch.io product page.

  Enjoy — and go build a world.
========================================================================
