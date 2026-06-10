#!/usr/bin/env python3
"""Selectively extract FaceScape TU-model source assets from the trainset zip.

Reference facts (not code decisions -- just what's true about the data):

  The zip (facescape_trainset_801_847.zip, ~7.4 GB) holds, per subject id
  (801-847) and per expression:
      <id>/models_reg/<exp>.obj       geometry: verts / faces / uv
      <id>/models_reg/<exp>.jpg       4K texture
      <id>/models_reg/<exp>.obj.mtl   material (ships Kd=0.0 -> renders black)
      <id>/dpmap/<exp>.png            displacement map (optional)
  No RGB-D, no params.json, no photos. The .obj names its .mtl and the .mtl
  names its .jpg by bare filename -- keep those filenames if you want the links
  to stay valid.
"""




# Outline -- what this tool needs to do. You decide the functions, args, layout.
#
#   1. Take inputs: which zip, which subject id(s), which expression(s), where to
#      write, whether to include the dpmap.
#   2. For a given (id, exp), figure out the archive paths to pull (see the
#      layout above -- mind the .obj.mtl suffix).
#   3. Open the zip and extract just those members (not the whole archive),
#      preserving their relative paths so the mesh/texture links survive.
#   4. Skip / warn on anything missing instead of crashing the batch.
#
# Build it your way from here.

import zipfile, argparse
from pathlib import Path

class Extract:
    # The 20 canonical FaceScape expression stems, in order, exactly as named in the
    # zip. Reference data -- here because it's tedious to retype, not a design choice.
    EXPRESSIONS = [
        "1_neutral", "2_smile", "3_mouth_stretch", "4_anger", "5_jaw_left",
        "6_jaw_right", "7_jaw_forward", "8_mouth_left", "9_mouth_right", "10_dimpler",
        "11_chin_raiser", "12_lip_puckerer", "13_lip_funneler", "14_sadness",
        "15_lip_roll", "16_grin", "17_cheek_blowing", "18_eye_closed",
        "19_brow_raiser", "20_brow_lower",
    ]

    def __init__(self, zip_path, out_root="data/facescape"):
        self.zip_path = Path(zip_path)
        self.out_root = Path(out_root)

    def discover_id(self):
        """
        Return the sorted, de-duplicated subject ids found in the zip,
        """
        with zipfile.ZipFile(self.zip_path) as zf:
            names = zf.namelist()        # every entry, e.g. "801/models_reg/1_neutral.obj"
        return sorted({name.split("/")[0] for name in names})

    def file_paths(self, id, exp):
        """
        Return the list of archive paths to pull for one (id, exp):
        the .obj, the .jpg, and the .obj.mtl, all under <id>/models_reg/.
        """
        obj = f"{id}/models_reg/{exp}.obj"
        jpg = f"{id}/models_reg/{exp}.jpg"
        mtl = f"{id}/models_reg/{exp}.obj.mtl"
        return [obj, jpg, mtl]


    def extract(self, id, exp):
        """
        Extract the members for a single (id, exp) into
        out_root/<id_range>/<id>/
        """
        with zipfile.ZipFile(self.zip_path) as zf:
            for f in self.file_paths(id, exp):
                name = Path(f).name
                out = self.out_root / self.id_range / id / name
                try:
                    data = zf.read(f)
                except KeyError:
                    print(f"  warning: missing in zip, skipped: {f}")
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data)

    def run(self, id=None, exp="1_neutral"):
        ids = self.discover_id()
        
        self.id_range = f"{ids[0]}_{ids[-1]}"
        
        targets = [id] if id else ids
        
        for t in targets:
            self.extract(t, exp)
            
        


if __name__ == "__main__":
    # 1. make a parser
    parser = argparse.ArgumentParser(
        description="Extract FaceScape TU-model assets (.obj/.jpg/.mtl) from the trainset zip."
    )
    # 2. declare each argument. One worked example -- the positional zip path:
    parser.add_argument("zip_path", help="path to the trainset zip")
    parser.add_argument("--id", default=None)
    parser.add_argument("--exp", default="1_neutral", choices=Extract.EXPRESSIONS)
    
    # 3. read sys.argv into an object whose attributes are the args above
    args = parser.parse_args()
    # 4. use them: build the tool and run it
    Extract(args.zip_path).run(id=args.id, exp=args.exp)