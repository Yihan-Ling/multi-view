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
        with zipfile.ZipFile(self.zip_path) as zf:
            names = zf.namelist()        # every entry, e.g. "801/models_reg/1_neutral.obj"
        return sorted({name.split("/")[0] for name in names})

    def file_paths(self, id, exp):
        obj = f"{id}/models_reg/{exp}.obj"
        jpg = f"{id}/models_reg/{exp}.jpg"
        mtl = f"{id}/models_reg/{exp}.obj.mtl"
        return [obj, jpg, mtl]


    def extract(self, id, exp):
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
    parser = argparse.ArgumentParser(
        description="Extract FaceScape TU-model assets (.obj/.jpg/.mtl) from the trainset zip."
    )
    parser.add_argument("zip_path", help="path to the trainset zip")
    parser.add_argument("--id", default=None)
    parser.add_argument("--exp", default="1_neutral", choices=Extract.EXPRESSIONS)
    
    args = parser.parse_args()

    Extract(args.zip_path).run(id=args.id, exp=args.exp)