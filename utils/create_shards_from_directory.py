import io
from pathlib import Path
import tarfile

from tqdm import tqdm


def shard_json_directory(input_dir: str | Path, output_dir: str | Path, samples_per_shard: int = 2_500, suffix: str = "json"):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob(f"*.{suffix}"))
    if len(json_files) == 0:
        print(f"No JSON files found in {input_dir}. Use `preprocess_mimi.py` to create JSON files from audio/text pairs.")
        return

    print(f"Found {len(json_files)} JSON files in {input_dir}")
    shard_idx = 0
    sample_idx = 0
    tar = None

    for i, file_path in tqdm(enumerate(json_files), total=len(json_files), desc="Sharding files"):
        if sample_idx % samples_per_shard == 0:
            if tar is not None:
                tar.close()
            shard_name = f"shard-{shard_idx:06d}.tar"
            tar = tarfile.open(output_dir / shard_name, "w")
            print(f"Creating shard {shard_idx} at {output_dir / shard_name}")
            shard_idx += 1

        with open(file_path, "rb") as f:
            data = f.read()
            key = file_path.stem
            tarinfo = tarfile.TarInfo(name=f"{key}.{suffix}")
            tarinfo.size = len(data)
            assert tar
            tar.addfile(tarinfo, io.BytesIO(data))

        sample_idx += 1

    if tar is not None:
        tar.close()

    print(f"Done. Created {shard_idx} shard(s).")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("directory", help="")
    parser.add_argument("output", help="")

    args = parser.parse_args()
    shard_json_directory(Path(args.directory).expanduser(), Path(args.output).expanduser())
