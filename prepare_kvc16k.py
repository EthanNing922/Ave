import argparse
import json
import shutil
from pathlib import Path

import torch


def read_id_file(path):
    pairs = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                parts = line.rsplit(" ", 1)
            if len(parts) != 2:
                raise ValueError(f"bad id line {path}:{line_number}: {line!r}")
            name, index = parts
            pairs.append((name, int(index)))
    pairs.sort(key=lambda item: item[1])
    expected = list(range(len(pairs)))
    actual = [index for _, index in pairs]
    if actual != expected:
        raise ValueError(f"{path} ids must be contiguous from 0")
    return pairs


def convert_split(source, target, entities, relations):
    count = 0
    with source.open("r", encoding="utf-8") as reader, target.open(
        "w", encoding="utf-8", newline="\n"
    ) as writer:
        for line_number, line in enumerate(reader, 1):
            parts = line.strip().split()
            if parts and parts[-1] == ".":
                parts = parts[:-1]
            if len(parts) != 3:
                raise ValueError(
                    f"bad triple line {source}:{line_number}: {line.rstrip()!r}"
                )
            head, relation, tail = parts
            if head not in entities:
                raise ValueError(f"unknown head {head!r} at {source}:{line_number}")
            if relation not in relations:
                raise ValueError(
                    f"unknown relation {relation!r} at {source}:{line_number}"
                )
            if tail not in entities:
                raise ValueError(f"unknown tail {tail!r} at {source}:{line_number}")
            writer.write(f"{head}\t{relation}\t{tail}\n")
            count += 1
    return count


def write_lines(path, values):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(f"{value}\n")


def write_id_file(path, pairs):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for name, index in pairs:
            handle.write(f"{name} {index}\n")


def write_feature_tokens(path, pairs):
    tokenized = {name: [index + 1] for name, index in pairs}
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(tokenized, handle, ensure_ascii=False)


def prepare(args):
    source = Path(args.source_dir)
    data_dir = Path(args.data_dir)
    token_dir = Path(args.token_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    token_dir.mkdir(parents=True, exist_ok=True)

    entity_pairs = read_id_file(source / "entity2id.txt")
    relation_pairs = read_id_file(source / "relation2id.txt")
    entity_names = [name for name, _ in entity_pairs]
    relation_names = [name for name, _ in relation_pairs]
    entities = set(entity_names)
    relations = set(relation_names)

    write_lines(data_dir / "entities.txt", entity_names)
    write_lines(data_dir / "relations.txt", relation_names)
    write_id_file(data_dir / "entity2id.txt", entity_pairs)
    write_id_file(data_dir / "relation2id.txt", relation_pairs)

    split_counts = {}
    for split in ("train", "valid", "test"):
        split_counts[split] = convert_split(
            source / f"{split}.txt",
            data_dir / f"{split}.txt",
            entities,
            relations,
        )

    image_features = torch.load(source / "img_features.pth", map_location="cpu")
    text_features = torch.load(source / "text_features.pth", map_location="cpu")
    expected_shape = (len(entity_pairs), image_features.shape[1])
    if tuple(image_features.shape) != expected_shape:
        raise ValueError(
            f"img_features shape must be {expected_shape}, got {tuple(image_features.shape)}"
        )
    expected_shape = (len(entity_pairs), text_features.shape[1])
    if tuple(text_features.shape) != expected_shape:
        raise ValueError(
            f"text_features shape must be {expected_shape}, got {tuple(text_features.shape)}"
        )

    image_padding = torch.zeros(1, image_features.shape[1])
    text_padding = torch.zeros(1, text_features.shape[1])
    torch.save(
        torch.cat([image_padding, image_features.float()], dim=0),
        token_dir / "KVC16K-visual-feature.pth",
    )
    torch.save(
        torch.cat([text_padding, text_features.float()], dim=0),
        token_dir / "KVC16K-textual-feature.pth",
    )
    write_feature_tokens(token_dir / "KVC16K-visual-feature.json", entity_pairs)
    write_feature_tokens(token_dir / "KVC16K-textual-feature.json", entity_pairs)

    if args.keep_source_copy:
        for name in ("img_features.pth", "text_features.pth"):
            target = data_dir / name
            if not target.exists():
                shutil.copy2(source / name, target)

    return {
        "entities": len(entity_pairs),
        "relations": len(relation_pairs),
        "splits": split_counts,
        "image_feature_shape": tuple(image_features.shape),
        "text_feature_shape": tuple(text_features.shape),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MoMoK KVC16K continuous features for Ave."
    )
    parser.add_argument("--source_dir", default="KVC16K")
    parser.add_argument("--data_dir", default="data/KVC16K")
    parser.add_argument("--token_dir", default="tokens")
    parser.add_argument("--keep_source_copy", action="store_true")
    args = parser.parse_args()
    summary = prepare(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
