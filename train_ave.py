import argparse
import json
import logging
import math
import os
import random
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from dataset import VTKG
from model_ave import (
    Ave,
    build_relation_role_prior,
    build_train_filter,
    role_prior_statistics,
)
from utils import metrics


LOGGER = logging.getLogger("ave")


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def parse_grid(value, include_one=False):
    values = sorted(
        {float(item.strip()) for item in value.split(",") if item.strip()}
    )
    if not values:
        raise ValueError("calibration grid cannot be empty")
    if any(not math.isfinite(item) or item < 0 for item in values):
        raise ValueError("calibration values must be finite and non-negative")
    required = [0.0, 1.0] if include_one else [0.0]
    return sorted(set(values + required))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Ave pipeline: FGC training, relation-role "
            "calibration, two-hop path calibration, and one final test"
        )
    )
    parser.add_argument("--data", default="MKG-W")
    parser.add_argument("--exp", default="ave")
    parser.add_argument(
        "--base_ckpt",
        default="",
        help="skip training and run calibration from a compatible checkpoint",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--non_deterministic", action="store_true")
    parser.add_argument("--no_write", action="store_true")
    parser.add_argument("--dim", default=256, type=int)
    parser.add_argument("--hidden_dim", default=1024, type=int)
    parser.add_argument("--num_head", default=4, type=int)
    parser.add_argument("--num_layer_enc_ent", default=1, type=int)
    parser.add_argument("--num_layer_enc_rel", default=1, type=int)
    parser.add_argument("--num_layer_dec", default=2, type=int)
    parser.add_argument("--dropout", default=0.01, type=float)
    parser.add_argument("--emb_dropout", default=0.9, type=float)
    parser.add_argument("--vis_dropout", default=0.4, type=float)
    parser.add_argument("--txt_dropout", default=0.1, type=float)
    parser.add_argument("--max_vis_token", default=8, type=int)
    parser.add_argument("--max_txt_token", default=8, type=int)
    parser.add_argument(
        "--text_tokenizer",
        default="bert",
        choices=["bert", "roberta", "llama", "feature"],
    )
    parser.add_argument(
        "--visual_tokenizer", default="beit", choices=["beit", "vqgan", "feature"]
    )
    parser.add_argument("--num_epoch", default=1500, type=int)
    parser.add_argument("--valid_epoch", default=50, type=int)
    parser.add_argument("--early_stop", default=0, type=int)
    parser.add_argument("--min_delta", default=0.0, type=float)
    parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--eval_batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--decay", default=0.0, type=float)
    parser.add_argument("--step_size", default=50, type=int)
    parser.add_argument("--grad_clip", default=0.1, type=float)
    parser.add_argument("--mu", default=0.001, type=float)
    parser.add_argument("--similar_roles", default=4, type=int)
    parser.add_argument("--role_direct_weight", default=0.5, type=float)
    parser.add_argument("--max_role_strength", default=2.0, type=float)
    parser.add_argument("--lambda_role_ce", default=1.0, type=float)
    parser.add_argument("--lambda_role_reg", default=1e-4, type=float)
    parser.add_argument("--role_grad_clip", default=1.0, type=float)
    parser.add_argument(
        "--role_scales", default="0,0.25,0.5,0.75,1,1.5"
    )
    parser.add_argument("--min_rule_support", default=2, type=int)
    parser.add_argument(
        "--path_alphas", default="0,0.25,0.5,1,2,4,8"
    )
    return parser.parse_args(argv)


def configure_logging(args, run_name):
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)
    if not args.no_write:
        directory = Path("logs") / args.exp / args.data
        directory.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(
            directory / f"{run_name}.log", encoding="utf-8"
        )
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def _read_json_or_zip(path):
    path = Path(path)
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    zip_path = Path(f"{path}.zip")
    if not zip_path.is_file():
        raise FileNotFoundError(f"missing token file: {path} or {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        member = next(
            name
            for name in archive.namelist()
            if not name.startswith("__MACOSX/") and name.endswith(".json")
        )
        with archive.open(member) as handle:
            return json.loads(handle.read().decode("utf-8"))


def _entity_map(dataset):
    data_dir = Path("data") / dataset
    mapping = {}
    if dataset in {"DB15K", "FB15K-237", "WN9"}:
        with (data_dir / "entities.txt").open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                entity = line.strip()
                mapping[entity] = index if dataset != "DB15K" else entity
    else:
        with (data_dir / "entity2id.txt").open("r", encoding="utf-8") as handle:
            for line in handle:
                entity, index = line.rstrip("\n").rsplit(" ", 1)
                mapping[entity] = int(index)
    return mapping


def _select_tokens(tokenized, entity_map, max_num):
    entity_tokens = [[] for _ in range(len(entity_map))]
    for entity, values in tokenized.items():
        if entity not in entity_map:
            continue
        entity_id = entity_map[entity]
        if isinstance(entity_id, str):
            entity_id = int(entity_id)
        entity_tokens[entity_id] = [
            int(token) for token, _ in Counter(values).most_common(max_num)
        ]
    ids, masks = [], []
    for values in entity_tokens:
        values = values[:max_num]
        padding = max_num - len(values)
        ids.append(values + [0] * padding)
        masks.append([False] * len(values) + [True] * padding)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(masks, dtype=torch.bool)


def get_visual_tokens(dataset, max_num, tokenizer):
    if tokenizer == "feature":
        suffix = "visual-feature"
    else:
        suffix = "visual" if tokenizer == "beit" else "visual-vqgan"
    tokenized = _read_json_or_zip(Path("tokens") / f"{dataset}-{suffix}.json")
    return _select_tokens(tokenized, _entity_map(dataset), max_num)


def get_text_tokens(dataset, max_num, tokenizer):
    if tokenizer == "feature":
        suffix = "textual-feature"
    elif dataset == "DB15K" and tokenizer == "bert":
        suffix = "textual-v2"
    elif tokenizer == "bert":
        suffix = "textual"
    else:
        suffix = f"textual-{tokenizer}"
    tokenized = _read_json_or_zip(Path("tokens") / f"{dataset}-{suffix}.json")
    return _select_tokens(tokenized, _entity_map(dataset), max_num)


def build_model(args, kg, role_prior, device):
    visual_ids, visual_mask = get_visual_tokens(
        args.data, args.max_vis_token, args.visual_tokenizer
    )
    text_ids, text_mask = get_text_tokens(
        args.data, args.max_txt_token, args.text_tokenizer
    )
    visual_ids = visual_ids.to(device)
    visual_mask = visual_mask.to(device)
    text_ids = text_ids.to(device)
    text_mask = text_mask.to(device)
    LOGGER.info(
        "tokens visual=%s text=%s missing visual=%d text=%d",
        tuple(visual_ids.shape),
        tuple(text_ids.shape),
        int(visual_mask.all(dim=1).sum()),
        int(text_mask.all(dim=1).sum()),
    )
    visual_embedding_path = None
    if args.visual_tokenizer == "feature":
        visual_embedding_path = str(
            Path("tokens") / f"{args.data}-visual-feature.pth"
        )
    text_embedding_path = None
    if args.text_tokenizer == "feature":
        text_embedding_path = str(
            Path("tokens") / f"{args.data}-textual-feature.pth"
        )
    return Ave(
        num_ent=kg.num_ent,
        num_rel=kg.num_rel,
        role_prior=role_prior,
        max_role_strength=args.max_role_strength,
        ent_vis_mask=visual_mask,
        ent_txt_mask=text_mask,
        dim_str=args.dim,
        num_head=args.num_head,
        dim_hid=args.hidden_dim,
        num_layer_enc_ent=args.num_layer_enc_ent,
        num_layer_enc_rel=args.num_layer_enc_rel,
        num_layer_dec=args.num_layer_dec,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        vis_dropout=args.vis_dropout,
        txt_dropout=args.txt_dropout,
        visual_token_index=visual_ids,
        text_token_index=text_ids,
        text_tokenizer=args.text_tokenizer,
        visual_tokenizer=args.visual_tokenizer,
        text_token_embedding_path=text_embedding_path,
        visual_token_embedding_path=visual_embedding_path,
    ).to(device)


def build_relation_types(kg):
    relation_pairs = defaultdict(list)
    for head, relation, tail in kg.train:
        relation_pairs[relation].append((head, tail))
    relation_types = {}
    for relation, pairs in relation_pairs.items():
        tails_by_head = defaultdict(set)
        heads_by_tail = defaultdict(set)
        for head, tail in set(pairs):
            tails_by_head[head].add(tail)
            heads_by_tail[tail].add(head)
        tph = sum(map(len, tails_by_head.values())) / len(tails_by_head)
        hpt = sum(map(len, heads_by_tail.values())) / len(heads_by_tail)
        if tph < 1.5 and hpt < 1.5:
            relation_types[relation] = "1-1"
        elif tph >= 1.5 and hpt < 1.5:
            relation_types[relation] = "1-N"
        elif tph < 1.5 and hpt >= 1.5:
            relation_types[relation] = "N-1"
        else:
            relation_types[relation] = "N-N"
    return relation_types


def encoded_queries(kg, split, device):
    mask_id = kg.num_ent + kg.num_rel
    queries, labels, keys, directions = [], [], [], []
    for head, relation, tail in getattr(kg, split):
        queries.append([mask_id, relation + kg.num_ent, tail + kg.num_rel])
        labels.append(head)
        keys.append((-1, relation, tail))
        directions.append("head")
        queries.append([head + kg.num_rel, relation + kg.num_ent, mask_id])
        labels.append(tail)
        keys.append((head, relation, -1))
        directions.append("tail")
    return (
        torch.tensor(queries, dtype=torch.long, device=device),
        torch.tensor(labels, dtype=torch.long, device=device),
        keys,
        directions,
    )


def filtered_ranks(scores, labels, keys, kg):
    filtered = scores.clone()
    rows, entities = [], []
    for row, key in enumerate(keys):
        for entity in kg.filter_dict[key]:
            rows.append(row)
            entities.append(entity)
    if rows:
        filtered[
            torch.tensor(rows, device=scores.device),
            torch.tensor(entities, device=scores.device),
        ] = -torch.inf
    targets = scores.gather(1, labels.unsqueeze(1))
    greater = filtered.gt(targets).sum(dim=1)
    tied = filtered.eq(targets).sum(dim=1)
    return greater + torch.div(tied, 2, rounding_mode="floor") + 1


def summarize_ranks(ranks, keys, directions, relation_types):
    ranks_array = np.asarray(ranks)
    mr, mrr, hit10, hit3, hit1 = metrics(ranks_array)
    relation_ranks = defaultdict(list)
    direction_ranks = defaultdict(list)
    for rank, key, direction in zip(ranks, keys, directions):
        relation_ranks[relation_types.get(key[1], "unknown")].append(rank)
        direction_ranks[direction].append(rank)

    def grouped(groups):
        output = {}
        for name, values in groups.items():
            g_mr, g_mrr, g_h10, g_h3, g_h1 = metrics(np.asarray(values))
            output[name] = {
                "mr": float(g_mr),
                "mrr": float(g_mrr),
                "hit10": float(g_h10),
                "hit3": float(g_h3),
                "hit1": float(g_h1),
                "num_queries": len(values),
            }
        return output

    return {
        "mr": float(mr),
        "mrr": float(mrr),
        "hit10": float(hit10),
        "hit3": float(hit3),
        "hit1": float(hit1),
        "relation_type_metrics": grouped(relation_ranks),
        "direction_metrics": grouped(direction_ranks),
        "num_queries": len(ranks),
    }


def format_metrics(result):
    fields = [
        f"MR={result['mr']:.4f}",
        f"MRR={result['mrr']:.6f}",
        f"H@1={result['hit1']:.6f}",
        f"H@3={result['hit3']:.6f}",
        f"H@10={result['hit10']:.6f}",
    ]
    for relation_type in ["1-1", "1-N", "N-1", "N-N"]:
        group = result["relation_type_metrics"].get(relation_type)
        if group:
            fields.append(f"MRR[{relation_type}]={group['mrr']:.6f}")
    for direction in ["head", "tail"]:
        group = result["direction_metrics"].get(direction)
        if group:
            fields.append(f"MRR[{direction}]={group['mrr']:.6f}")
    return " | ".join(fields)


def choose_directional(results):
    forward = max(
        results,
        key=lambda value: (
            results[value]["direction_metrics"]["tail"]["mrr"],
            -abs(value),
        ),
    )
    inverse = max(
        results,
        key=lambda value: (
            results[value]["direction_metrics"]["head"]["mrr"],
            -abs(value),
        ),
    )
    return {"forward": float(forward), "inverse": float(inverse)}


@torch.no_grad()
def evaluate_schema(
    model,
    kg,
    split,
    device,
    batch_size,
    relation_types,
    role_scales,
):
    model.eval()
    entities, relations = model()
    queries, labels, keys, directions = encoded_queries(kg, split, device)
    rank_by_scale = {float(scale): [] for scale in role_scales}
    for start in range(0, labels.shape[0], batch_size):
        stop = min(start + batch_size, labels.shape[0])
        base, bias = model.score_components(
            entities, relations, queries[start:stop]
        )
        for scale in rank_by_scale:
            ranks = filtered_ranks(
                base + scale * bias,
                labels[start:stop],
                keys[start:stop],
                kg,
            )
            rank_by_scale[scale].extend(ranks.cpu().tolist())
    results = {
        scale: summarize_ranks(ranks, keys, directions, relation_types)
        for scale, ranks in rank_by_scale.items()
    }
    selected_scales = choose_directional(results)
    selected_ranks = [
        rank_by_scale[selected_scales["inverse"]][index]
        if direction == "head"
        else rank_by_scale[selected_scales["forward"]][index]
        for index, direction in enumerate(directions)
    ]
    selected = summarize_ranks(selected_ranks, keys, directions, relation_types)
    return results, selected_scales, selected


def query_key(query, num_ent, num_rel):
    mask_id = num_ent + num_rel
    head, relation, tail = [int(value) for value in query]
    relation -= num_ent
    if head == mask_id:
        return (-1, relation, tail - num_rel)
    return (head - num_rel, relation, -1)


def filter_role_answers(scores, labels, queries, kg, train_filter):
    filtered = scores.clone()
    rows, entities = [], []
    for row, (query, label) in enumerate(
        zip(queries.detach().cpu().tolist(), labels.detach().cpu().tolist())
    ):
        for entity in train_filter.get(
            query_key(query, kg.num_ent, kg.num_rel), ()
        ):
            if entity != label:
                rows.append(row)
                entities.append(entity)
    if rows:
        filtered[
            torch.tensor(rows, device=scores.device),
            torch.tensor(entities, device=scores.device),
        ] = -torch.inf
    return filtered


def train_epoch(model, loader, kg, train_filter, optimizer, args, device):
    model.train()
    totals = defaultdict(float)
    for queries, labels in loader:
        queries = queries.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        entities, relations = model()
        base_scores, role_bias = model.score_components(
            entities, relations, queries
        )
        base_ce = F.cross_entropy(base_scores, labels)
        fgc = model.finegrained_contrastive_loss(entities)
        role_scores = filter_role_answers(
            base_scores.detach() + role_bias,
            labels,
            queries,
            kg,
            train_filter,
        )
        role_ce = F.cross_entropy(role_scores, labels)
        role_reg = model.role_regularization()
        loss = (
            base_ce
            + args.mu * fgc
            + args.lambda_role_ce * role_ce
            + args.lambda_role_reg * role_reg
        )
        loss.backward()
        base_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if name != "raw_role_strength"
        ]
        torch.nn.utils.clip_grad_norm_(base_parameters, args.grad_clip)
        torch.nn.utils.clip_grad_norm_(
            [model.raw_role_strength], args.role_grad_clip
        )
        optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["base_ce"] += float(base_ce.detach())
        totals["fgc"] += float(fgc.detach())
        totals["role_ce"] += float(role_ce.detach())
        totals["updates"] += 1
    denominator = max(1, int(totals["updates"]))
    for key in ["loss", "base_ce", "fgc", "role_ce"]:
        totals[key] /= denominator
    totals["role_strength"] = model.role_strength_summary()
    return totals


def clone_state(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def save_checkpoint(path, model, epoch, args, schema_scales, validation):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "stage": "ave_base",
            "args": vars(args),
            "selected_role_scales": schema_scales,
            "valid_result": validation,
        },
        path,
    )


def build_augmented_graph(kg):
    outgoing = [[] for _ in range(kg.num_ent)]
    incoming = [[] for _ in range(kg.num_ent)]
    for head, relation, tail in kg.train:
        for source, edge_relation, target in (
            (head, relation, tail),
            (tail, relation + kg.num_rel, head),
        ):
            outgoing[source].append((edge_relation, target))
            incoming[target].append((edge_relation, source))
    return outgoing, incoming


def mine_rules(kg, outgoing, min_support):
    direct = defaultdict(set)
    for head, relation, tail in kg.train:
        direct[(head, tail)].add(relation)
    body_count = Counter()
    support_count = Counter()
    path_count = 0
    for head, first_edges in enumerate(outgoing):
        for first_relation, middle in first_edges:
            for second_relation, tail in outgoing[middle]:
                if head == tail:
                    continue
                body = (first_relation, second_relation)
                body_count[body] += 1
                path_count += 1
                for target_relation in direct.get((head, tail), ()):
                    support_count[
                        (first_relation, second_relation, target_relation)
                    ] += 1
    rules = {}
    for rule, support in support_count.items():
        if support < min_support:
            continue
        confidence = support / body_count[rule[:2]]
        reliability = support / (support + 5.0)
        rules[rule] = confidence * reliability
    return rules, {
        "two_hop_paths": path_count,
        "body_types": len(body_count),
        "supported_rules": len(support_count),
        "retained_rules": len(rules),
        "min_rule_support": min_support,
    }


def path_evidence(query, num_ent, num_rel, outgoing, incoming, rules):
    mask_id = num_ent + num_rel
    head, relation, tail = [int(value) for value in query]
    relation -= num_ent
    evidence = {}
    if tail == mask_id:
        source = head - num_rel
        for first_relation, middle in outgoing[source]:
            for second_relation, candidate in outgoing[middle]:
                value = rules.get(
                    (first_relation, second_relation, relation), 0.0
                )
                if value > evidence.get(candidate, 0.0):
                    evidence[candidate] = value
    else:
        target = tail - num_rel
        for second_relation, middle in incoming[target]:
            for first_relation, candidate in incoming[middle]:
                value = rules.get(
                    (first_relation, second_relation, relation), 0.0
                )
                if value > evidence.get(candidate, 0.0):
                    evidence[candidate] = value
    return evidence


@torch.no_grad()
def evaluate_path(
    model,
    kg,
    split,
    device,
    batch_size,
    relation_types,
    outgoing,
    incoming,
    rules,
    schema_scales,
    path_alphas,
    selected_alphas=None,
):
    model.eval()
    entities, relations = model()
    queries, labels, keys, directions = encoded_queries(kg, split, device)
    rank_by_alpha = {float(alpha): [] for alpha in path_alphas}
    true_evidence = defaultdict(int)
    any_evidence = defaultdict(int)
    for start in range(0, labels.shape[0], batch_size):
        stop = min(start + batch_size, labels.shape[0])
        query_batch = queries[start:stop]
        label_batch = labels[start:stop]
        base, role_bias = model.score_components(
            entities, relations, query_batch
        )
        scales = torch.tensor(
            [
                schema_scales["inverse"]
                if direction == "head"
                else schema_scales["forward"]
                for direction in directions[start:stop]
            ],
            dtype=base.dtype,
            device=device,
        ).unsqueeze(1)
        schema_scores = base + scales * role_bias
        rows, candidates, values = [], [], []
        for row, (query, label) in enumerate(
            zip(
                query_batch.detach().cpu().tolist(),
                label_batch.detach().cpu().tolist(),
            )
        ):
            evidence = path_evidence(
                query, kg.num_ent, kg.num_rel, outgoing, incoming, rules
            )
            direction = directions[start + row]
            if evidence:
                any_evidence[direction] += 1
            if label in evidence:
                true_evidence[direction] += 1
            for candidate, value in evidence.items():
                rows.append(row)
                candidates.append(candidate)
                values.append(value)
        if rows:
            row_ids = torch.tensor(rows, device=device)
            candidate_ids = torch.tensor(candidates, device=device)
            path_values = torch.tensor(values, dtype=base.dtype, device=device)
        else:
            row_ids = candidate_ids = path_values = None
        for alpha in rank_by_alpha:
            scores = schema_scores.clone()
            if row_ids is not None and alpha:
                scores[row_ids, candidate_ids] += alpha * path_values
            ranks = filtered_ranks(
                scores, label_batch, keys[start:stop], kg
            )
            rank_by_alpha[alpha].extend(ranks.cpu().tolist())
    results = {
        alpha: summarize_ranks(ranks, keys, directions, relation_types)
        for alpha, ranks in rank_by_alpha.items()
    }
    if selected_alphas is None:
        selected_alphas = choose_directional(results)
    selected_ranks = [
        rank_by_alpha[selected_alphas["inverse"]][index]
        if direction == "head"
        else rank_by_alpha[selected_alphas["forward"]][index]
        for index, direction in enumerate(directions)
    ]
    selected = summarize_ranks(selected_ranks, keys, directions, relation_types)
    triple_count = len(getattr(kg, split))
    coverage = {
        "head_true": true_evidence["head"] / triple_count,
        "tail_true": true_evidence["tail"] / triple_count,
        "head_any": any_evidence["head"] / triple_count,
        "tail_any": any_evidence["tail"] / triple_count,
    }
    return results, selected_alphas, selected, coverage


def main(argv=None):
    args = parse_args(argv)
    role_scales = parse_grid(args.role_scales, include_one=True)
    path_alphas = parse_grid(args.path_alphas)
    set_seed(args.seed, deterministic=not args.non_deterministic)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by Ave")
    device = torch.device(args.device)
    run_name = f"final_seed{args.seed}_d{args.dim}_lr{args.lr:g}"
    configure_logging(args, run_name)
    LOGGER.info("PID=%s run=%s", os.getpid(), run_name)
    LOGGER.info("args=%s", json.dumps(vars(args), sort_keys=True))

    kg = VTKG(args.data, LOGGER)
    relation_types = build_relation_types(kg)
    train_filter = build_train_filter(kg.train)
    role_prior = build_relation_role_prior(
        kg.train,
        kg.num_ent,
        kg.num_rel,
        similar_roles=args.similar_roles,
        direct_weight=args.role_direct_weight,
    )
    prior_stats = role_prior_statistics(role_prior, kg.valid, kg.num_rel)
    LOGGER.info(
        "role_prior valid_coverage=%.4f forward=%.4f inverse=%.4f "
        "mean_support=%.1f/%d",
        prior_stats["coverage"],
        prior_stats["forward_coverage"],
        prior_stats["inverse_coverage"],
        prior_stats["mean_support"],
        kg.num_ent,
    )
    model = build_model(args, kg, role_prior, device)
    checkpoint_path = (
        Path("ckpt") / args.exp / args.data / f"{run_name}_best.ckpt"
    )
    result_path = (
        Path("result") / args.exp / args.data / f"{run_name}.json"
    )

    if args.base_ckpt:
        payload = torch.load(args.base_ckpt, map_location=device)
        model.load_state_dict(payload["model_state_dict"])
        best_epoch = int(payload.get("epoch", 0))
        schema_grid, schema_scales, best_validation = evaluate_schema(
            model,
            kg,
            "valid",
            device,
            args.eval_batch_size,
            relation_types,
            role_scales,
        )
        LOGGER.info("loaded base checkpoint=%s", args.base_ckpt)
        if not args.no_write:
            save_checkpoint(
                checkpoint_path,
                model,
                best_epoch,
                args,
                schema_scales,
                best_validation,
            )
            LOGGER.info("copied compatible checkpoint=%s", checkpoint_path)
    else:
        loader = torch.utils.data.DataLoader(
            kg,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, args.step_size, T_mult=2
        )
        best_mrr = -math.inf
        best_epoch = 0
        best_state = None
        best_validation = None
        schema_grid = None
        schema_scales = {"forward": 0.0, "inverse": 0.0}
        stale = 0
        started = time.time()
        LOGGER.info(
            "training=ave_fgc queries_per_epoch=%d expected_updates=%d",
            len(kg),
            math.ceil(len(kg) / args.batch_size),
        )
        for epoch in range(1, args.num_epoch + 1):
            stats = train_epoch(
                model, loader, kg, train_filter, optimizer, args, device
            )
            scheduler.step()
            strength = stats["role_strength"]
            LOGGER.info(
                "epoch=%d loss=%.6f base_ce=%.6f fgc=%.6f role_ce=%.6f "
                "role_strength=(f:%+.4f,i:%+.4f,max:%.4f) "
                "updates=%d elapsed=%.1fs",
                epoch,
                stats["loss"],
                stats["base_ce"],
                stats["fgc"],
                stats["role_ce"],
                strength["forward_mean"],
                strength["inverse_mean"],
                strength["maximum"],
                int(stats["updates"]),
                time.time() - started,
            )
            if epoch % args.valid_epoch != 0 and epoch != args.num_epoch:
                continue
            grid, scales, validation = evaluate_schema(
                model,
                kg,
                "valid",
                device,
                args.eval_batch_size,
                relation_types,
                role_scales,
            )
            LOGGER.info(
                "validation epoch=%d schema_scale=(forward:%g,inverse:%g) "
                "| %s | delta_MRR=%+.6f",
                epoch,
                scales["forward"],
                scales["inverse"],
                format_metrics(validation),
                validation["mrr"] - grid[0.0]["mrr"],
            )
            if validation["mrr"] > best_mrr + args.min_delta:
                best_mrr = validation["mrr"]
                best_epoch = epoch
                best_state = clone_state(model)
                best_validation = validation
                schema_grid = grid
                schema_scales = scales
                stale = 0
                if not args.no_write:
                    save_checkpoint(
                        checkpoint_path,
                        model,
                        epoch,
                        args,
                        scales,
                        validation,
                    )
                    LOGGER.info("saved validation-best checkpoint=%s", checkpoint_path)
            else:
                stale += 1
            if args.early_stop > 0 and stale >= args.early_stop:
                LOGGER.info("early stop after %d stale checks", stale)
                break
        if best_state is None:
            raise RuntimeError("no validation checkpoint was produced")
        model.load_state_dict(best_state)

    LOGGER.info(
        "best schema validation epoch=%d scale=(forward:%g,inverse:%g) | %s",
        best_epoch,
        schema_scales["forward"],
        schema_scales["inverse"],
        format_metrics(best_validation),
    )

    outgoing, incoming = build_augmented_graph(kg)
    rules, rule_stats = mine_rules(
        kg, outgoing, args.min_rule_support
    )
    LOGGER.info("rule_statistics=%s", json.dumps(rule_stats, sort_keys=True))
    path_grid, path_alphas_selected, path_validation, valid_coverage = (
        evaluate_path(
            model,
            kg,
            "valid",
            device,
            args.eval_batch_size,
            relation_types,
            outgoing,
            incoming,
            rules,
            schema_scales,
            path_alphas,
        )
    )
    LOGGER.info(
        "path validation alpha=(forward:%g,inverse:%g) | %s "
        "| delta_MRR=%+.6f",
        path_alphas_selected["forward"],
        path_alphas_selected["inverse"],
        format_metrics(path_validation),
        path_validation["mrr"] - path_grid[0.0]["mrr"],
    )
    test_alphas = sorted(
        {
            0.0,
            path_alphas_selected["forward"],
            path_alphas_selected["inverse"],
        }
    )
    test_grid, _, final_test, test_coverage = evaluate_path(
        model,
        kg,
        "test",
        device,
        args.eval_batch_size,
        relation_types,
        outgoing,
        incoming,
        rules,
        schema_scales,
        test_alphas,
        selected_alphas=path_alphas_selected,
    )
    LOGGER.info(
        "FINAL TEST schema_scale=(forward:%g,inverse:%g) "
        "path_alpha=(forward:%g,inverse:%g) | %s | path_delta_MRR=%+.6f",
        schema_scales["forward"],
        schema_scales["inverse"],
        path_alphas_selected["forward"],
        path_alphas_selected["inverse"],
        format_metrics(final_test),
        final_test["mrr"] - test_grid[0.0]["mrr"],
    )

    summary = {
        "run_name": run_name,
        "best_epoch": best_epoch,
        "checkpoint": str(
            checkpoint_path
            if not args.no_write or not args.base_ckpt
            else args.base_ckpt
        ),
        "schema_scales": schema_scales,
        "schema_validation": best_validation,
        "schema_validation_by_scale": schema_grid,
        "path_alphas": path_alphas_selected,
        "path_validation": path_validation,
        "path_validation_by_alpha": path_grid,
        "validation_path_coverage": valid_coverage,
        "test": final_test,
        "test_schema_ablation": test_grid[0.0],
        "test_by_path_alpha": test_grid,
        "test_path_coverage": test_coverage,
        "role_prior_validation_statistics": prior_stats,
        "rule_statistics": rule_stats,
        "args": vars(args),
    }
    if not args.no_write:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        LOGGER.info("wrote final summary=%s", result_path)
    return summary


if __name__ == "__main__":
    main()
