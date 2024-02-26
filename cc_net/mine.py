# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Main script to download a CC dump, remove duplicates, split by language and
filter the documents.

The pipeline parameters are described in the `Config` class.
"""

import hashlib
import json
import time
import os
import warnings
import logging
import contextlib
from argparse import ArgumentParser
from collections import defaultdict
from itertools import repeat
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union, Callable
from multiprocessing import Pool, Process, Queue, current_process

import func_argparse

# Local scripts
from cc_net import dedup, execution, jsonql, minify, perplexity, process_wet_file
from cc_net import regroup as regroup_module
from cc_net import split_by_lang

from cc_net.flat_hash_set import AbstractDedupHashSet, FlatHashSet
from cc_net.config import Config, CUTOFF_CSV, FILE_DIR


BASE_CONFIG = Config()

NORDIC_PILE_V2_CONFIG = Config(
    dump="2023-50",
    num_shards=1000,
    hash_in_mem=50,
    lang_whitelist=["sv", "no", "da", "is"],
    lang_threshold=0.2,
    pipeline=["dedup", "lid", "keep_lang", "sp", "lm", "pp_bucket", "drop", "split_by_lang"]
)

NORDIC_PILE_V2_DARDEL_CONFIG = NORDIC_PILE_V2_CONFIG._replace(
    hashes_task_mem=220,
    hashes_shards_per_task=20,
    mine_task_mem = 220,
    mine_task_timeout=10, #24,
    mine_task_cpus=120,
    mine_num_processes=90,
    cache_dir=Path("data/cache")
)

BYLANG_CONFIG = Config(
    config_name="by_lang",
    mined_dir="mined_by_lang",
    pipeline=list(BASE_CONFIG.pipeline[:-1]) + ["split_by_lang"],
)

REPRODUCE_CONFIG = Config(
    config_name="reproduce",
    dump="2019-09",
    mined_dir="reproduce",
    pipeline=["fetch_metadata", "keep_lang", "keep_bucket", "split_by_lang"],
    metadata="https://dl.fbaipublicfiles.com/cc_net/1.0.0",
    # Optional filtering:
    # It won't change much the execution speed, but decreases the disk requirement.
    # Restrict languages
    lang_whitelist=["fr"],
    # Restrict perplexity buckets
    # Top languages have been split in perplexity buckets according
    # to a Wikipedia trained LM.
    # The buckets from low perplexity (good) to high (bad) are:
    # ["head", "middle", "tail"]
    # Languages without a LM have only one bucket "all".
    # It won't change much the execution speed, but decreases the disk requirement.
    keep_bucket=["head", "all"],
    mine_num_processes=1,
)

TEST_CONFIG = BASE_CONFIG._replace(
    config_name="test",
    dump="2019-09",
    output_dir=Path("test_data"),
    execution="local",
    num_shards=4,
    num_segments_per_shard=1,
    hash_in_mem=2,
    mine_num_processes=2,
    lang_whitelist=["de", "it", "fr"],
    target_size="32M",
    cleanup_after_regroup=False,
    cache_dir=Path("test_data/wet_cache"),
)

PREDEF_CONFIGS = {
    "base": BASE_CONFIG,
    "nordic_pile_v2_dardel": NORDIC_PILE_V2_DARDEL_CONFIG,
    "by_lang": BYLANG_CONFIG,
    "test": TEST_CONFIG,
    "test_slurm": TEST_CONFIG._replace(execution="slurm,partition=dev"),
    "debug": TEST_CONFIG._replace(config_name="debug", mine_num_processes=0),
    "reproduce": REPRODUCE_CONFIG,
    "augment": BASE_CONFIG._replace(
        config_name="augment", dump="2019-13", lang_blacklist=["en"]
    ),
}


def tmp(output: Path) -> Path:
    return output.parent / (output.stem + ".tmp" + output.suffix)


def finalize(tmp_output: Path, output: Path) -> None:
    if not tmp_output.exists():
        warnings.warn(f"Targeted tmp output {tmp_output} doesn't exists.")
        return

    tmp_index = tmp_output.parent / (tmp_output.name + ".index")
    tmp_output.rename(output)

    if tmp_index.exists():
        tmp_index.rename(output.parent / (output.name + ".index"))


def _transpose(iterable: Sequence[Tuple[Any, ...]], n=-1) -> Tuple[List, ...]:
    if n < 0:
        n = len(iterable[0])
    columns: tuple = tuple([] for _ in range(n))
    for row in iterable:
        assert len(row) == n, f"Found tuple of len({len(row)}, expected {n}: {row}"
        for i in range(n):
            columns[i].append(row[i])

    return columns


def hashes(conf: Config) -> List[Path]:
    """Computes hashes for each shard."""

    hashes_dir = conf.output_dir / "hashes" / conf.dump
    outputs = [hashes_dir / f"{shard:04d}.bin" for shard in range(conf.num_shards)]
    missing_outputs = [(shard, o) for shard, o in enumerate(outputs) if not o.exists()]

    if not missing_outputs:
        return outputs

    hashes_dir.mkdir(parents=True, exist_ok=True)

    ex = conf.get_executor(
        f"hashes_{conf.dump}", 
        mem_gb=conf.hashes_task_mem, 
        timeout_hour=4
    )

    # Group shards in groups
    missing_outputs = list(jsonql.grouper(missing_outputs, conf.hashes_shards_per_task))
    missing_shards = [[shard for shard, _ in g] for g in missing_outputs]
    missing_outputs = [[output for _, output in g] for g in missing_outputs]
    ex(_hashes_shards, repeat(conf), missing_shards, missing_outputs)

    # Wait a bit so that files appears on the disk.
    time.sleep(20)
    assert all(o.exists() for o in outputs)
    return outputs


def _hashes_shards(conf: Config, shards: List[int], outputs: List[Path]):
    from cc_net.mine import _hashes_shard
    # Process all 20 shards in parallel
    with Pool(conf.hashes_shards_per_task) as p:
        p.starmap(_hashes_shard, zip(repeat(conf), shards, outputs))
    return f"Hashed shards {', '.join(map(str, shards))}"


def _hashes_shard(conf: Config, shard: int, output: Path):
    tmp_output = tmp(output)
    jsonql.run_pipes(
        dedup.HashesCollector(field="raw_content", output=tmp_output),
        inputs=conf.get_cc_shard(shard),
    )
    finalize(tmp_output, output)
    return f"Hashed {output}"


HASHES_IN_MEM = [0, 1, 2, 5, 10, 20, 50, 100, 200, 400]


def mine(conf: Config) -> List[Path]:
    """Remove dups, run LID and LMs, and split by lang and quality."""
    mined_dir = conf.get_mined_dir()
    if conf.will_split:
        # Give a directories when splitting
        outputs = [mined_dir / f"{shard:04d}" for shard in range(conf.num_shards)]
    else:
        # Files otherwise
        outputs = [
            mined_dir / f"{shard:04d}.json.gz" for shard in range(conf.num_shards)
        ]

    if "mini_again" in conf.experiments:
        mined_dir = conf.output_dir / "mini_again" / conf.dump
        outputs = [mined_dir / f"{shard:04d}" for shard in range(conf.num_shards)]

    # TODO: try to reduce this / make it a function of "hash_in_mem" / num_langs
    mem_gb = 120 #conf.hash_in_mem
    timeout_hour = 5
    if "hashes" in conf.experiments:
        # HACK: used for generating paper figures
        outputs = [
            conf.output_dir / f"hashes_exp/{conf.dump}_0000_dedup{h:03d}.json.gz"
            for h in HASHES_IN_MEM
        ]
        mem_gb = int(max(HASHES_IN_MEM) * 1.2)
        timeout_hour = 8

    missing_outputs = [(shard, o) for shard, o in enumerate(outputs) if not o.exists()]

    if "mini_again" in conf.experiments:
        missing_outputs = [
            (shard, o)
            for shard, o in enumerate(outputs)
            if shard in [5, 139] and not o.exists()
        ]

    if not missing_outputs:
        return outputs

    mined_dir.mkdir(parents=True, exist_ok=True)
    ex = conf.get_executor(
        f"mine_{conf.dump}",
        mem_gb=mem_gb,
        timeout_hour=timeout_hour,
        cpus=conf.mine_num_processes,
    )

    # Compute hashes firsts.
    if "dedup" in conf.pipeline:
        hashes_groups = list(jsonql.grouper(hashes(conf), conf.hash_in_mem))
        hashes_files: Iterable[List[Path]] = [
            hashes_groups[shard // conf.hash_in_mem] for shard, o in missing_outputs
        ]
    else:
        hashes_files = repeat([])

    ex(_mine_shard, repeat(conf), hashes_files, *_transpose(missing_outputs))

    assert all(o.exists() for o in outputs)
    return outputs


def _get_segment(tmp_output: Path, doc: dict) -> str:
    segment: str = doc["cc_segment"].split("/")[-1]
    return str(tmp_output / segment.replace(".warc.wet.gz", ".json.gz"))


def _mine_shard(conf: Config, hashes: List[Path], shard: int, output: Path) -> str:
    assert conf.pipeline
    tmp_output = tmp(output)
    if "hashes" in conf.experiments:
        # HACK: used for generating paper figures
        hashes_in_mem = shard
        hashes = hashes[: HASHES_IN_MEM[hashes_in_mem]]
        shard = 0
    cc_shard = conf.get_cc_shard(shard)

    steps: Dict[str, Optional[jsonql.Transformer]] = {}
    lang_id = Path(__file__).parent / ".."/ "bin" / "lid.bin"
    steps["lid_before_dedup"] = split_by_lang.Classifier(
        model=lang_id, field="raw_content", out_field="lid_before_dedup", top=5
    )
    steps["dedup"] = dedup.DuplicatesRemover(field="raw_content", hashes_files=hashes, load_parallelism=conf.mine_num_processes)

    steps["lid"] = split_by_lang.Classifier(
        model=lang_id,
        field="raw_content",
        out_field="language",
        top=1,
        threshold=conf.lang_threshold,
    )
    steps["lid_after_dedup"] = split_by_lang.Classifier(
        model=lang_id, field="raw_content", out_field="lid_after_dedup", top=5
    )

    if conf.lang_blacklist:
        steps["keep_lang"] = jsonql.where(
            [lambda doc: doc.get("language") not in set(conf.lang_blacklist)]
        )
    elif conf.lang_whitelist:
        steps["keep_lang"] = jsonql.where(
            [lambda doc: doc.get("language") in set(conf.lang_whitelist)]
        )
    else:
        steps["keep_lang"] = None

    tok_field = "tokenized"
    steps["sp"] = perplexity.MultiSentencePiece(
        {l: conf.lm_dir / f"{l}.sp.model" for l in conf.get_lm_languages()},
        field="raw_content",
        output_field=tok_field,
        normalize=True,
    )
    steps["lm"] = perplexity.DocLM(
        {l: conf.lm_dir / f"{l}.arpa.bin" for l in conf.get_lm_languages()},
        field=tok_field,
        output_field="perplexity",
        normalize=False,  # Normalization is done before SentencePiece
        # load_method=kenlm.LoadMethod.PARALLEL_READ,
    )
    steps["pp_bucket"] = perplexity.PerplexityBucket(CUTOFF_CSV)
    steps["drop"] = perplexity.DropKeys(tok_field)

    steps["keep_bucket"] = None
    if conf.keep_bucket:
        steps["keep_bucket"] = jsonql.where(
            [lambda doc: doc.get("bucket", "all") in conf.keep_bucket]
        )

    if "fetch_metadata" in conf.pipeline:
        # TODO: better default
        assert conf.metadata is not None
        steps["fetch_metadata"] = minify.MetadataFetcher(
            f"{conf.metadata}/{conf.dump}/"
        )

    steps["minify"] = minify.Minifier()

    pattern = str(tmp_output / "{language}_{bucket}.json.gz")
    steps["split_by_lang"] = jsonql.split(pattern=str(pattern), mkdir=True)

    steps["split_by_segment"] = jsonql.split(
        split_fn=lambda doc: _get_segment(tmp_output, doc), mkdir=True
    )

    pipeline = filter(None, (steps[s] for s in conf.pipeline))

    jsonql.run_pipes(
        *pipeline,
        inputs=cc_shard,
        processes=conf.mine_num_processes,
        chunksize=100,
        # The splitter takes care of writing to files.
        output=tmp_output if not conf.will_split else None,
    )
    finalize(tmp_output, output)
    return f"Mined {output}"


def mine_parallel(conf: Config) -> List[Path]:
    mined_dir = conf.get_mined_dir()
    if conf.will_split:
        # Give a directories when splitting
        outputs = [mined_dir / f"{shard:04d}" for shard in range(conf.num_shards)]
    else:
        # Files otherwise
        outputs = [
            mined_dir / f"{shard:04d}.json.gz" for shard in range(conf.num_shards)
        ]

    missing_outputs = [(shard, o) for shard, o in enumerate(outputs) if not o.exists()]
    if not missing_outputs:
        return outputs
    
    # Compute hashes firsts.
    hashes_groups = list(jsonql.grouper(hashes(conf), conf.hash_in_mem))
    mined_dir.mkdir(parents=True, exist_ok=True)

    # Request full node
    ex = conf.get_executor(
        f"mine_{conf.dump}",
        mem_gb=conf.mine_task_mem,
        timeout_hour=conf.mine_task_timeout,
        cpus=conf.mine_task_cpus,
    )
    # Group shards
    missing_shards = []
    missing_output_paths = []
    relevant_hashes_groups = []
    for group_idx in range(len(hashes_groups)):
        group_shards = [shard for shard, _ in missing_outputs if shard // conf.hash_in_mem == group_idx]
        group_output_paths = [p for shard, p in missing_outputs if shard // conf.hash_in_mem == group_idx]
        if len(group_shards) > 0:
            missing_shards.append(group_shards)
            missing_output_paths.append(group_output_paths)
            relevant_hashes_groups.append(hashes_groups[group_idx])
    
    ex(_mine_shard_group, repeat(conf), hashes_groups, missing_shards, missing_output_paths)

    assert all(o.exists() for o in outputs)
    return outputs


def _mine_shard_group(conf: Config, hashes: List[Path], shards: List[int], outputs: List[Path]):
    # Create and load FlatHashSet
    duplicates = FlatHashSet()
    duplicates.load_many(hashes, parallelism=4)

    # Process each shard in group sequentially (hard to parallelize as the FlatHashSet can't be memory shared easily)
    for shard, output in zip(shards, outputs):
        _mine_single_shard(conf, shard, output, duplicates)

    return f"Mined shards {', '.join(map(str, shards))}"


def _mine_single_shard(conf: Config, shard: int, output: Path, duplicates: AbstractDedupHashSet):
    # Create pipeline
    # Run jsonql.run_pipes(...)
    assert conf.pipeline
    tmp_output = tmp(output)
    cc_shard = conf.get_cc_shard(shard)

    steps: Dict[str, Optional[jsonql.Transformer]] = {}
    lang_id = Path(__file__).parent / ".."/ "bin" / "lid.bin"
    steps["lid_before_dedup"] = split_by_lang.Classifier(
        model=lang_id, field="raw_content", out_field="lid_before_dedup", top=5
    )
    steps["dedup"] = dedup.DuplicatesRemover(
        field="raw_content", duplicates=duplicates
    )

    steps["lid"] = split_by_lang.Classifier(
        model=lang_id,
        field="raw_content",
        out_field="language",
        top=1,
        threshold=conf.lang_threshold,
    )
    steps["lid_after_dedup"] = split_by_lang.Classifier(
        model=lang_id, field="raw_content", out_field="lid_after_dedup", top=5
    )

    if conf.lang_blacklist:
        steps["keep_lang"] = jsonql.where(
            [lambda doc: doc.get("language") not in set(conf.lang_blacklist)]
        )
    elif conf.lang_whitelist:
        steps["keep_lang"] = jsonql.where(
            [lambda doc: doc.get("language") in set(conf.lang_whitelist)]
        )
    else:
        steps["keep_lang"] = None

    tok_field = "tokenized"
    steps["sp"] = perplexity.MultiSentencePiece(
        {l: conf.lm_dir / f"{l}.sp.model" for l in conf.get_lm_languages()},
        field="raw_content",
        output_field=tok_field,
        normalize=True,
    )
    steps["lm"] = perplexity.DocLM(
        {l: conf.lm_dir / f"{l}.arpa.bin" for l in conf.get_lm_languages()},
        field=tok_field,
        output_field="perplexity",
        normalize=False,  # Normalization is done before SentencePiece
        # load_method=kenlm.LoadMethod.PARALLEL_READ,
    )
    steps["pp_bucket"] = perplexity.PerplexityBucket(CUTOFF_CSV)
    steps["drop"] = perplexity.DropKeys(tok_field)

    steps["keep_bucket"] = None
    if conf.keep_bucket:
        steps["keep_bucket"] = jsonql.where(
            [lambda doc: doc.get("bucket", "all") in conf.keep_bucket]
        )

    if "fetch_metadata" in conf.pipeline:
        # TODO: better default
        assert conf.metadata is not None
        steps["fetch_metadata"] = minify.MetadataFetcher(
            f"{conf.metadata}/{conf.dump}/"
        )

    steps["minify"] = minify.Minifier()

    pattern = str(tmp_output / "{language}_{bucket}.json.gz")
    steps["split_by_lang"] = jsonql.split(pattern=str(pattern), mkdir=True)

    steps["split_by_segment"] = jsonql.split(
        split_fn=lambda doc: _get_segment(tmp_output, doc), mkdir=True
    )

    pipeline = filter(None, (steps[s] for s in conf.pipeline))

    run_pipes_queue_feeder(
        *pipeline,
        segments_reader=cc_shard,
        processes=conf.mine_num_processes,
    )
    
    finalize(tmp_output, output)


def segment_queue_feeder(segment_queue, doc_chunk_queue, cache_dir: Path, min_len: int, doc_chunk_size: int):
    while segment := segment_queue.get():
        url = process_wet_file.segment_url(segment)
        file = None
        if cache_dir:
            file = cache_dir / segment.split("/")[-1]

        chunk = []
        for doc in process_wet_file.parse_warc_file(jsonql.open_remote_file(url, cache=file), min_len=min_len):
            doc["cc_segment"] = segment
            chunk.append(doc)

            if len(chunk) == doc_chunk_size:
                doc_chunk_queue.put(chunk)
                chunk = []

        doc_chunk_queue.put(chunk)


def pipeline_queue_consumer(doc_chunk_queue, output_queue, transform):
    while (chunk := doc_chunk_queue.get()) is not None:
        for doc in chunk:
            out = transform(doc)
            if out is not None:
                output_queue.put(out)


def run_pipes_from_queue(output_queue, pipes):
    with contextlib.ExitStack() as stack:
        pipes = [stack.enter_context(pipe) if isinstance(pipe, jsonql.Transformer) else pipe for pipe in pipes]
        #count = 0
        while (doc := output_queue.get()) is not None:
            for fn in pipes:
                doc = fn(doc)


def run_pipes_queue_feeder(
    *fns: Union[jsonql.Transformer, Callable[[Iterable], Iterable]],
    segments_reader,
    processes: int = 1,
    doc_chunk_size: int = 64,
    doc_chunk_queue_size = 10_000,
):
    transformers = []
    for t in fns:
        if not isinstance(t, jsonql.Transformer):
            break
        if not t.parallelisable:
            break
        transformers.append(t)
    pipes = fns[len(transformers) :]

    log = logging.getLogger(__name__).info

    with contextlib.ExitStack() as stack:
        log(f"Preparing {transformers}")
        transform = stack.enter_context(jsonql.compose(transformers))

        segment_queue = Queue()
        for segment in segments_reader.segments:
            segment_queue.put(segment)
        for _ in range(processes):
            segment_queue.put(None)
        doc_chunk_queue = Queue(maxsize=doc_chunk_queue_size)
        producers = [
            Process(target=segment_queue_feeder, args=(segment_queue, doc_chunk_queue, segments_reader.cache_dir, segments_reader.min_len, doc_chunk_size)) 
            for _ in range(processes)
        ]
        log(f"Starting {processes} parallel segment reader processes")
        for p in producers:
            p.start()
        
        log(f"Starting {processes} parallel pipeline processes")
        output_queue = Queue()
        consumers = [Process(target=pipeline_queue_consumer, args=(doc_chunk_queue, output_queue, transform)) for _ in range(processes)]
        for p in consumers:
            p.start()

        log("Starting pipes process")
        output_writer_proc = Process(target=run_pipes_from_queue, args=(output_queue, pipes))
        output_writer_proc.start()

        # Wait for producers to finish
        log("Waiting for producers to finish")
        for p in producers:
            p.join()
        # When there are no producers left, signal end to all consumers
        log("Signal end to all consumers")
        for _ in range(processes):
            doc_chunk_queue.put(None)
        # Wait for all consumers
        log("Wait for all consumers")
        for p in consumers:
            p.join()
        # When there are no consumers, signal end to pipes process
        log("Signal end to pipes process")
        output_queue.put(None)
        output_writer_proc.join()

        log("All processes finished successfully")


def regroup(conf: Config, all_dirs: List[Path]) -> Path:
    """Reshards each language/quality after 'mine'."""
    regroup_dir = conf.get_mined_dir(regroup=True)
    assert all_dirs
    all_files = [f for d in all_dirs for f in d.glob("*.json.gz")]
    if not all_files:
        logging.info(f"No .json.gz file found in {all_dirs[0]}")

    splits: Dict[str, List[Path]] = defaultdict(list)
    for f in all_files:
        split = f.name.split(".")[0]
        splits[split].append(f)

    logging.info(f"Identified {len(all_files)} files to regroup from {len(splits)} splits.")
    inputs: List[List[Path]] = []
    outputs: List[Path] = []
    target_size = jsonql.parse_size(conf.target_size)
    for split, files in splits.items():
        cuts = list(regroup_module.determine_groups(files, target_size=target_size))
        if not cuts:
            continue

        pattern = f"{split}_????.json.gz"
        existing_outputs = sorted(regroup_dir.glob(pattern))

        if not conf.cleanup_after_regroup:
            # We still have all the inputs so it is safe to overwrite existing outputs.
            assert len(existing_outputs) <= len(cuts)
            existing_outputs = []

        if len(existing_outputs) > 0 and len(cuts) == 1:
            # append to existing file if size allows it.
            new_size = (
                sum(f.stat().st_size for f in cuts[0])
                + existing_outputs[-1].stat().st_size
            )
            if new_size < target_size:
                logging.info(f"Will append {cuts[0]} to {existing_outputs[-1]}")
                cuts[0].insert(0, existing_outputs.pop(-1))

        n_existing = len(existing_outputs)
        for i, cut in enumerate(cuts):
            # avoid overwriting existing files.
            j = i + n_existing
            output = regroup_dir / f"{split}_{j:04}.json.gz"
            inputs.append(cut)
            outputs.append(output)
        logging.info(f"{regroup_dir / pattern} -> {len(cuts)} shards ({n_existing} already there).")

    ex = conf.get_executor(f"regroup_{conf.dump}", mem_gb=4, timeout_hour=12, cpus=2)
    ex(_regroup, repeat(conf), inputs, outputs)

    return regroup_dir


def _regroup(conf: Config, inputs: List[Path], output: Path) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    regroup_module.fast_reshard(
        inputs, output, tmp=tmp(output), rm_original=conf.cleanup_after_regroup
    )
    return f"Regrouped {output}"


def move_segments(conf: Config, all_dirs: Sequence[Path]) -> Path:
    """Reshards each language/quality after 'mine'."""
    # check that mining is over.
    regroup_dir = conf.get_mined_dir(regroup=True)
    assert all_dirs, "Received no dirs to move"
    assert all(
        d.is_dir() for d in all_dirs
    ), f"move_segments was expecting dirs received files: {all_dirs[:10]}..."

    regroup_dir.parent.mkdir(exist_ok=True)
    regroup_dir.mkdir(exist_ok=True)
    ex = conf.get_executor(f"moveseg_{conf.dump}", mem_gb=1, timeout_hour=1, cpus=2)

    def _move_segments(subdir: Path, regroup_dir: Path) -> str:
        n = 0
        for f in subdir.iterdir():
            if not f.is_file() or f.is_symlink():
                continue
            n += f.name.endswith(".json.gz")
            new_name = regroup_dir / f.name
            target = new_name.resolve()
            assert f.resolve() != target
            # this make the job idempotent.
            f.rename(new_name)
            f.symlink_to(target)

        if n == 0:
            return ""

        return f"Moved {n} .json.gz files from {subdir} to {regroup_dir}"

    ex(_move_segments, all_dirs, repeat(regroup_dir))
    logging.info(f"Results are in {regroup_dir}")
    return regroup_dir


def _validate_test(conf: Config, output_dir: Path, generate: bool = False):
    stats: Dict[str, dict] = {}
    for file in sorted(output_dir.glob("*.json.gz")):
        fname = "/".join((file.parent.name, file.name))
        # The order of documents is not guaranteed inside a shard,
        lines = sorted(jsonql.open_read(file))
        content = "\n".join(lines)
        size = len(content)
        checksum = hashlib.sha1(bytes(content, encoding="utf-8")).hexdigest()
        # first_document = json.loads(lines[0])
        stats[fname] = {"size": size, "checksum": checksum}

    def dump(x):
        return json.dumps(x, indent=2, ensure_ascii=False)

    logging.info("*** Stats ***")
    stats_raw = dump(stats)
    stats_file = FILE_DIR / "data" / "test_stats.json"
    if generate:
        logging.info(f"Saving stats to {stats_file}")
        stats_file.write_text(stats_raw)
        return

    expected_stats: Dict[str, dict] = {}
    if stats_file.exists():
        expected_stats = json.loads(stats_file.read_text())

    if expected_stats == stats:
        logging.info("Everything looks good !")
        return

    stats_file.with_suffix(".actual.json").write_text(stats_raw)
    logging.info("*** Expected Stats ***")
    logging.info(dump(expected_stats))

    logging.info("*** Diff ***")
    for fname in sorted(expected_stats.keys()):
        logging.info(fname)
        assert fname in expected_stats, "missing file " + fname
        if expected_stats[fname]["size"] != stats[fname]["size"]:
            logging.info(f"  - Expected size {expected_stats[fname]['size']}, size {stats[fname]['size']}")
        if expected_stats[fname]["checksum"] != stats[fname]["checksum"]:
            logging.info(f"  - Expected checksum {expected_stats[fname]['checksum']}, checksum {stats[fname]['checksum']}")


def get_main_parser() -> ArgumentParser:
    # Generates the 'main' parser by patching a 'Config' parser
    p = func_argparse.func_argparser(Config)

    # Override defaults value to None, so we know what was set by the user.
    # Note that it will keep the original default values in the help message.
    p.set_defaults(**{f: None for f in Config._fields})
    p.add_argument("--config", type=str, default="base")
    p.set_defaults(__command=main)
    return p


def main(config: str = "base", **config_as_dict: Any) -> None:
    # Use the given 'config' as default value.
    config_base = config
    if config_base in PREDEF_CONFIGS:
        conf = PREDEF_CONFIGS[config_base]
    elif Path(config_base).exists():
        conf = Config.from_json(Path(config_base))
    else:
        raise ValueError(
            f"Invalid value {config_base} for --config. "
            f"Choose from ({', '.join(PREDEF_CONFIGS)}) or give an existing .json file."
        )
    conf = conf._replace(**{k: v for (k, v) in config_as_dict.items() if v is not None})

    logging.info(f"Will run cc_net.mine.main with the following config: {conf}")

    #all_files = mine(conf)
    all_files = mine_parallel(conf)
    if conf.will_split:
        assert all_files
        assert all(d.is_dir() for d in all_files)
        all_dirs = all_files
        if "split_by_lang" in conf.pipeline:
            # Only try regrouping if we split the shards.
            regroup(conf, all_dirs)
        elif "split_by_segment" in conf.pipeline:
            # If we split by segment then regrouping is trivial, since segments appear in only one shard.
            move_segments(conf, all_dirs)

    if conf.config_name == "test":
        _validate_test(conf, conf.get_mined_dir(regroup=True))


if __name__ == "__main__":
    func_argparse.parse_and_call(get_main_parser())
