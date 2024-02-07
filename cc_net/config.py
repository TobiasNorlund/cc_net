from typing import NamedTuple, Optional, Sequence
from pathlib import Path
from cc_net.execution import Executor
from cc_net import execution, process_wet_file
import json

# Constant
FILE_DIR = Path(__file__).parent
CUTOFF_CSV = FILE_DIR / "data" / "cutoff.csv"


DEFAULT_PIPELINE = [
    "dedup",
    "lid",
    "keep_lang",
    "sp",
    "lm",
    "pp_bucket",
    "drop",
    "split_by_lang",
]


class Config(NamedTuple):
    """
    Mine Common Crawl with the given settings.

    config_name
    dump: CC dump id
    output_dir: working directory
    mined_dir: name of the destination folder, full path will be {ouput_dir}/{mined_dir}/{dump_id}
    execution: chose how to parallelize the execution
    num_shards: number of shards to split the dump
    num_segments_per_shard: allow to download a small portion of CC (eg for tests)
    min_len: remove documents shorter than this (in chars)
    hashes_in_mem: number of shards hashes to use for dedup
    lang_whitelist: only treat those languages
    lang_blacklist: ignore those languages
    lang_threshold: remove docs whose top language score is lower than this
    keep_bucket: keep only those perplexity bucket chose from (head, middle, tail, all)
    lm_dir: folder containing LMs
    lm_languages: only use LMs for the following languages
    cutoff: cutoff file to use for split in head/middle/tail
    mine_num_processes: number of processes to use for mining
    target_size: size of finals files produce during `regroup` stage
    cleanup_after_regroup: delete intermediary files after regroup
    task_parallelism: max number of task to run in parallel
    pipeline: restricts the mining pipeline to the given steps. Order is important !
    experiments: (HACK) enable specific experiments in the code
    """

    config_name: str = "base"
    dump: str = "2017-51"
    output_dir: Path = Path("data")
    mined_dir: str = "mined"
    execution: str = "auto"
    num_shards: int = 1600
    num_segments_per_shard: int = -1
    metadata: Optional[str] = None
    min_len: int = 300
    hash_in_mem: int = 50
    lang_whitelist: Sequence = []
    lang_blacklist: Sequence[str] = []
    lang_threshold: float = 0.5
    keep_bucket: Sequence[str] = []
    lm_dir: Path = Path(__file__).parent / ".." / "data/lm_sp"
    cutoff: Path = CUTOFF_CSV
    lm_languages: Optional[Sequence[str]] = None
    mine_num_processes: int = 16
    target_size: str = "4G"
    cleanup_after_regroup: bool = True
    task_parallelism: int = -1
    pipeline: Sequence[str] = DEFAULT_PIPELINE
    experiments: Sequence[str] = []
    cache_dir: Optional[Path] = None

    # hashes infra
    hashes_task_mem: int = 220  # memory in GB for one hashes task
    hashes_shards_per_task: int = 1  # The number of shards to hash (in parallel) in each task
    
    # mine infra
    mine_task_mem: int = 220  # memory in GB for one mine task
    mine_task_timeout: int = 12  # timeout in hours
    mine_task_cpus: int = 120  # num cpu cores for one mine task

    def get_executor(
        self, name: str, timeout_hour: int = 1, mem_gb: int = 1, cpus: int = 1, **options
    ) -> Executor:
        name = "_".join((name, self.config_name, *self.experiments))
        return execution.get_executor(
            name,
            self.output_dir / "logs",
            self.execution,
            timeout_hour=timeout_hour,
            mem_gb=mem_gb,
            cpus=cpus,
            task_parallelism=self.task_parallelism,
            options=options
        )

    def get_cc_shard(self, shard: int) -> process_wet_file.CCShardReader:
        dump_cache: Optional[Path] = None
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)
            dump_cache = self.cache_dir / self.dump
            dump_cache.mkdir(exist_ok=True)

        return process_wet_file.CCShardReader(
            self.dump,
            shard=shard,
            num_shards=self.num_shards,
            num_segments_per_shard=self.num_segments_per_shard,
            min_len=self.min_len,
            cache_dir=dump_cache,
        )

    @classmethod
    def from_json(cls, json_file: Path) -> "Config":
        raw_lines = json_file.read_text().splitlines()
        raw_lines = [l for l in raw_lines if not l.strip().startswith("//")]
        json_config = json.loads("".join(raw_lines))
        path_keys = ["cache_dir", "lm_dir", "output_dir"]
        for key in path_keys:
            if key in json_config:
                json_config[key] = Path(json_config[key])
        return Config(**json_config)

    @property
    def will_split(self) -> bool:
        return "split_by_lang" in self.pipeline or "split_by_segment" in self.pipeline

    def get_lm_languages(self) -> Sequence[str]:
        if self.lm_languages is not None:
            return self.lm_languages

        if self.lang_whitelist:
            return self.lang_whitelist

        languages = [m.name.split(".")[0] for m in self.lm_dir.glob("*.arpa.bin")]
        if self.lang_blacklist:
            languages = [l for l in languages if l not in self.lang_blacklist]
        return languages

    def get_mined_dir(self, regroup: bool = False) -> Path:
        if self.will_split and not regroup:
            return self.output_dir / f"{self.mined_dir}_split" / self.dump
        return self.output_dir / self.mined_dir / self.dump

