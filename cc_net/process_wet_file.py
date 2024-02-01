# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

import contextlib
import functools
import logging
import re
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import ContextManager, Iterable, Iterator, List, Optional, Sequence
from urllib.parse import urlparse
from multiprocessing import Pool, Queue, Manager, Process

import func_argparse
from bs4 import BeautifulSoup  # type: ignore

from cc_net import jsonql

WET_URL_ROOT = "https://data.commoncrawl.org"


logger = logging.getLogger(__name__)


def cc_wet_paths_url(dump_id: str) -> str:
    return "/".join([WET_URL_ROOT, "crawl-data", "CC-MAIN-" + dump_id, "wet.paths.gz"])


@functools.lru_cache()
def cc_segments(dump_id: str, cache_dir: Path = None) -> List[str]:
    wet_paths = cc_wet_paths_url(dump_id)
    cache_dir = cache_dir or jsonql._tmp_dir()
    wet_paths_cache = cache_dir / f"wet_{dump_id}.paths.gz"
    f = jsonql.open_remote_file(wet_paths, cache=wet_paths_cache)
    return [segment.strip() for segment in f]


def list_dumps() -> List[str]:
    home_page = BeautifulSoup(
        urllib.request.urlopen("http://index.commoncrawl.org/"), features="html.parser"
    )
    dumps = [a.get("href").strip("/") for a in home_page.findAll("a")]
    dumps = [a[8:] for a in dumps if re.match(r"^CC-MAIN-\d\d\d\d-\d\d$", a)]

    return sorted(dumps)


def ls():
    for dump in list_dumps():
        print(dump, "->", cc_wet_paths_url(dump))


def parse_doc(headers: List[str], doc: List[str]) -> Optional[dict]:
    """Headers format is:
    WARC/1.0
    WARC-Type: conversion
    WARC-Target-URI: [url]
    WARC-Date: [crawldate: 2019-02-15T19:15:59Z]
    WARC-Record-ID: <urn:uuid:8865156e-d5f1-4734-9c68-4b46eaf2bb7e>
    WARC-Refers-To: <urn:uuid:340152e2-65cf-4143-b522-8ce4e2d069d7>
    WARC-Block-Digest: sha1:S3DTWCONT2L6ORTGCY2KXEZ37LNBB7V2
    Content-Type: text/plain
    Content-Length: 7743
    """
    if not headers or not doc:
        return None

    try:
        headers_dict = {}
        for header in headers:
            try:
                k, v = header.split(": ")
                headers_dict[k] = v
            except:
                pass

        warc_type = headers_dict["WARC-Type"]
        if warc_type != "conversion":
            return None
        url = headers_dict["WARC-Target-URI"]
        date = headers_dict["WARC-Date"]
        digest = headers_dict["WARC-Block-Digest"]
        length = int(headers_dict["Content-Length"])

    except Exception as e:
        logger.warning("Can't parse header:", e, headers, doc)
        return None

    # Docs are separated by two empty lines.
    last = None
    if not doc[-1] and not doc[-2]:
        last = -2
    title, doc = doc[0], doc[1:last]

    return {
        "url": url,
        "date_download": date,
        "digest": digest,
        "length": length,
        "nlines": len(doc),
        "source_domain": urlparse(url).netloc,
        "title": title,
        "raw_content": "\n".join(doc),
    }


def group_by_docs(warc_lines: Iterable[str]) -> Iterable[dict]:
    doc: List[str] = []
    headers, read_headers = [], True
    for warc in warc_lines:
        warc = warc.strip()
        if read_headers:
            headers.append(warc)
            read_headers = warc != ""
            continue

        if warc == "WARC/1.0":
            # We reached the beginning of the new doc.
            parsed = parse_doc(headers, doc)
            if parsed is not None:
                yield parsed
            headers, doc, read_headers = [warc], [], True
            continue

        doc.append(warc)

    # Return the last document
    if doc:
        parsed = parse_doc(headers, doc)
        if parsed is not None:
            yield parsed


def parse_warc_file(lines: Iterable[str], min_len: int = 1) -> Iterator[dict]:
    n_doc = 0
    n_ok = 0
    for doc in group_by_docs(lines):
        n_doc += 1
        if not doc or len(doc["raw_content"]) < min_len:
            continue
        n_ok += 1
        yield doc
    if n_doc > 0:
        logger.info(f"Kept {n_ok:_d} documents over {n_doc:_d} ({n_ok / n_doc:.1%}).")
    else:
        logger.info(f"Found no documents")


def segment_url(segment: str):
    return "/".join((WET_URL_ROOT, segment))


def process_segment(segment, cache_dir, min_len, queue):
    url = segment_url(segment)
    file: Optional[Path] = None
    if cache_dir:
        file = cache_dir / segment.split("/")[-1]

    for doc in parse_warc_file(jsonql.open_remote_file(url, cache=file), min_len=min_len):
        doc["cc_segment"] = segment
        queue.put(doc)

    # Signal EOF
    #queue.put(None)


import random
def consumer(queue):
    while chunk := queue.get():
        if random.randint(0, 50000) == 0:
            print(queue.qsize())


def dl(
    dump: str,
    shard: int,
    num_shards: int,
    output: Path = None,
    num_segments_per_shard: int = 0,
    parallelism: int = 1,
    cache_dir: Optional[str] = None
):
    """Download a shard of the common crawl, and export it to json.

    Arguments:
        output: filename of the output file
        dump: CC dump id
        shard: id of the shard
        num_shards: total number of shards
        num_segments_per_shard: manual control of the number of segment per shard.
    """
    reader = CCShardReader(
        dump, 
        shard, 
        num_shards, 
        num_segments_per_shard, 
        cache_dir=Path(cache_dir) if cache_dir is not None else None, 
        parallelism=parallelism
    )
    #reader.read()
    jsonql.run_pipes(inputs=reader, output=output)
    logger.info(f"Done. {output} is ready.")


class CCSegmentsReader(Iterable[dict]):
    def __init__(
        self, segments: Sequence[str], min_len: int = 0, cache_dir: Path = None, parallelism: int = 1
    ):
        self._segments = segments
        self.min_len = min_len
        self.parallelism = parallelism
        if cache_dir is not None:
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(exist_ok=True)
        self.cache_dir = cache_dir
        self.retrieved_segments = 0

    @property
    def segments(self) -> Sequence[str]:
        return self._segments

    def open_segment(self, segment: str) -> Iterable[str]:
        url = segment_url(segment)
        file: Optional[Path] = None
        if self.cache_dir:
            file = self.cache_dir / segment.split("/")[-1]
        if not file or not file.exists():
            self.retrieved_segments += 1

        return jsonql.open_remote_file(url, cache=file)

    def read(self):
        #manager = Manager()
        segment_queue = Queue()
        for segment in self.segments:
            segment_queue.put(segment)
        for _ in range(self.parallelism):
            segment_queue.put(None)

        doc_queue = Queue(maxsize=10_000)
        #output_queue = manager.Queue(maxsize=10_000)
        #with Pool(self.parallelism) as producer_pool:
        producers = [Process(target=process_segment_queue, args=(segment_queue, doc_queue, self.cache_dir, self.min_len)) for _ in range(self.parallelism)]
        for p in producers:
            p.start()

        print("Started producers")
        #res = [producer_pool.apply_async(process_segment, args=(segment, self.cache_dir, self.min_len, input_queue)) for segment in self.segments]

        consumers = [Process(target=consumer, args=(doc_queue,)) for _ in range(self.parallelism)]
        for p in consumers:
            p.start()

        print("Started consumers")

        #for r in res:
        #    r.wait()
        for p in producers:
            p.join()

        print("All producers finished. Sending signals to consumers")

        for _ in range(self.parallelism):
            doc_queue.put(None)

        for p in consumers:
            p.join()
        
        print("All consumers finished")

    """
    def __iter__(self) -> Iterator[dict]:
        import random, sys
        manager = Manager()
        q = manager.Queue(maxsize=10_000)
        with Pool(self.parallelism) as p:
            res = [p.apply_async(process_segment, args=(segment, self.cache_dir, self.min_len, q)) for segment in self.segments]
            n_finished = 0
            while n_finished < len(self.segments):
                if random.randint(0, 1000) == 0:
                    print(q.qsize(), file=sys.stderr)
                doc = q.get()
                if doc is None:
                    n_finished += 1
                    logger.info(f"Finished {n_finished}/{len(self.segments)} segments")
                else:
                    yield doc
            for r in res:
                r.wait()
    """
    def __iter__(self) -> Iterator[dict]:
        n = len(self.segments)
        for i, segment in enumerate(self.segments):
            start = time.time()
            # TODO: start downloading the next segment in the background
            chunk = []
            for doc in parse_warc_file(self.open_segment(segment), self.min_len):
                doc["cc_segment"] = segment
                chunk.append(doc)

                if len(chunk) == 64:
                    yield chunk
                    chunk = []

            if i + 1 >= n:
                continue
            end = time.time()
            delay = (end - start) / 3600 * (n - 1 - i)
            logger.info(
                f"Parsed {i + 1} / {n} files. Estimated remaining time: {delay:.1f}h"
            )
    


class CCShardReader(CCSegmentsReader):
    def __init__(
        self,
        dump: str,
        shard: int,
        num_shards: int = -1,
        num_segments_per_shard: int = 40,
        min_len: int = 300,
        cache_dir: Path = None,
        parallelism: int = 1
    ):
        """Downloads a shard of Common Crawl, and yields dict.

        Arguments:
            dump: CC dump id
            shard: id of the shard
            num_shards: total number of shards
            num_segments_per_shard: if set will limit the number of files by shard.
                Useful for testing.
        """
        super().__init__([], min_len=min_len, cache_dir=cache_dir, parallelism=parallelism)
        self.dump = dump
        self.shard = shard
        assert num_shards > 0 or num_segments_per_shard > 0
        self.num_shards = num_shards
        self.num_segments_per_shard = num_segments_per_shard

    @property
    def segments(self) -> Sequence[str]:
        # Delaying the initialization allows to delay the looking up of the WET files
        if self._segments:
            return self._segments
        segments = cc_segments(self.dump, self.cache_dir)
        n = len(segments)
        if self.num_shards < 0:
            self.num_shards = n // self.num_segments_per_shard
        i_min = (self.shard * n) // self.num_shards
        i_max = ((self.shard + 1) * n) // self.num_shards
        if self.num_segments_per_shard > 0:
            i_max = min(i_max, i_min + self.num_segments_per_shard)
        self._segments = segments[i_min:i_max]
        return self._segments


def _tmp(prefix: str = None, suffix: str = None, dir: Path = None) -> Path:
    _, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=dir)
    return Path(tmp_path)


@contextlib.contextmanager
def timer(name: str = "-"):
    start = time.time()
    yield None
    delay = time.time() - start
    print(f"{name} took {delay:.1f}s")


def benchmark(tmp_path: Path):
    segments = [
        "crawl-data/CC-MAIN-2019-09/segments/1550249406966.99/wet/CC-MAIN-20190222220601-20190223002601-00441.warc.wet.gz"
    ]
    seg_file = tmp_path / "CC-MAIN-20190222220601-20190223002601-00441.warc.wet.gz"

    with timer("from network"):
        list(CCSegmentsReader(segments))

    with timer("from network, with caching"):
        list(CCSegmentsReader(segments, cache_dir=tmp_path))
    assert seg_file.exists()

    with timer("from disk"):
        CCSegmentsReader(segments, cache_dir=tmp_path)
    seg_file.unlink()


if __name__ == "__main__":
    func_argparse.main(ls, dl)
