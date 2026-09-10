import json
import time
from pathlib import Path


class DispatchLogger:
    """Records which requests are scheduled at each execute_model step.

    One line per ``execute_model()`` call, written alongside the msprobe
    ``generate_sequence`` dump so that dispatch records can be correlated
    with ``generate_sequence/step{N}`` through the shared ``request_id``.

    Output file: ``{dump_path}/{pid}/dispatch_log.jsonl``
    """

    def __init__(self, dump_path: str, pid: int, rank: int = 0):
        log_dir = Path(dump_path) / str(pid)
        log_dir.mkdir(parents=True, exist_ok=True)
        self._fp = open(log_dir / "dispatch_log.jsonl", "w")
        self._step_counter = 0
        self._pid = pid
        self._rank = rank

    def log_step(self, scheduler_output, attn_state) -> None:
        from vllm_ascend.attention.attention_v1 import AscendAttentionState

        is_prefill = attn_state != AscendAttentionState.DecodeOnly

        requests = []
        for req in scheduler_output.scheduled_new_reqs:
            requests.append({
                "request_id": req.req_id,
                "type": "new",
                "tokens": scheduler_output.num_scheduled_tokens.get(req.req_id, 0),
            })
        for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
            requests.append({
                "request_id": req_id,
                "type": "cached",
                "tokens": scheduler_output.num_scheduled_tokens.get(req_id, 0),
            })

        record = {
            "source": "dispatch_logger",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
            "pid": self._pid,
            "rank": self._rank,
            "step": self._step_counter,
            "phase": "prefill" if is_prefill else "decode",
            "total_num_scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
            "requests": requests,
        }
        self._fp.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fp.flush()
        self._step_counter += 1

    def close(self) -> None:
        if self._fp and not self._fp.closed:
            self._fp.close()
