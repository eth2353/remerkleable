import os
import time
import statistics
import pytest

from remerkleable.basic import uint64, uint8
from remerkleable.bitfields import Bitlist
from remerkleable.byte_arrays import Bytes32, Bytes48, ByteList, ByteVector
from remerkleable.complex import Container, List


# -----------------------------
# Tunables (via env vars)
# -----------------------------
# Make it block-like but not insane for unit test runs.
ATTESTATIONS = int(os.getenv("REMERKLEABLE_PERF_ATTESTATIONS", "256"))
TXS = int(os.getenv("REMERKLEABLE_PERF_TXS", "256"))
TX_BYTES = int(os.getenv("REMERKLEABLE_PERF_TX_BYTES", "256"))  # per tx payload
COMMITTEE_BITS = int(os.getenv("REMERKLEABLE_PERF_COMMITTEE_BITS", "512"))

REPEATS = int(os.getenv("REMERKLEABLE_PERF_REPEATS", "10"))
WARMUP = int(os.getenv("REMERKLEABLE_PERF_WARMUP", "2"))

# Optional strict bounds off by default to avoid flaky CI
ASSERT = os.getenv("REMERKLEABLE_PERF_ASSERT", "").lower() in ("1", "true", "yes")
MAX_SERIALIZE = float(os.getenv("REMERKLEABLE_PERF_MAX_SERIALIZE", "inf"))
MAX_DESERIALIZE = float(os.getenv("REMERKLEABLE_PERF_MAX_DESERIALIZE", "inf"))
MAX_HTR = float(os.getenv("REMERKLEABLE_PERF_MAX_HTR", "inf"))


# -----------------------------
# Helpers
# -----------------------------
def _time_many(fn, repeats: int) -> list[float]:
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def _make_bytes(n: int, seed: int) -> bytes:
    # deterministic non-zero-ish payload
    return bytes(((i * 73 + seed * 29 + 17) & 0xFF) for i in range(n))


def _make_bitlist(bitlen: int, seed: int) -> Bitlist:
    # deterministic bit pattern
    return Bitlist[bitlen]([((i + seed) % 5) == 0 for i in range(bitlen)])


# -----------------------------
# Block-like SSZ types
# -----------------------------
class Root(Bytes32):
    pass


class Hash32(Bytes32):
    pass


class BLSSignature(Bytes48):
    pass


class ExecutionAddress(ByteVector[20]):
    pass


class Transaction(ByteList[TX_BYTES]):
    pass


class AttestationData(Container):
    # Similar to consensus attestation_data; fixed-size
    slot: uint64
    index: uint64
    beacon_block_root: Root
    source_epoch: uint64
    source_root: Root
    target_epoch: uint64
    target_root: Root


class Attestation(Container):
    # Similar to consensus attestation; includes dynamic aggregation bits
    aggregation_bits: Bitlist[COMMITTEE_BITS]
    data: AttestationData
    signature: BLSSignature


class ExecutionPayload(Container):
    # “Execution layer payload-ish”: has transactions list (dynamic)
    parent_hash: Hash32
    fee_recipient: ExecutionAddress
    state_root: Bytes32
    receipts_root: Bytes32
    logs_bloom: ByteVector[256]      # smaller than mainnet 256 bytes bloom? (bloom is 256 bytes)
    prev_randao: Bytes32
    block_number: uint64
    gas_limit: uint64
    gas_used: uint64
    timestamp: uint64
    extra_data: ByteList[32]         # small extra_data
    base_fee_per_gas: Bytes32        # stand-in for uint256
    block_hash: Hash32
    transactions: List[Transaction, TXS]


class BeaconBlockBody(Container):
    randao_reveal: BLSSignature
    graffiti: Bytes32
    attestations: List[Attestation, ATTESTATIONS]
    execution_payload: ExecutionPayload


class BeaconBlock(Container):
    slot: uint64
    proposer_index: uint64
    parent_root: Root
    state_root: Root
    body: BeaconBlockBody


# -----------------------------
# Value construction
# -----------------------------
def make_block() -> BeaconBlock:
    # Attestations
    atts = []
    for i in range(ATTESTATIONS):
        att_data = AttestationData(
            slot=uint64(12345 + i),
            index=uint64(i % 64),
            beacon_block_root=Root(_make_bytes(32, i)),
            source_epoch=uint64(100),
            source_root=Root(_make_bytes(32, i + 1)),
            target_epoch=uint64(101),
            target_root=Root(_make_bytes(32, i + 2)),
        )
        att = Attestation(
            aggregation_bits=_make_bitlist(COMMITTEE_BITS, i),
            data=att_data,
            signature=BLSSignature(_make_bytes(48, i)),
        )
        atts.append(att)

    # Transactions
    txs = [Transaction(_make_bytes(TX_BYTES, i)) for i in range(TXS)]

    payload = ExecutionPayload(
        parent_hash=Hash32(_make_bytes(32, 1)),
        fee_recipient=ExecutionAddress(_make_bytes(20, 2)),
        state_root=Bytes32(_make_bytes(32, 3)),
        receipts_root=Bytes32(_make_bytes(32, 4)),
        logs_bloom=ByteVector[256](_make_bytes(256, 5)),
        prev_randao=Bytes32(_make_bytes(32, 6)),
        block_number=uint64(1),
        gas_limit=uint64(30_000_000),
        gas_used=uint64(15_000_000),
        timestamp=uint64(1_700_000_000),
        extra_data=ByteList[32](_make_bytes(32, 7)),
        base_fee_per_gas=Bytes32(_make_bytes(32, 8)),
        block_hash=Hash32(_make_bytes(32, 9)),
        transactions=txs,
    )

    body = BeaconBlockBody(
        randao_reveal=BLSSignature(_make_bytes(48, 10)),
        graffiti=Bytes32(_make_bytes(32, 11)),
        attestations=atts,
        execution_payload=payload,
    )

    return BeaconBlock(
        slot=uint64(12345),
        proposer_index=uint64(42),
        parent_root=Root(_make_bytes(32, 12)),
        state_root=Root(_make_bytes(32, 13)),
        body=body,
    )


# -----------------------------
# The perf test
# -----------------------------
@pytest.mark.perf
def test_perf_block_like_serialize_deserialize_hash_tree_root():
    block = make_block()
    encoded = block.encode_bytes()
    assert len(encoded) > 100_000, f"encoded too small: {len(encoded)} bytes"

    # Warmup
    for _ in range(WARMUP):
        _ = block.encode_bytes()
        _ = BeaconBlock.decode_bytes(encoded)
        # For HTR, avoid measuring a cached result by hashing a freshly decoded block
        _ = BeaconBlock.decode_bytes(encoded).hash_tree_root()

    ser_times = _time_many(lambda: block.encode_bytes(), REPEATS)
    deser_times = _time_many(lambda: BeaconBlock.decode_bytes(encoded), REPEATS)
    htr_times = _time_many(lambda: BeaconBlock.decode_bytes(encoded).hash_tree_root(), REPEATS)

    ser_med = statistics.median(ser_times)
    deser_med = statistics.median(deser_times)
    htr_med = statistics.median(htr_times)

    print(
        f"\nBlock-like container encoded={len(encoded)} bytes "
        f"(atts={ATTESTATIONS}, txs={TXS}, tx_bytes={TX_BYTES}, committee_bits={COMMITTEE_BITS})\n"
        f"serialize median:   {ser_med:.6f}s (min {min(ser_times):.6f}s)\n"
        f"deserialize median: {deser_med:.6f}s (min {min(deser_times):.6f}s)\n"
        f"htr median:         {htr_med:.6f}s (min {min(htr_times):.6f}s)\n"
    )

    if ASSERT:
        assert ser_med <= MAX_SERIALIZE, f"serialize median {ser_med:.6f}s > {MAX_SERIALIZE}s"
        assert deser_med <= MAX_DESERIALIZE, f"deserialize median {deser_med:.6f}s > {MAX_DESERIALIZE}s"
        assert htr_med <= MAX_HTR, f"htr median {htr_med:.6f}s > {MAX_HTR}s"
