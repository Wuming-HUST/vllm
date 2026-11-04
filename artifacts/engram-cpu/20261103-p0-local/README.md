# P0 local validation — 2026-11-03

Both repositories use `feat/engram-cpu-v41-integration` based on their paired
`v0.30.0` tags. This receipt covers the synchronous CPU operator implementation.
It does not accept a serving backend or establish a performance improvement.

The [manifest](manifest.json) records full tag commits, source SHA256 hashes,
local runtime versions, commands and exit codes. Native build/install completed
in the session; raw build terminal output was not saved. Its installed binary
hash and build commands are recorded. The local build directory is retained.

| Check | Result | Evidence |
| --- | --- | --- |
| macOS arm64 host build/install | Passed | Manifest; `build/engram-cpu` cache |
| Native import and numerical contracts | 21 passed | [Final pytest](pytest-final.txt) |
| SVE2 translation unit | Cross compile passed, no execution | [Object type](sve-object.txt) |
| Python lint and formatting | Passed | Manifest and individual text receipts |
| C++ formatting and Ascend syntax | Passed | Manifest |
| Ascend runtime UT | Import failed, exit 4 | [Failure](ascend-ut-attempt.txt) |
| ARM target | SSH banner timeout, state unknown | [Probe transcription](remote-probe.txt) |

The first numerical run had four BF16 oracle dtype mismatches. The reference
was corrected to expect the API's BF16 output; the final run passed all cases.
The [original failure](pytest-attempt1.txt) is preserved. The final run also
checks the extension's Python initializer and uses PyTorch's native E8M0 decoder
for extreme-scale reference values.

TP1/2/4/8 checks reconstruct synthetic shards in one process. No distributed
collectives or device execution were tested. Linux native build/SVE2 execution,
complete wheel installation, Ascend runtime tests, model rows/logits/generation,
graph integration and performance remain pending. The paired runtime dependencies
are not fully installed in this local operator environment; Ascend UT failed
while importing `regex` through the parent conftest.

The original device UVA lookup remains the default. Do not enable CPU serving
from this receipt. Continue with P1's eager transfer and consumer integration
after the target and numerical parity checks in the integration plan.
