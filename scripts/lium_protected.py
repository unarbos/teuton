"""Hard-coded list of Lium pod IDs that this repo is forbidden from touching.

Any script that mutates pods (switch-template, edit, terminate) MUST import
`PROTECTED_POD_IDS` and skip them.
"""
from __future__ import annotations

# zesty-wolf-ab  (Allan Affine,         8x B300, ssh root@95.133.253.18 -p 10300)
# noble-raven-29 (Affine Team Evals,    8x B300, ssh root@31.22.104.217 -p 10300)
PROTECTED_POD_IDS = frozenset(
    {
        "71950490-1be4-40df-88f7-ef5a6a27ec71",
        "3c87b2fe-4e20-4d7c-9802-42b3796bb499",
    }
)

# Same set keyed by SSH host so callers that only have a host string can also
# guard against accidental mutations.
PROTECTED_SSH_HOSTS = frozenset({"95.133.253.18", "31.22.104.217"})


def assert_unprotected(*, pod_id: str | None = None, ssh_host: str | None = None) -> None:
    if pod_id and pod_id in PROTECTED_POD_IDS:
        raise RuntimeError(f"refusing to touch protected pod_id={pod_id}")
    if ssh_host and ssh_host in PROTECTED_SSH_HOSTS:
        raise RuntimeError(f"refusing to touch protected ssh host={ssh_host}")
