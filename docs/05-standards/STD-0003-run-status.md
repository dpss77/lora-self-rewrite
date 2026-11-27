# Run status vocabulary

Exactly these statuses (enforced by `manifest.validate_manifest` and
`scripts/finalize_run.py`):

```
planned -> running -> preempted / failed / withdrawn / submitted
        -> validated-null / validated-positive -> archived
```

| Status | Meaning |
| :--- | :--- |
| `planned` | manifest written; no execution yet |
| `running` | executing |
| `preempted` | infrastructure interrupted; **not** a scientific result |
| `failed` | run failed for a non-preemption reason |
| `withdrawn` | model chose `WITHDRAW`; no validation expected |
| `submitted` | adapter persisted and reserved validation ran |
| `validated-null` | held-out gate complete, not significant |
| `validated-positive` | held-out gate complete, significant (`p<0.05`) |
| `archived` | terminal, retained for the record |

`resumed` is a manifest flag/field, not a terminal status. A `preempted` run may
return to `running` after a correct resume (pass budget + transcript restored).

## Allowed transitions

```
planned        -> running | preempted | failed | withdrawn
running        -> preempted | failed | withdrawn | submitted
preempted      -> running | failed | withdrawn | submitted
submitted      -> validated-null | validated-positive | archived
withdrawn      -> archived
validated-*    -> archived
failed         -> archived
archived       -> (none)
```
