# Dataset manifest checklist

**Not audited yet.** Copy `data_manifest.example.json` to a local `data/data_manifest.json` after obtaining access. Do not track raw dataset files or the local manifest containing potentially sensitive paths.

Required before any real benchmark:
1. Confirm source card, access conditions, licence, revision and exact downloaded artifacts.
2. Record schema, encoding, class names and **verified** label ID mapping.
3. Verify official train/dev/test split membership; compute class distribution, nulls and duplicates, including cross-split near duplicates.
4. Preserve raw text and example IDs; explain any preprocessing without silently changing the benchmark.
5. Record limitations and audit owner decision. Mark UNKNOWN/BLOCKED where provenance or permission is unclear.

The manifest template intentionally contains `null` values for unverified dataset facts.
