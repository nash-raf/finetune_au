# Archived Legacy Scripts

This folder contains the old preparation and evaluation scripts from the original repository.

They are archived here on purpose and should not be used with the corrected local MEAD pipeline.

Why they are archived:

- `make_train_data_jsonl.py` leaks the emotion label into the user prompt.
- `make_train_data_jsonl.py` uses the older assistant output format.
- `make_simple_json.py` applies an additional 5x temporal reduction meant for older dense labels.
- `calculate_au_loss.py` evaluates only the first line/frame of each file.

Use the corrected local pipeline instead:

- `/home/user/D/au_extract.py`
- `/home/user/D/prepare_mead.py`
- `/home/user/D/check_token_lengths.py`
- `/home/user/D/evaluate_au_sequence.py`
