PYTHON ?= python
CARGO ?= cargo

.PHONY: build-rust test-rust test verify doctor export serve

build-rust:
	$(PYTHON) scripts/build_rust.py

test-rust:
	$(CARGO) test --manifest-path rust_core/Cargo.toml --no-default-features

test:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py" -v

verify:
	$(PYTHON) tests/test_v8_verification.py

doctor:
	$(PYTHON) -m src.cli doctor

export:
	$(PYTHON) -m src.cli export --mock --output checkpoints/spatial_projector_ts.pt

serve:
	$(PYTHON) -m src.cli serve --mock --host 127.0.0.1 --port 8000
