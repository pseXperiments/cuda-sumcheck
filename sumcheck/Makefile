CARGO = cargo

all: build doc

build:
	@$(CARGO) build --release

doc:
	@$(CARGO) doc

check: build test

test:
	@$(CARGO) test --release

bench:
	@$(CARGO) bench

clean:
	@$(CARGO) clean

.PHONY: all build doc check test bench clean
