CARGO = cargo

all: build

build:
	git submodule init
	git submodule update
	@$(CARGO) build --release

doc:
	@$(CARGO) doc

check: build test

test:
	@$(CARGO) test --release -- --test-threads=1

bench:
	@$(CARGO) bench

clean:
	@$(CARGO) clean

.PHONY: all build doc check test bench clean
