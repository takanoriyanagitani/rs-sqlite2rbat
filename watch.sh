#!/bin/sh

cargo \
	watch \
	--shell ./check.sh \
	--watch ./src \
	--watch ./Cargo.toml
