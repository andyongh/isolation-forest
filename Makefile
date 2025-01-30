CC = gcc
CFLAGS = -Wall -Wextra -Werror -O3 -lm -lpthread
PYTHON = python3

.PHONY: all generate_data build run clean

all: generate_data build run

generate_data:
	@echo "Generating test data..."
	@$(PYTHON) generate_data.py

build: isolation_forest.c
	@echo "Building C implementation..."
	@$(CC) $(CFLAGS) -o isolation_forest isolation_forest.c

run: isolation_forest
	@echo "Running isolation forest..."
	@./isolation_forest

clean:
	@rm -f isolation_forest test_data.csv c_scores.txt