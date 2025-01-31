CC = gcc
CFLAGS = -fPIC -Wall
LDFLAGS = -lpthread
TARGET_LIB = libiforest.so
TEST_EXEC = test
PYTHON = python3

.PHONY: all clean data

all: $(TARGET_LIB) $(TEST_EXEC)

$(TARGET_LIB): isolation_forest.c
	$(CC) $(CFLAGS) -shared -o $@ $< $(LDFLAGS)

$(TEST_EXEC): test.c $(TARGET_LIB)
	$(CC) $(CFLAGS) -o $@ test.c -L. -liforest -g $(LDFLAGS)

data:
	@echo "Generating test data..."
	@$(PYTHON) generate_data.py

clean:
	rm -f $(TARGET_LIB) $(TEST_EXEC) *.o c_scores.txt test_data.csv