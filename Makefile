CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
LDFLAGS ?=

APP := NNFS_Diploma
APP_SRC := NNFS_Diploma.cpp

TEST_BIN := tests
TEST_SRC := tests.cpp

GCOVR ?= gcovr
GCOVR_EXCLUDES := --exclude 'doctest.h'
COVERAGE_FLAGS := -O0 -g --coverage

.PHONY: all run test clean coverage all-in-one

all: $(APP)

$(APP): $(APP_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TEST_BIN): $(TEST_SRC) NNFS_Diploma.cpp doctest.h
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_SRC) $(LDFLAGS)

run: $(APP)
	./$(APP)

test: $(TEST_BIN)
	./$(TEST_BIN)

coverage: clean
	$(MAKE) CXXFLAGS="$(filter-out -O%,$(CXXFLAGS)) $(COVERAGE_FLAGS)" LDFLAGS="$(LDFLAGS)" $(TEST_BIN)
	./$(TEST_BIN)
	PATH="$$HOME/.local/bin:$$PATH" $(GCOVR) -r . $(GCOVR_EXCLUDES)

all-in-one: run test coverage
	@echo "Completed run, test, coverage"

clean:
	rm -f $(APP) $(TEST_BIN)
	rm -f *.gcda *.gcno coverage.txt
