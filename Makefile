CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra
LDFLAGS ?=

APP := NNFS_Diploma
APP_SRC := NNFS_Diploma.cpp

TEST_BIN := tests
TEST_SRC := tests.cpp

GCOVR ?= gcovr --exclude-noncode-lines
GCOVR_EXCLUDES := --exclude 'doctest.h' --exclude 'fashion_mnist/.*'
COVERAGE_FLAGS := -O0 -g --coverage

.PHONY: all run test coverage all-in-one clean clean-coverage

all: $(APP)

$(APP): $(APP_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TEST_BIN): $(TEST_SRC) NNFS_Diploma.cpp doctest.h
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_SRC) $(LDFLAGS)

run: $(APP)
	./$(APP)

test: $(TEST_BIN)
	./$(TEST_BIN)

coverage: clean-coverage
	$(MAKE) CXXFLAGS="$(filter-out -O%,$(CXXFLAGS)) $(COVERAGE_FLAGS)" LDFLAGS="$(LDFLAGS)" $(TEST_BIN)
	./$(TEST_BIN)
	PATH="$$HOME/.local/bin:$$PATH" $(GCOVR) -r . $(GCOVR_EXCLUDES)
	rm -f *.gcda *.gcno

all-in-one: run coverage
	@echo "Completed run, test, coverage"

clean: clean-coverage
	rm -f $(APP)

clean-coverage:
	rm -f $(TEST_BIN)
	rm -f *.gcda *.gcno *.gcov coverage.txt coverage.html coverage.xml coverage.json
