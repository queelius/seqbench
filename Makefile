CXX ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -Iinclude

LIB_SRC := $(wildcard src/*.cpp) $(wildcard models/*.cpp)
LIB_OBJ := $(LIB_SRC:.cpp=.o)
TOOLS   := $(patsubst %.cpp,%,$(wildcard tools/*.cpp))
TESTS   := $(patsubst %.cpp,%,$(wildcard tests/*.cpp))

all: $(TOOLS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TOOLS): %: %.cpp $(LIB_OBJ)
	$(CXX) $(CXXFLAGS) $< $(LIB_OBJ) -o $@

$(TESTS): %: %.cpp $(LIB_OBJ)
	$(CXX) $(CXXFLAGS) $< $(LIB_OBJ) -o $@

test: $(TESTS)
	@fail=0; for t in $(TESTS); do echo "== $$t =="; ./$$t || fail=1; done; \
	 if [ $$fail -ne 0 ]; then echo "TESTS FAILED"; exit 1; fi; echo "ALL TESTS PASSED"

data/enwik8:
	mkdir -p data
	cd data && curl -L -o enwik8.zip http://mattmahoney.net/dc/enwik8.zip && unzip enwik8.zip && rm -f enwik8.zip

clean:
	rm -f $(LIB_OBJ) $(TOOLS) $(TESTS)

.PHONY: all test clean
