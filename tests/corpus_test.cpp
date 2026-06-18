#include "test_util.hpp"
#include "seqbench/corpus.hpp"
#include <cstdio>
#include <fstream>
#include <string>

using namespace seqbench;

static void test_toy_corpus_deterministic() {
  ByteSpan a = toy_corpus();
  ByteSpan b = toy_corpus();
  CHECK(a.len > 0);
  CHECK(a.data == b.data);
  CHECK(a.len == b.len);
}

static void test_mmap_roundtrip_and_splits() {
  const std::string path = "/tmp/seqbench_corpus_test.bin";
  {
    std::ofstream f(path, std::ios::binary);
    for (int i = 0; i < 1000; ++i) f.put(char(i % 256));
  }
  Corpus c(path);
  CHECK(c.bytes().len == 1000);
  CHECK(c.bytes()[0] == 0);
  CHECK(c.bytes()[255] == 255);
  CHECK(c.train().len == 900);
  CHECK(c.val().len == 50);
  CHECK(c.test().len == 50);
  // val starts right after train.
  CHECK(c.val()[0] == c.bytes()[900]);
  std::remove(path.c_str());
}

int main() {
  RUN(test_toy_corpus_deterministic);
  RUN(test_mmap_roundtrip_and_splits);
  return test_summary();
}
