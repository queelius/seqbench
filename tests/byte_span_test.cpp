#include "test_util.hpp"
#include "seqbench/byte_span.hpp"

using namespace seqbench;

static void test_byte_span_basics() {
  const uint8_t buf[5] = {10, 20, 30, 40, 50};
  ByteSpan s{buf, 5};
  CHECK(s.len == 5);
  CHECK(s[0] == 10);
  CHECK(s[4] == 50);
  ByteSpan sub = s.subspan(1, 3);
  CHECK(sub.len == 3);
  CHECK(sub[0] == 20);
  CHECK(sub[2] == 40);
}

int main() {
  RUN(test_byte_span_basics);
  return test_summary();
}
