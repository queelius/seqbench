#include "test_util.hpp"
#include "seqbench/experiment.hpp"
#include <cstdio>
#include <fstream>
#include <string>

using namespace seqbench;

static bool has(const std::string& s, const std::string& sub) {
  return s.find(sub) != std::string::npos;
}

static RunRecord sample() {
  RunRecord r;
  r.arch = "t";
  r.version = "v1";
  r.seed = 7;
  r.config["dim"] = JsonValue::n(64);
  r.config["rule"] = JsonValue::s("delta");
  r.corpus_name = "toy";
  r.corpus_bytes = 100;
  r.results["bpb"] = 3.5;
  return r;
}

static void test_to_json_shape() {
  RunRecord r = sample();
  r.timestamp = "2026-01-01T00:00:00Z";
  r.git_sha = "abc1234";
  std::string s = to_json(r);
  CHECK(s.front() == '{' && s.back() == '}');
  CHECK(has(s, "\"arch\":\"t\""));
  CHECK(has(s, "\"seed\":7"));
  CHECK(has(s, "\"git_sha\":\"abc1234\""));
  CHECK(has(s, "\"dim\":64"));          // numbers are unquoted
  CHECK(has(s, "\"rule\":\"delta\""));  // strings are quoted
  CHECK(has(s, "\"corpus\":{\"name\":\"toy\",\"bytes\":100}"));
  CHECK(has(s, "\"bpb\":3.5"));
  CHECK(s.find('\n') == std::string::npos);  // single line
}

static void test_string_escaping() {
  RunRecord r = sample();
  r.config["note"] = JsonValue::s("x\"y");  // a quote must be escaped
  std::string s = to_json(r);
  CHECK(has(s, "x\\\"y"));
}

static void test_fill_provenance() {
  RunRecord r = sample();
  fill_provenance(r);
  CHECK(!r.git_sha.empty());                 // real SHA or "unknown"
  CHECK(!r.timestamp.empty());
  CHECK(r.timestamp.back() == 'Z');          // ISO-8601 UTC
}

static void test_append_two_lines() {
  const std::string path = "/tmp/seqbench_exp_test.jsonl";
  std::remove(path.c_str());
  RunRecord a = sample();
  RunRecord b = sample();
  append_record(a, path);
  append_record(b, path);
  std::ifstream f(path);
  int lines = 0;
  std::string line, last;
  while (std::getline(f, line)) { ++lines; last = line; }
  f.close();
  CHECK(lines == 2);
  CHECK(has(last, "\"arch\":\"t\""));
  std::remove(path.c_str());
}

int main() {
  RUN(test_to_json_shape);
  RUN(test_string_escaping);
  RUN(test_fill_provenance);
  RUN(test_append_two_lines);
  return test_summary();
}
