#pragma once
#include <cstddef>
#include <map>
#include <string>

namespace seqbench {

// A JSON scalar: number or string (enough for config values).
struct JsonValue {
  enum Type { Number, String };
  Type type = Number;
  double num = 0.0;
  std::string str;
  static JsonValue n(double v) { JsonValue j; j.type = Number; j.num = v; return j; }
  static JsonValue s(std::string v) { JsonValue j; j.type = String; j.str = std::move(v); return j; }
};

// A provenance-complete record of one experiment run, serialized as one JSON line.
struct RunRecord {
  std::string arch;
  std::string version;
  std::string timestamp;   // ISO-8601 UTC; auto-filled by fill_provenance if empty
  std::string git_sha;     // short SHA; auto-filled by fill_provenance if empty
  long seed = 0;
  std::map<std::string, JsonValue> config;
  std::string corpus_name;
  std::size_t corpus_bytes = 0;
  std::map<std::string, double> results;
};

// Serialize to a single-line JSON string (no trailing newline).
std::string to_json(const RunRecord& r);

// Fill git_sha (via `git rev-parse --short HEAD`) and timestamp (ISO-8601 UTC) if empty.
void fill_provenance(RunRecord& r);

// Append one record as a JSON line to `path` (creating the parent dir if needed).
// Fills provenance first. Throws std::runtime_error on I/O failure.
void append_record(RunRecord& r, const std::string& path = "runs/results.jsonl");

}  // namespace seqbench
