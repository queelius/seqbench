#include "seqbench/experiment.hpp"
#include <sys/stat.h>
#include <cstdio>
#include <ctime>
#include <stdexcept>

namespace seqbench {

namespace {

std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char b[8];
          std::snprintf(b, sizeof(b), "\\u%04x", static_cast<unsigned>(c));
          out += b;
        } else {
          out += c;
        }
    }
  }
  return out;
}

std::string num_str(double v) {
  char b[32];
  std::snprintf(b, sizeof(b), "%.10g", v);
  return std::string(b);
}

std::string jval(const JsonValue& v) {
  if (v.type == JsonValue::Number) return num_str(v.num);
  return "\"" + json_escape(v.str) + "\"";
}

}  // namespace

std::string to_json(const RunRecord& r) {
  std::string o = "{";
  o += "\"arch\":\"" + json_escape(r.arch) + "\",";
  o += "\"version\":\"" + json_escape(r.version) + "\",";
  o += "\"timestamp\":\"" + json_escape(r.timestamp) + "\",";
  o += "\"git_sha\":\"" + json_escape(r.git_sha) + "\",";
  o += "\"seed\":" + std::to_string(r.seed) + ",";
  o += "\"config\":{";
  bool first = true;
  for (const auto& kv : r.config) {
    if (!first) o += ",";
    first = false;
    o += "\"" + json_escape(kv.first) + "\":" + jval(kv.second);
  }
  o += "},";
  o += "\"corpus\":{\"name\":\"" + json_escape(r.corpus_name) +
       "\",\"bytes\":" + std::to_string(r.corpus_bytes) + "},";
  o += "\"results\":{";
  first = true;
  for (const auto& kv : r.results) {
    if (!first) o += ",";
    first = false;
    o += "\"" + json_escape(kv.first) + "\":" + num_str(kv.second);
  }
  o += "}}";
  return o;
}

void fill_provenance(RunRecord& r) {
  if (r.git_sha.empty()) {
    r.git_sha = "unknown";
    std::FILE* p = ::popen("git rev-parse --short HEAD 2>/dev/null", "r");
    if (p) {
      char buf[64];
      if (std::fgets(buf, sizeof(buf), p)) {
        std::string s(buf);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        if (!s.empty()) r.git_sha = s;
      }
      ::pclose(p);
    }
  }
  if (r.timestamp.empty()) {
    std::time_t t = std::time(nullptr);
    std::tm tm_utc;
    ::gmtime_r(&t, &tm_utc);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
    r.timestamp = buf;
  }
}

void append_record(RunRecord& r, const std::string& path) {
  fill_provenance(r);
  std::size_t slash = path.find_last_of('/');
  if (slash != std::string::npos) {
    std::string dir = path.substr(0, slash);
    ::mkdir(dir.c_str(), 0755);  // ignore errors (e.g., dir already exists)
  }
  std::FILE* f = std::fopen(path.c_str(), "a");
  if (!f) throw std::runtime_error("append_record: cannot open " + path);
  std::string line = to_json(r);
  if (std::fprintf(f, "%s\n", line.c_str()) < 0) {
    std::fclose(f);
    throw std::runtime_error("append_record: write failed for " + path);
  }
  if (std::fclose(f) != 0)
    throw std::runtime_error("append_record: close failed for " + path);
}

}  // namespace seqbench
